"""Job routes: create / list / get / resume + the index HTML page."""

from __future__ import annotations

import asyncio
import logging
import secrets
import string
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse

from ..pipeline.artifacts import ChaptersIndex
from ..pipeline.manifest import (
    AspectRatio,
    Depth,
    JobConfig,
    JobManifest,
    LengthPreset,
    LipSyncConfig,
    NarrationConfig,
    Profile,
    ProvidersConfig,
)
from ..state import JobRecord, JobRepository, JobStatus
from .deps import JobRunner
from .schemas import (
    ChapterSummary,
    HealthResponse,
    JobCreatedResponse,
    JobListResponse,
    JobSummary,
)

_logger = logging.getLogger(__name__)
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MiB cap; bigger PDFs need explicit override


def build_job_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        repo: JobRepository = request.app.state.repo
        jobs = [_summarize(record) for record in repo.list(limit=20)]
        response = request.app.state.templates.TemplateResponse(
            request, "index.html", {"jobs": jobs}
        )
        return cast(HTMLResponse, response)

    @router.get("/jobs/{job_id}", response_class=HTMLResponse, include_in_schema=False)
    async def job_page(job_id: str, request: Request) -> HTMLResponse:
        record = _require_job(request, job_id)
        manifest = _load_manifest(request, job_id)
        chapters = _load_chapters(request, job_id)
        summary = _summarize(record, manifest=manifest, chapters=chapters)
        response = request.app.state.templates.TemplateResponse(
            request, "job.html", {"job": summary}
        )
        return cast(HTMLResponse, response)

    # ----- JSON API ----------------------------------------------------------

    @router.get("/api/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        from .. import __version__

        return HealthResponse(
            version=__version__,
            providers=request.app.state.config_overrides.get("providers", {}),
        )

    @router.get("/api/jobs", response_model=JobListResponse)
    async def list_jobs(request: Request) -> JobListResponse:
        repo: JobRepository = request.app.state.repo
        return JobListResponse(jobs=[_summarize(r) for r in repo.list(limit=100)])

    @router.get("/api/jobs/{job_id}", response_model=JobSummary)
    async def get_job(job_id: str, request: Request) -> JobSummary:
        record = _require_job(request, job_id)
        manifest = _load_manifest(request, job_id)
        chapters = _load_chapters(request, job_id)
        return _summarize(record, manifest=manifest, chapters=chapters)

    @router.post(
        "/api/jobs",
        response_model=JobCreatedResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_job(
        request: Request,
        pdf: UploadFile,
        anime_style: str = Form("shounen-bright"),
        voice_id: str = Form(...),
        language: str = Form("en-US"),
        speed: float = Form(1.0),
        depth: Depth = Form("undergraduate"),
        length_preset: LengthPreset = Form("standard"),
        minutes_per_topic: float | None = Form(None),
        aspect_ratio: AspectRatio = Form("16:9"),
        profile: Profile = Form("default"),
        lipsync_enabled: bool = Form(False),
    ) -> JobCreatedResponse:
        if pdf.content_type not in {"application/pdf", "application/octet-stream", None}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"unexpected content type: {pdf.content_type!r}",
            )

        runner: JobRunner = request.app.state.runner
        repo: JobRepository = request.app.state.repo
        data_dir: Path = request.app.state.settings.data_dir
        providers: ProvidersConfig = request.app.state.config_overrides["providers_obj"]
        default_lipsync_provider = providers.lipsync

        job_id = _generate_job_id()
        job_dir = data_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        source_pdf = job_dir / "source.pdf"
        await _stream_upload(pdf, source_pdf)

        config = JobConfig(
            anime_style=anime_style,
            narration=NarrationConfig(voice_id=voice_id, language=language, speed=speed),
            depth=depth,
            length_preset=length_preset,
            minutes_per_topic=minutes_per_topic,
            aspect_ratio=aspect_ratio,
            profile=profile,
            providers=providers,
            lipsync=LipSyncConfig(
                enabled=lipsync_enabled,
                provider=default_lipsync_provider,
            ),
        )

        manifest = JobManifest.for_new_job(job_id=job_id, config=config)
        manifest.save(job_dir / "manifest.json")
        repo.create_job(
            job_id=job_id,
            source_pdf=source_pdf,
            data_dir=data_dir,
            config=config.model_dump(mode="json"),
        )
        runner.start(job_id=job_id)

        return JobCreatedResponse(
            job_id=job_id,
            status="running",
            events_url=f"/api/jobs/{job_id}/events",
            job_url=f"/jobs/{job_id}",
        )

    @router.post("/api/jobs/{job_id}/resume", response_model=JobCreatedResponse)
    async def resume_job(job_id: str, request: Request) -> JobCreatedResponse:
        record = _require_job(request, job_id)
        if record.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            if record.status == JobStatus.RUNNING and request.app.state.runner.is_running(job_id):
                # Already running; return its bus.
                return JobCreatedResponse(
                    job_id=job_id,
                    status="running",
                    events_url=f"/api/jobs/{job_id}/events",
                    job_url=f"/jobs/{job_id}",
                )
            if record.status == JobStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="job already completed",
                )

        if _load_manifest(request, job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="manifest.json missing or corrupt; cannot resume",
            )
        request.app.state.runner.start(job_id=job_id)
        return JobCreatedResponse(
            job_id=job_id,
            status="running",
            events_url=f"/api/jobs/{job_id}/events",
            job_url=f"/jobs/{job_id}",
        )

    return router


# ------------------------------------------------------------------ helpers


def _require_job(request: Request, job_id: str) -> JobRecord:
    repo: JobRepository = request.app.state.repo
    record = repo.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return record


def _load_manifest(request: Request, job_id: str) -> JobManifest | None:
    data_dir: Path = request.app.state.settings.data_dir
    manifest_path = data_dir / "jobs" / job_id / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return JobManifest.from_path(manifest_path)
    except Exception as exc:
        _logger.warning("failed to load manifest for %s: %s", job_id, exc)
        return None


def _summarize(
    record: JobRecord,
    *,
    manifest: JobManifest | None = None,
    chapters: list[ChapterSummary] | None = None,
) -> JobSummary:
    stages: dict[str, str] = {}
    if manifest is not None:
        stages = {name: state.status.value for name, state in manifest.stages.items()}
    return JobSummary(
        job_id=record.job_id,
        status=record.status.value,
        current_stage=record.current_stage,
        created_at=record.created_at,
        updated_at=record.updated_at,
        source_pdf=record.source_pdf,
        error_message=record.error_message,
        stages=stages,  # type: ignore[arg-type]
        chapters=chapters or [],
    )


def _load_chapters(request: Request, job_id: str) -> list[ChapterSummary]:
    """Read ``chapters/index.json`` if present and return URL-rooted summaries.

    Returns an empty list when the index is missing (older jobs, or jobs that
    failed before assembly). A corrupt index is logged and treated as missing
    rather than failing the whole detail page.
    """

    data_dir: Path = request.app.state.settings.data_dir
    index_path = data_dir / "jobs" / job_id / "chapters" / "index.json"
    if not index_path.is_file():
        return []
    try:
        index = ChaptersIndex.from_path(index_path)
    except Exception as exc:
        _logger.warning("failed to load chapters/index.json for %s: %s", job_id, exc)
        return []
    summaries: list[ChapterSummary] = []
    for record in index.items:
        mp4_name = Path(record.file).name
        srt_name = Path(record.srt_file).name
        summaries.append(
            ChapterSummary(
                order=record.order,
                topic_id=record.topic_id,
                duration_seconds=record.duration_seconds,
                mp4_url=f"/api/jobs/{job_id}/files/{mp4_name}",
                srt_url=f"/api/jobs/{job_id}/files/{srt_name}",
            )
        )
    return summaries


_ALPHABET = string.ascii_uppercase + string.digits


def _generate_job_id() -> str:
    """26-char alphanumeric job id. Sufficiently unique for one user's jobs;
    we don't need ULID's monotonic-time guarantee.
    """

    return "".join(secrets.choice(_ALPHABET) for _ in range(26))


async def _stream_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an upload to disk with a size cap. ``UploadFile.read`` is async.

    Disk writes are hopped through ``asyncio.to_thread`` so a slow filesystem
    (NFS, sshfs) cannot stall the event loop while the upload streams.
    """

    written = 0
    chunk_size = 1024 * 1024
    handle = await asyncio.to_thread(dest.open, "wb")
    try:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes",
                )
            await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)
    await upload.close()
