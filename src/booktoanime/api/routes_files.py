"""Serve completed-job artifacts (output.mp4, output.srt, chapter_NNN.mp4) over HTTP.

The route accepts an explicit allow-list of static filenames plus a tightly
scoped regex for per-chapter artifacts. It refuses any path containing ``..``
or absolute components, mirroring the JobRelPath validators on the artifact
JSON fields.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from ..state import JobRepository

# Top-level files we know are safe to expose. Add new entries explicitly
# when new artifacts are surfaced in the UI.
_SERVABLE_FILES: dict[str, str] = {
    "output.mp4": "video/mp4",
    "output.srt": "text/plain; charset=utf-8",
}

# Per-chapter artifacts live under ``chapters/`` inside the job dir.
_CHAPTER_FILENAME_RE = re.compile(r"^chapter_\d{3}\.(?P<ext>mp4|srt)$")
_CHAPTER_MEDIA_TYPES: dict[str, str] = {
    "mp4": "video/mp4",
    "srt": "text/plain; charset=utf-8",
}


def _resolve_artifact(filename: str) -> tuple[Path, str] | None:
    """Map a requested filename to ``(relative_subpath, media_type)``.

    ``relative_subpath`` is rooted at the job directory (e.g. ``output.mp4``
    or ``chapters/chapter_001.mp4``). Returns ``None`` if the filename is
    not in the allowed shape.
    """

    if filename in _SERVABLE_FILES:
        return Path(filename), _SERVABLE_FILES[filename]

    chapter_match = _CHAPTER_FILENAME_RE.match(filename)
    if chapter_match is not None:
        ext = chapter_match.group("ext")
        return Path("chapters") / filename, _CHAPTER_MEDIA_TYPES[ext]

    return None


def build_files_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs/{job_id}/files/{filename}")
    async def get_file(job_id: str, filename: str, request: Request) -> FileResponse:
        resolved = _resolve_artifact(filename)
        if resolved is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"file {filename!r} not exposed",
            )
        sub_path, media_type = resolved

        repo: JobRepository = request.app.state.repo
        record = repo.get(job_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
            )

        data_dir: Path = request.app.state.settings.data_dir
        path = (data_dir / "jobs" / job_id / sub_path).resolve()
        # Defense in depth: refuse any resolved path that escapes the job dir.
        job_root = (data_dir / "jobs" / job_id).resolve()
        try:
            path.relative_to(job_root)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid path"
            ) from exc

        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{filename} not produced for this job yet",
            )

        return FileResponse(path, media_type=media_type)

    return router
