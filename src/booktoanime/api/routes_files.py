"""Serve completed-job artifacts (output.mp4, output.srt) over HTTP.

Only an explicit allow-list of filenames is exposed; the route refuses any
path containing ``..`` or absolute components, mirroring the JobRelPath
validators on the artifact JSON fields.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from ..state import JobRepository

# Only files we know are safe to expose. Add new entries explicitly when
# new artifacts are surfaced in the UI.
_SERVABLE_FILES: dict[str, str] = {
    "output.mp4": "video/mp4",
    "output.srt": "text/plain; charset=utf-8",
}


def build_files_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs/{job_id}/files/{filename}")
    async def get_file(job_id: str, filename: str, request: Request) -> FileResponse:
        if filename not in _SERVABLE_FILES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"file {filename!r} not exposed",
            )

        repo: JobRepository = request.app.state.repo
        record = repo.get(job_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
            )

        data_dir: Path = request.app.state.settings.data_dir
        path = (data_dir / "jobs" / job_id / filename).resolve()
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

        return FileResponse(
            path,
            media_type=_SERVABLE_FILES[filename],
        )

    return router
