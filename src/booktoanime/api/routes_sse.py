"""Server-Sent Events stream of pipeline progress.

The route subscribes to the per-job :class:`ProgressEventBus` and forwards
every event as a JSON-encoded SSE message. When the bus is closed by the
orchestrator (success or failure), we send one final ``{"kind": "done"}``
event and close the stream so the browser stops waiting.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from .deps import JobRunner


def build_sse_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> EventSourceResponse:
        runner: JobRunner = request.app.state.runner
        bus = runner.get_bus(job_id)
        if bus is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="no live event stream for this job (it may have finished or never started)",
            )

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            async with bus.subscribe() as queue:
                while True:
                    if await request.is_disconnected():
                        break
                    event = await queue.get()
                    if event is None:
                        # Sentinel: bus closed, signal end-of-stream.
                        yield {
                            "event": "done",
                            "data": json.dumps({"kind": "done"}),
                        }
                        break
                    yield {
                        "event": event.kind.value,
                        "data": json.dumps(event.to_dict()),
                    }

        return EventSourceResponse(event_stream())

    return router
