"""Server-Sent Events stream of pipeline progress.

The route subscribes to the per-job :class:`ProgressEventBus` and forwards
every event as a JSON-encoded SSE message. When the bus is closed by the
orchestrator (success or failure), we send one final ``{"kind": "done"}``
event and close the stream so the browser stops waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
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
                    # Race the queue against client disconnect so an idle
                    # subscriber doesn't hold a slot forever after the
                    # browser tab closes. ``asyncio.wait`` with
                    # FIRST_COMPLETED lets us notice either side without
                    # blocking on the other.
                    queue_task = asyncio.create_task(queue.get())
                    disconnect_task = asyncio.create_task(request.is_disconnected())
                    try:
                        done, pending = await asyncio.wait(
                            {queue_task, disconnect_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except asyncio.CancelledError:
                        queue_task.cancel()
                        disconnect_task.cancel()
                        raise

                    for task in pending:
                        task.cancel()

                    if disconnect_task in done and disconnect_task.result():
                        # If a queue item arrived simultaneously, drain its
                        # result so the task isn't garbage-collected with an
                        # unretrieved exception.
                        if queue_task in done:
                            with contextlib.suppress(Exception):
                                queue_task.result()
                        break

                    if queue_task not in done:
                        # Disconnect fired without a queue item; loop top
                        # will re-check and break above.
                        continue

                    event = queue_task.result()
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
