"""Progress event bus shared between the orchestrator, the SSE endpoint,
and the on-disk events log.

Events are written through the bus to two sinks:

1. An append-only NDJSON file (``events.log`` in the job directory) used as
   the source of truth for replay and post-mortem inspection.
2. Any number of in-memory async subscribers (the SSE route subscribes here).

The bus is intentionally tiny — no broker, no history queue beyond what
NDJSON gives us. Subscribers attach with :meth:`subscribe` and receive
every event emitted *after* their subscription via an ``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


class ProgressKind(StrEnum):
    """Kinds of events emitted during a pipeline run.

    Pure status transitions (``stage_started`` / ``stage_completed`` /
    ``stage_failed``) carry stage-level context. ``shot_completed`` /
    ``shot_failed`` carry per-shot context inside the images / audio stages.
    ``info`` is for non-actionable progress chatter.
    """

    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    SHOT_COMPLETED = "shot_completed"
    SHOT_FAILED = "shot_failed"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One observable progress event."""

    kind: ProgressKind
    stage: str
    message: str
    progress: float | None = None
    shot_id: str | None = None
    user_message: str | None = None
    log_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="microseconds"))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    def to_ndjson(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"


_SUBSCRIBER_QUEUE_MAXSIZE = 1024


class ProgressEventBus:
    """Fan-out bus: writes every event to NDJSON and to all subscribers.

    Ordering guarantee: events appear in the on-disk log and in every
    subscriber queue in the same global order, because the log append + queue
    fan-out happen atomically under one lock. A slow subscriber whose queue
    fills up sees its oldest events dropped (with a warning) rather than
    blocking the bus.
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._subscribers: set[asyncio.Queue[ProgressEvent | None]] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    async def emit(self, event: ProgressEvent) -> None:
        if self._closed:
            raise RuntimeError("event bus is closed")

        async with self._lock:
            await asyncio.to_thread(self._append_log_line, event.to_ndjson())
            for queue in tuple(self._subscribers):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    _logger.warning(
                        "dropping SSE event for full queue (slow subscriber)"
                    )

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[ProgressEvent | None]]:
        """Async context yielding a queue that receives every future event.

        On exit the queue is removed from the subscriber set; subscribers
        also get a ``None`` sentinel when the bus is closed so they can
        cleanly stop iterating.

        Subscribing after ``close()`` immediately yields a queue containing
        only the sentinel so the caller's iterator terminates without hanging.
        """

        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue(
            maxsize=_SUBSCRIBER_QUEUE_MAXSIZE
        )
        if self._closed:
            queue.put_nowait(None)
            try:
                yield queue
            finally:
                pass
            return

        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    async def close(self) -> None:
        """Mark the bus as closed and signal all subscribers."""

        self._closed = True
        for queue in tuple(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    def _append_log_line(self, line: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
