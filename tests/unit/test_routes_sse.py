"""Unit tests for the SSE ``event_stream`` disconnect race.

The route's inner ``event_stream`` async generator must:

1. Detect ``request.is_disconnected()`` while the per-subscriber queue is
   idle — i.e. without blocking on ``queue.get()`` forever.
2. Clean up the orphan queue task on every iteration so no
   ``Task was destroyed but it is pending!`` warnings leak.
3. Leave the bus subscriber set empty after the generator exits (the
   ``async with bus.subscribe()`` context guarantees this — these tests
   verify it).

The route closes over the FastAPI ``Request``/``app.state.runner``
plumbing, so for unit testing we extract the generator coroutine via a
``FakeRequest`` and a real :class:`ProgressEventBus`. This avoids
spinning up ``httpx.AsyncClient`` for what is fundamentally a
single-coroutine assertion.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from booktoanime.api.routes_sse import build_sse_router
from booktoanime.pipeline.events import ProgressEvent, ProgressEventBus, ProgressKind
from booktoanime.pipeline.stages import Stage


class _FakeRequest:
    """Minimal stand-in for ``starlette.requests.Request``.

    Real starlette ``Request.is_disconnected`` returns ``True`` only once
    the underlying ASGI scope reports the client gone; until then it
    parks the coroutine. We mirror that semantic so the ``asyncio.wait``
    race in the route only resolves the disconnect side when the test
    actually flips ``disconnected`` — otherwise the queue side must win.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._disconnect_event = asyncio.Event()

    @property
    def disconnected(self) -> bool:
        return self._disconnect_event.is_set()

    @disconnected.setter
    def disconnected(self, value: bool) -> None:
        if value:
            self._disconnect_event.set()
        else:
            self._disconnect_event.clear()

    async def is_disconnected(self) -> bool:
        await self._disconnect_event.wait()
        return True


class _FakeRunner:
    def __init__(self, bus: ProgressEventBus) -> None:
        self._bus = bus

    def get_bus(self, job_id: str) -> ProgressEventBus | None:
        _ = job_id
        return self._bus


def _build_stream(bus: ProgressEventBus, request: _FakeRequest):
    """Recreate the route's ``event_stream`` generator against a fake request.

    We rebuild the router so the closure captures the fake runner, then
    call into the route handler the same way FastAPI would. The handler
    returns an ``EventSourceResponse`` wrapping our generator; we reach
    inside to pull the generator out for direct iteration.
    """

    app = FastAPI()
    app.state.runner = _FakeRunner(bus)
    request.app = app
    router = build_sse_router()
    # The router has one route; reach in for the endpoint coroutine.
    endpoint = router.routes[0].endpoint  # type: ignore[attr-defined]
    return endpoint, app


def _make_event(message: str) -> ProgressEvent:
    return ProgressEvent(
        kind=ProgressKind.INFO,
        stage=Stage.PARSING.value,
        message=message,
    )


async def _start_stream_and_first_event(
    bus: ProgressEventBus,
    body_iter,
    message: str,
) -> dict:
    """Schedule a ``bus.emit`` concurrently with ``__anext__``.

    The route's generator only subscribes to the bus once iteration
    begins, so the emit has to fire after the generator has entered its
    ``async with bus.subscribe()`` block. We launch the emit as a
    background task and let the event loop interleave.
    """

    async def _emit_after_subscribe() -> None:
        # Give the generator one tick to call ``bus.subscribe`` before
        # we push the event into the queue.
        for _ in range(5):
            await asyncio.sleep(0)
            if bus._subscribers:
                break
        await bus.emit(_make_event(message))

    emit_task = asyncio.create_task(_emit_after_subscribe())
    try:
        first = await asyncio.wait_for(body_iter.__anext__(), timeout=2.0)
    finally:
        await emit_task
    return first


@pytest.mark.asyncio
async def test_disconnect_breaks_loop_while_queue_idle(tmp_path: Path) -> None:
    """Disconnect mid-stream with an empty queue must terminate the generator.

    Without the FIRST_COMPLETED race, ``queue.get()`` would block forever
    holding a subscriber slot. Test sets disconnect=True, awaits one tick,
    and asserts the generator returns and the subscriber set is empty.
    """

    bus = ProgressEventBus(tmp_path / "events.log")
    request = _FakeRequest(app=None)
    endpoint, _ = _build_stream(bus, request)

    response = await endpoint(job_id="job_x", request=request)
    # EventSourceResponse stores the iterator as ``body_iterator``.
    body_iter = response.body_iterator

    first = await _start_stream_and_first_event(bus, body_iter, "hello")
    assert json.loads(first["data"])["message"] == "hello"

    request.disconnected = True

    # Without the bug fix the next __anext__ would block forever. With
    # the fix the disconnect_task wins the race and the generator returns.
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(body_iter.__anext__(), timeout=2.0)

    # Subscriber slot freed by the ``async with bus.subscribe()`` finally.
    assert len(bus._subscribers) == 0
    await bus.close()


@pytest.mark.asyncio
async def test_no_pending_tasks_after_disconnect(tmp_path: Path) -> None:
    """Generator exit must leave no dangling queue.get / is_disconnected tasks.

    We snapshot ``asyncio.all_tasks`` before and after the stream runs and
    assert the delta is zero (plus the current test coroutine itself).
    """

    bus = ProgressEventBus(tmp_path / "events.log")
    request = _FakeRequest(app=None)
    endpoint, _ = _build_stream(bus, request)

    tasks_before = {t for t in asyncio.all_tasks() if not t.done()}

    response = await endpoint(job_id="job_x", request=request)
    body_iter = response.body_iterator
    await _start_stream_and_first_event(bus, body_iter, "event")

    request.disconnected = True
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(body_iter.__anext__(), timeout=2.0)

    # Give the event loop one tick to settle cancelled tasks.
    await asyncio.sleep(0)
    tasks_after = {t for t in asyncio.all_tasks() if not t.done()}
    new_tasks = tasks_after - tasks_before
    assert new_tasks == set(), f"leaked tasks: {new_tasks!r}"

    await bus.close()


@pytest.mark.asyncio
async def test_bus_close_sentinel_emits_done(tmp_path: Path) -> None:
    """When the bus closes mid-stream the generator emits ``done`` and exits."""

    bus = ProgressEventBus(tmp_path / "events.log")
    request = _FakeRequest(app=None)
    endpoint, _ = _build_stream(bus, request)

    response = await endpoint(job_id="job_x", request=request)
    body_iter = response.body_iterator

    first = await _start_stream_and_first_event(bus, body_iter, "first")
    assert first["event"] == ProgressKind.INFO.value

    # Closing the bus pushes a None sentinel.
    await bus.close()
    done = await asyncio.wait_for(body_iter.__anext__(), timeout=1.0)
    assert done["event"] == "done"

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(body_iter.__anext__(), timeout=1.0)

    assert len(bus._subscribers) == 0
