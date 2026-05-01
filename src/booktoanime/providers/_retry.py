"""Shared retry policy used by language adapters.

Centralizes backoff so every adapter behaves consistently when an upstream
returns a 429 or 5xx. Auth errors are NOT retried — the user can't fix them
with a re-attempt, and burning credentials at the rate limit is unhelpful.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..errors import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
)

T = TypeVar("T")


async def with_retries(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    initial: float = 0.5,
    maximum: float = 8.0,
) -> T:
    """Retry ``func`` on transient/rate-limit errors with jittered backoff.

    The final failure is re-raised as the original :class:`ProviderError`
    subclass, not as :class:`tenacity.RetryError`, so callers can match on
    a meaningful type.
    """

    retrying = AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=initial, max=maximum),
        retry=retry_if_exception_type((ProviderRateLimitError, ProviderTransientError)),
        reraise=True,
    )

    try:
        async for attempt in retrying:
            with attempt:
                return await func()
    except RetryError as exc:  # pragma: no cover - tenacity reraise=True covers this
        last = exc.last_attempt.exception()
        if isinstance(last, ProviderError):
            raise last from exc
        raise

    # Unreachable; the loop either returns or raises. Keeps mypy happy.
    raise ProviderError("retry loop exited without a result")
