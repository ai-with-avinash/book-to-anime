"""Anthropic Claude adapter using the native ``anthropic`` SDK."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from .._retry import with_retries
from ..base import (
    ChatMessage,
    CompletionRequest,
    ImageExplanation,
    LanguageProvider,
    VisionInput,
)
from ..registry import register_language_provider
from ._sdk_helpers import resolve_api_key

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic


class AnthropicProvider(LanguageProvider):
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str, client: AsyncAnthropic | None = None) -> None:
        # Track ownership *before* the lazy import so we don't accidentally
        # close a client the caller injected.
        self._owns_client = client is None
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
        self._client = client
        self._model = model

    # ----------------------------------------------------- LanguageProvider

    async def complete(self, request: CompletionRequest) -> str:
        system_prompt, history = _split_system(request.messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in history],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        if request.json_mode:
            kwargs["system"] = (
                (system_prompt + "\n\n" if system_prompt else "")
                + "Respond with valid JSON only. No prose, no code fences."
            )

        message = await with_retries(lambda: self._messages_create(kwargs))
        return _extract_text_blocks(message)

    async def explain_image(
        self,
        image: VisionInput,
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> ImageExplanation:
        media_type, encoded = _encode_image(image.image_path)
        prompt = (
            "Explain the figure for a narrator. Use the surrounding text to ground "
            "your explanation. Respond with JSON: {\"summary\": str, \"detail\": str}.\n\n"
            f"Caption hint: {image.caption_hint or '(none)'}\n"
            f"Surrounding text:\n{image.surrounding_text[:2000]}"
        )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": "Respond with valid JSON only. No prose, no code fences.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        message = await with_retries(lambda: self._messages_create(kwargs))
        text = _extract_text_blocks(message)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"vision response was not JSON: {text[:200]}") from exc

        summary = str(parsed.get("summary", "")).strip()
        detail = str(parsed.get("detail", summary)).strip()
        if not summary:
            raise ProviderError("vision response missing 'summary' field")
        return ImageExplanation(summary=summary, detail=detail)

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result

    # ----------------------------------------------------- internals

    async def _messages_create(self, kwargs: Mapping[str, Any]) -> Any:
        try:
            return await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _translate_anthropic_exception(exc) from exc


def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
    system_parts: list[str] = []
    rest: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            rest.append(msg)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def _extract_text_blocks(message: Any) -> str:
    blocks = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    if not parts:
        raise ProviderError("Anthropic response had no text content")
    return "".join(parts)


def _encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".") or "png"
    media_type = f"image/{ {'jpg': 'jpeg'}.get(suffix, suffix) }"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return media_type, encoded


def _translate_anthropic_exception(exc: Exception) -> Exception:
    """Map ``anthropic`` SDK exceptions to our provider error hierarchy."""

    try:
        from anthropic import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )
    except ImportError:
        return exc

    if isinstance(exc, AuthenticationError):
        return ProviderAuthError(str(exc))
    if isinstance(exc, RateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return ProviderTransientError(str(exc))
    if isinstance(exc, APIStatusError) and 500 <= exc.status_code < 600:
        return ProviderTransientError(str(exc))
    if isinstance(exc, APIStatusError):
        return ProviderError(str(exc))
    return exc


@register_language_provider("anthropic")
def _factory(sub_config: Mapping[str, Any]) -> AnthropicProvider:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the 'anthropic' package is required for the Anthropic provider. "
            "Install with `pip install booktoanime[anthropic]`."
        ) from exc

    model = sub_config.get("model")
    if not model:
        raise ValueError("anthropic provider requires `model:` in config.")
    api_key = resolve_api_key(sub_config, default_env="ANTHROPIC_API_KEY")
    return AnthropicProvider(model=str(model), api_key=api_key)
