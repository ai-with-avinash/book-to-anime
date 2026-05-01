"""Google Gemini adapter using the native ``google-genai`` SDK.

Gemini's wire format is not OpenAI-compatible, so this is a real native
adapter. The SDK ships an async client (``client.aio.models.generate_content``)
so we don't need ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
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
    from google.genai import Client as GeminiClient


class GeminiProvider(LanguageProvider):
    name = "gemini"

    def __init__(self, *, model: str, api_key: str, client: GeminiClient | None = None) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model

    # ----------------------------------------------------- LanguageProvider

    async def complete(self, request: CompletionRequest) -> str:
        config = self._build_config(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            seed=request.seed,
            json_mode=request.json_mode,
            system=_system_prompt(request.messages),
        )
        contents = _to_gemini_contents(request.messages)
        response = await with_retries(lambda: self._generate(contents=contents, config=config))
        return _extract_text(response)

    async def explain_image(
        self,
        image: VisionInput,
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> ImageExplanation:
        from google.genai import types

        prompt = (
            "Explain the figure for a narrator. Use the surrounding text to ground "
            "your explanation. Respond as JSON with two fields: 'summary' "
            "(1-3 sentences) and 'detail' (longer reasoning).\n\n"
            f"Caption hint: {image.caption_hint or '(none)'}\n"
            f"Surrounding text:\n{image.surrounding_text[:2000]}"
        )

        image_part = types.Part.from_bytes(
            data=image.image_path.read_bytes(),
            mime_type=_mime_for(image.image_path),
        )
        text_part = types.Part.from_text(text=prompt)

        config = self._build_config(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            seed=None,
            json_mode=True,
            system="Respond with valid JSON only. No prose, no code fences.",
        )
        response = await with_retries(
            lambda: self._generate(contents=[image_part, text_part], config=config)
        )
        text = _extract_text(response)

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
        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result

    # ----------------------------------------------------- internals

    def _build_config(
        self,
        *,
        max_tokens: int,
        temperature: float,
        stop: Sequence[str] | None,
        seed: int | None,
        json_mode: bool,
        system: str | None,
    ) -> Any:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            config_kwargs["stop_sequences"] = list(stop)
        if seed is not None:
            config_kwargs["seed"] = seed
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if system:
            config_kwargs["system_instruction"] = system
        return types.GenerateContentConfig(**config_kwargs)

    async def _generate(self, *, contents: Any, config: Any) -> Any:
        try:
            return await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except Exception as exc:
            raise _translate_gemini_exception(exc) from exc


def _system_prompt(messages: Sequence[ChatMessage]) -> str | None:
    parts = [m.content for m in messages if m.role == "system"]
    return "\n\n".join(parts) if parts else None


def _to_gemini_contents(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        # Gemini uses "model" for assistant turns.
        role = "model" if msg.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg.content}]})
    return contents


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            value = getattr(part, "text", None)
            if value:
                return str(value)
    raise ProviderError("Gemini response had no text content")


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    return f"image/{ {'jpg': 'jpeg'}.get(suffix, suffix) }"


def _translate_gemini_exception(exc: Exception) -> Exception:
    """Map ``google.genai`` SDK errors to our provider error hierarchy."""

    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return exc

    if isinstance(exc, genai_errors.APIError):
        status = getattr(exc, "status", None) or getattr(exc, "code", None)
        if status in (401, 403):
            return ProviderAuthError(str(exc))
        if status == 429:
            return ProviderRateLimitError(str(exc))
        if isinstance(status, int) and 500 <= status < 600:
            return ProviderTransientError(str(exc))
        return ProviderError(str(exc))
    return exc


@register_language_provider("gemini")
def _factory(sub_config: Mapping[str, Any]) -> GeminiProvider:
    try:
        import google.genai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the 'google-genai' package is required for the Gemini provider. "
            "Install with `pip install booktoanime[gemini]`."
        ) from exc

    model = sub_config.get("model")
    if not model:
        raise ValueError("gemini provider requires `model:` in config.")
    api_key = resolve_api_key(sub_config, default_env="GEMINI_API_KEY")
    return GeminiProvider(model=str(model), api_key=api_key)
