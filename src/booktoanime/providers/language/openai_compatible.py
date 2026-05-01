"""OpenAI-compatible Chat Completions adapter (httpx-based).

This adapter intentionally talks raw HTTP rather than depending on the
``openai`` SDK so it ships in the default install. Any endpoint that
implements ``POST /v1/chat/completions`` works:

* OpenAI itself (``https://api.openai.com/v1``)
* Ollama (``http://localhost:11434/v1``)
* vLLM (``http://localhost:8000/v1``)
* LM Studio (``http://localhost:1234/v1``)
* llama.cpp server (``http://localhost:8080/v1``)
* DeepSeek (``https://api.deepseek.com/v1``) — used by ``DeepSeekProvider``.

Vision support is conditional: if a config'd model is in
``vision_capable_models`` (or the user passes ``vision: true``), images are
encoded as data URLs and sent as the OpenAI multimodal content array. Otherwise
``explain_image`` raises :class:`CapabilityNotSupportedError` so the
orchestrator falls back to ``language.vision_fallback``.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from ...errors import (
    CapabilityNotSupportedError,
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

# Models that are known to accept multimodal (image) inputs through the
# OpenAI-compatible chat-completions schema. Adapters can pass an explicit
# ``vision: true`` to override this allow-list (useful for self-hosted VLMs).
_VISION_CAPABLE_MODELS = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-vision-preview",
        "llava",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "qwen2-vl-7b-instruct",
        "qwen2.5-vl-7b-instruct",
    }
)


_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class OpenAICompatibleProvider(LanguageProvider):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = _DEFAULT_TIMEOUT,
        vision: bool | None = None,
        client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._vision = (
            vision
            if vision is not None
            else any(model.startswith(prefix) for prefix in _VISION_CAPABLE_MODELS)
        )
        self._owns_client = client is None
        timeout_value = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        self._client = client or httpx.AsyncClient(timeout=timeout_value)
        self._extra_headers = dict(extra_headers or {})

    # ----------------------------------------------------- LanguageProvider API

    async def complete(self, request: CompletionRequest) -> str:
        payload = self._chat_payload(
            messages=[self._serialize_message(msg) for msg in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            json_mode=request.json_mode,
            stop=request.stop,
            seed=request.seed,
        )
        data = await with_retries(lambda: self._post_chat(payload))
        return self._extract_text(data)

    async def explain_image(
        self,
        image: VisionInput,
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> ImageExplanation:
        if not self._vision:
            raise CapabilityNotSupportedError(
                f"model {self._model!r} is not configured for vision. "
                "Set `vision: true` in config or use a vision-capable model."
            )

        prompt = (
            "You are explaining an embedded figure from a book to a narrator.\n"
            "Use the surrounding text to ground your explanation. "
            "Respond as JSON with two string fields: 'summary' (1-3 sentences, "
            "narration-ready) and 'detail' (longer reasoning).\n\n"
            f"Caption hint: {image.caption_hint or '(none)'}\n"
            f"Surrounding text:\n{image.surrounding_text[:2000]}"
        )
        encoded = _encode_image_as_data_url(image.image_path)
        payload = self._chat_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": encoded}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
        data = await with_retries(lambda: self._post_chat(payload))
        text = self._extract_text(data)

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
        if self._owns_client:
            await self._client.aclose()

    # ----------------------------------------------------- internals

    def _chat_payload(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = seed
        return payload

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        return {"role": message.role, "content": message.content}

    async def _post_chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        headers = self._build_headers()
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise ProviderTransientError(f"network error talking to {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP error talking to {url}: {exc}") from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:  # JSONDecodeError subclass
            raise ProviderError(f"non-JSON response from {url}: {response.text[:200]}") from exc

        if not isinstance(data, dict):
            raise ProviderError(f"unexpected non-object response: {data!r}")
        return data

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:500]
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"{response.status_code}: {body}")
        if response.status_code == 429:
            raise ProviderRateLimitError(f"429: {body}")
        if 500 <= response.status_code < 600:
            raise ProviderTransientError(f"{response.status_code}: {body}")
        raise ProviderError(f"{response.status_code}: {body}")

    @staticmethod
    def _extract_text(data: Mapping[str, Any]) -> str:
        try:
            choices = data["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed chat response: {data}") from exc

        if isinstance(content, list):
            # Some providers return content as a list of segments even for text-only.
            text_parts = [
                segment.get("text", "")
                for segment in content
                if isinstance(segment, dict) and segment.get("type") == "text"
            ]
            content = "".join(text_parts)
        if not isinstance(content, str):
            raise ProviderError(f"unexpected content shape: {type(content).__name__}")
        return content


def _encode_image_as_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _resolve_api_key(sub_config: Mapping[str, Any]) -> str | None:
    """Pick the API key from explicit config or env var."""

    if sub_config.get("api_key"):
        return str(sub_config["api_key"])
    env_var = sub_config.get("api_key_env")
    if env_var:
        value = os.environ.get(str(env_var))
        if value:
            return value
    return None


def _build(sub_config: Mapping[str, Any], *, name: str | None = None) -> OpenAICompatibleProvider:
    base_url = sub_config.get("base_url", "https://api.openai.com/v1")
    model = sub_config.get("model")
    if not model:
        raise ValueError("openai_compatible provider requires `model:` in config.")
    timeout = float(sub_config.get("request_timeout_s", 60.0))
    vision = sub_config.get("vision")
    return OpenAICompatibleProvider(
        base_url=str(base_url),
        model=str(model),
        api_key=_resolve_api_key(sub_config),
        timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        vision=bool(vision) if vision is not None else None,
        name=name,
    )


@register_language_provider("openai_compatible")
def _factory(sub_config: Mapping[str, Any]) -> OpenAICompatibleProvider:
    return _build(sub_config)


# Re-exported for adapters that wrap this one (vendor wrappers).
build_openai_compatible = _build


__all__ = [
    "OpenAICompatibleProvider",
    "build_openai_compatible",
]
