"""Tests for the Gemini adapter via a stub client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("google.genai")

from booktoanime.errors import ProviderError
from booktoanime.providers.base import (
    ChatMessage,
    CompletionRequest,
    VisionInput,
)
from booktoanime.providers.language.gemini import GeminiProvider


@dataclass
class _Response:
    text: str | None = None


class _StubModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> _Response:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _Response(text=self._text)


class _StubAio:
    def __init__(self, text: str) -> None:
        self.models = _StubModels(text)


class _StubClient:
    def __init__(self, text: str) -> None:
        self.aio = _StubAio(text)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_complete_returns_text_and_passes_system() -> None:
    client = _StubClient("hello")
    provider = GeminiProvider(model="gemini-flash", api_key="k", client=client)

    text = await provider.complete(
        CompletionRequest(
            messages=[
                ChatMessage(role="system", content="be helpful"),
                ChatMessage(role="user", content="hi"),
            ],
            max_tokens=32,
            temperature=0.5,
            seed=11,
        )
    )

    assert text == "hello"
    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-flash"
    config = call["config"]
    assert config.system_instruction == "be helpful"
    assert config.max_output_tokens == 32
    assert config.temperature == 0.5
    assert config.seed == 11

    contents = call["contents"]
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_explain_image_parses_json(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")

    client = _StubClient(json.dumps({"summary": "s", "detail": "d"}))
    provider = GeminiProvider(model="gemini-flash", api_key="k", client=client)

    explanation = await provider.explain_image(
        VisionInput(image_path=image_path, surrounding_text="ctx", caption_hint="Figure 1")
    )

    assert explanation.summary == "s"
    assert explanation.detail == "d"

    call = client.aio.models.calls[0]
    config = call["config"]
    assert config.response_mime_type == "application/json"
    contents = call["contents"]
    assert len(contents) == 2  # image part + text part


@pytest.mark.asyncio
async def test_explain_image_rejects_non_json(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")

    client = _StubClient("not json")
    provider = GeminiProvider(model="gemini-flash", api_key="k", client=client)

    with pytest.raises(ProviderError):
        await provider.explain_image(
            VisionInput(image_path=image_path, surrounding_text="ctx")
        )
