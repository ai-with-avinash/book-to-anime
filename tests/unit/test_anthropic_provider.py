"""Tests for the Anthropic adapter using a stub SDK client.

We don't require the real ``anthropic`` package to be installed for unit tests:
the adapter accepts an injected client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("anthropic")  # only the import-time path; client is stubbed below.

from booktoanime.errors import ProviderError
from booktoanime.providers.base import (
    ChatMessage,
    CompletionRequest,
    VisionInput,
)
from booktoanime.providers.language.anthropic import AnthropicProvider


@dataclass
class _Block:
    text: str


@dataclass
class _Message:
    content: list[_Block]


class _StubMessages:
    def __init__(self, response: _Message) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Message:
        self.calls.append(kwargs)
        return self._response


@dataclass
class _StubClient:
    response: _Message
    messages: _StubMessages = field(init=False)

    def __post_init__(self) -> None:
        self.messages = _StubMessages(self.response)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_complete_passes_system_and_returns_text() -> None:
    client = _StubClient(_Message(content=[_Block(text="hello"), _Block(text=" world")]))
    provider = AnthropicProvider(model="claude-test", api_key="k", client=client)

    text = await provider.complete(
        CompletionRequest(
            messages=[
                ChatMessage(role="system", content="be helpful"),
                ChatMessage(role="user", content="hi"),
            ],
            max_tokens=64,
            temperature=0.3,
            stop=["END"],
        )
    )

    assert text == "hello world"
    call = client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.3
    assert call["system"] == "be helpful"
    assert call["stop_sequences"] == ["END"]
    assert call["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_json_mode_appends_system_instruction() -> None:
    client = _StubClient(_Message(content=[_Block(text='{"k": "v"}')]))
    provider = AnthropicProvider(model="claude-test", api_key="k", client=client)

    await provider.complete(
        CompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=8,
            json_mode=True,
        )
    )

    assert "JSON only" in client.messages.calls[0]["system"]


@pytest.mark.asyncio
async def test_explain_image_parses_json(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake-bytes")

    client = _StubClient(
        _Message(content=[_Block(text=json.dumps({"summary": "a chart", "detail": "longer"}))])
    )
    provider = AnthropicProvider(model="claude-test", api_key="k", client=client)

    explanation = await provider.explain_image(
        VisionInput(
            image_path=image_path,
            surrounding_text="ctx",
            caption_hint="Figure 1.1",
        )
    )

    assert explanation.summary == "a chart"
    assert explanation.detail == "longer"
    call = client.messages.calls[0]
    user_content = call["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[0]["source"]["media_type"] == "image/png"
    assert user_content[1]["type"] == "text"


@pytest.mark.asyncio
async def test_explain_image_rejects_non_json(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake-bytes")

    client = _StubClient(_Message(content=[_Block(text="not json")]))
    provider = AnthropicProvider(model="claude-test", api_key="k", client=client)

    with pytest.raises(ProviderError):
        await provider.explain_image(
            VisionInput(image_path=image_path, surrounding_text="ctx")
        )
