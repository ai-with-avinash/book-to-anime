"""Tests for the OpenAI-compatible HTTP adapter using respx mocks."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from booktoanime.errors import (
    CapabilityNotSupportedError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)
from booktoanime.providers.base import (
    ChatMessage,
    CompletionRequest,
    VisionInput,
)
from booktoanime.providers.language.openai_compatible import (
    OpenAICompatibleProvider,
    build_openai_compatible,
)

BASE_URL = "https://example.test/v1"


def _provider(*, vision: bool | None = None, model: str = "test-model") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url=BASE_URL,
        model=model,
        api_key="sk-test",
        vision=vision,
    )


@pytest.mark.asyncio
async def test_complete_sends_expected_payload_and_returns_text() -> None:
    async with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello world"}}]},
            )
        )

        provider = _provider()
        try:
            text = await provider.complete(
                CompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=64,
                    temperature=0.1,
                    json_mode=True,
                    seed=7,
                )
            )
        finally:
            await provider.close()

    assert text == "hello world"
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.1
    assert payload["seed"] == 7
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_handles_segmented_content_array() -> None:
    async with respx.mock() as router:
        router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "foo "},
                                    {"type": "text", "text": "bar"},
                                ]
                            }
                        }
                    ]
                },
            )
        )

        provider = _provider()
        try:
            text = await provider.complete(
                CompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=8,
                )
            )
        finally:
            await provider.close()

    assert text == "foo bar"


@pytest.mark.asyncio
async def test_401_raises_auth_error_without_retry() -> None:
    async with respx.mock(assert_all_called=False) as router:
        route = router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "bad key"})
        )

        provider = _provider()
        try:
            with pytest.raises(ProviderAuthError):
                await provider.complete(
                    CompletionRequest(
                        messages=[ChatMessage(role="user", content="hi")],
                        max_tokens=8,
                    )
                )
        finally:
            await provider.close()

        assert route.call_count == 1


@pytest.mark.asyncio
async def test_429_retries_then_succeeds() -> None:
    async with respx.mock() as router:
        route = router.post(f"{BASE_URL}/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": "slow down"}),
                httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
            ]
        )

        provider = _provider()
        try:
            text = await provider.complete(
                CompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=8,
                )
            )
        finally:
            await provider.close()

    assert text == "ok"
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_429_exhausts_attempts_and_raises() -> None:
    async with respx.mock() as router:
        router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": "slow down"})
        )

        provider = _provider()
        try:
            with pytest.raises(ProviderRateLimitError):
                await provider.complete(
                    CompletionRequest(
                        messages=[ChatMessage(role="user", content="hi")],
                        max_tokens=8,
                    )
                )
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_explain_image_without_vision_raises_capability_error(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    provider = _provider(vision=False)
    try:
        with pytest.raises(CapabilityNotSupportedError):
            await provider.explain_image(
                VisionInput(image_path=image_path, surrounding_text="ctx")
            )
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_explain_image_with_vision_returns_explanation(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    async with respx.mock() as router:
        route = router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"summary": "a chart", "detail": "longer explanation"}'
                            }
                        }
                    ]
                },
            )
        )

        provider = _provider(vision=True)
        try:
            explanation = await provider.explain_image(
                VisionInput(
                    image_path=image_path,
                    surrounding_text="surrounding text",
                    caption_hint="Figure 1.1",
                )
            )
        finally:
            await provider.close()

    assert explanation.summary == "a chart"
    assert explanation.detail == "longer explanation"

    payload = json.loads(route.calls.last.request.content)
    assert payload["response_format"] == {"type": "json_object"}
    user_msg = payload["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_non_json_vision_response_raises_provider_error(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    async with respx.mock() as router:
        router.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not json"}}]},
            )
        )

        provider = _provider(vision=True)
        try:
            with pytest.raises(ProviderError):
                await provider.explain_image(
                    VisionInput(image_path=image_path, surrounding_text="ctx")
                )
        finally:
            await provider.close()


def test_factory_requires_model() -> None:
    with pytest.raises(ValueError, match="requires `model:`"):
        build_openai_compatible({"base_url": "http://x/v1"})
