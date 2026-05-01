"""Tests for the provider registry and config-driven instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from booktoanime.providers import base, registry
from booktoanime.providers.base import LanguageProvider


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide an empty registry for the duration of one test."""

    monkeypatch.setattr(registry, "_LANGUAGE_REGISTRY", {})
    monkeypatch.setattr(registry, "_AUDIO_REGISTRY", {})
    monkeypatch.setattr(registry, "_VISUAL_REGISTRY", {})
    # Skip auto-loading built-in modules so we control what's registered.
    monkeypatch.setattr(registry, "_BUILTIN_LANGUAGE_MODULES", ())


class _StubProvider(LanguageProvider):
    name = "stub"

    def __init__(self, *, model: str) -> None:
        self.model = model

    async def complete(self, request: base.CompletionRequest) -> str:
        return "ok"

    async def explain_image(self, image, *, max_tokens=400, temperature=0.2):
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_register_and_build(clean_registry: None) -> None:
    @registry.register_language_provider("stub")
    def factory(sub: Mapping[str, Any]) -> _StubProvider:
        return _StubProvider(model=str(sub["model"]))

    provider = registry.build_language_provider({"active": "stub", "stub": {"model": "abc"}})
    assert isinstance(provider, _StubProvider)
    assert provider.model == "abc"


def test_duplicate_registration_rejected(clean_registry: None) -> None:
    @registry.register_language_provider("stub")
    def first(_: Mapping[str, Any]) -> _StubProvider:
        return _StubProvider(model="x")

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_language_provider("stub")
        def _second(_: Mapping[str, Any]) -> _StubProvider:
            return _StubProvider(model="x")


def test_missing_active_key_raises(clean_registry: None) -> None:
    with pytest.raises(ValueError, match="missing required key 'active'"):
        registry.build_language_provider({"stub": {"model": "x"}})


def test_unknown_provider_lists_registered_options(clean_registry: None) -> None:
    @registry.register_language_provider("stub")
    def factory(_: Mapping[str, Any]) -> _StubProvider:
        return _StubProvider(model="x")

    with pytest.raises(ValueError, match="Registered providers: stub"):
        registry.build_language_provider({"active": "missing"})


def test_missing_sub_block_raises(clean_registry: None) -> None:
    @registry.register_language_provider("stub")
    def factory(_: Mapping[str, Any]) -> _StubProvider:
        return _StubProvider(model="x")

    with pytest.raises(ValueError, match="no `stub:` block"):
        registry.build_language_provider({"active": "stub"})


def test_builtin_modules_self_register() -> None:
    """Once we call build, the openai_compatible adapter should be present.

    Run with the real (non-cleaned) registry so the lazy module load happens.
    """

    config = {
        "active": "openai_compatible",
        "openai_compatible": {
            "model": "test-model",
            "base_url": "http://localhost:9999/v1",
            "api_key": "irrelevant",
        },
    }
    provider = registry.build_language_provider(config)
    assert provider.name == "openai_compatible"
