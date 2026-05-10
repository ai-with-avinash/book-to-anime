"""Tests for the lightweight vendor wrappers (DeepSeek/Groq/Together/Fireworks/Mistral).

These all reuse the OpenAI-compatible adapter under the hood with vendor-
specific defaults. We verify the defaults are wired correctly and that user
config can override them.
"""

from __future__ import annotations

import pytest

from booktoanime.providers import registry


@pytest.fixture(autouse=True)
def _ensure_modules_loaded() -> None:
    # Touching build_language_provider triggers the lazy module import that
    # registers the vendor wrappers.
    registry._ensure_language_modules_loaded()


@pytest.mark.parametrize(
    ("name", "default_base_url", "env_var"),
    [
        ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        ("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        ("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        ("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    ],
)
def test_vendor_wrappers_have_correct_defaults(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    default_base_url: str,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, "key-from-env")

    config = {"active": name, name: {"model": "vendor-model"}}
    provider = registry.build_language_provider(config)

    assert provider._base_url == default_base_url  # type: ignore[attr-defined]
    assert provider._model == "vendor-model"  # type: ignore[attr-defined]
    assert provider._api_key == "key-from-env"  # type: ignore[attr-defined]


def test_user_config_overrides_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k")
    config = {
        "active": "groq",
        "groq": {"model": "m", "base_url": "https://custom.example/v1"},
    }
    provider = registry.build_language_provider(config)
    assert provider._base_url == "https://custom.example/v1"  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["groq", "together", "fireworks", "mistral", "deepseek"])
def test_vendor_wrappers_carry_their_config_name(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(f"{name.upper()}_API_KEY", "k")
    provider = registry.build_language_provider({"active": name, name: {"model": "m"}})
    assert provider.name == name
