"""Together AI adapter — Together's hosted endpoint speaks OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import register_language_provider
from .openai_compatible import OpenAICompatibleProvider, build_openai_compatible


@register_language_provider("together")
def _factory(sub_config: Mapping[str, Any]) -> OpenAICompatibleProvider:
    merged: dict[str, Any] = {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        **dict(sub_config),
    }
    return build_openai_compatible(merged, name="together")
