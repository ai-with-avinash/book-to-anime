"""DeepSeek adapter — thin wrapper over the OpenAI-compatible adapter.

DeepSeek's hosted API is OpenAI wire-format compatible, so we reuse the
``openai_compatible`` factory but default the base URL and surface a separate
provider name for users (and to make config explicit about which key/env the
adapter expects).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import register_language_provider
from .openai_compatible import OpenAICompatibleProvider, build_openai_compatible


@register_language_provider("deepseek")
def _factory(sub_config: Mapping[str, Any]) -> OpenAICompatibleProvider:
    merged: dict[str, Any] = {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        **dict(sub_config),
    }
    return build_openai_compatible(merged, name="deepseek")
