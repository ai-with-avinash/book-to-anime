"""Groq adapter — Groq's hosted endpoint speaks OpenAI Chat Completions.

We reuse :class:`OpenAICompatibleProvider` with Groq-specific defaults so users
configure ``language.active: groq`` (with no ``base_url``) and it just works.
The native ``groq`` SDK is available as an install extra for users who want
its streaming/function-calling helpers; the wire format for ``complete`` is
identical, so we don't need to depend on it for the default flow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import register_language_provider
from .openai_compatible import OpenAICompatibleProvider, build_openai_compatible


@register_language_provider("groq")
def _factory(sub_config: Mapping[str, Any]) -> OpenAICompatibleProvider:
    merged: dict[str, Any] = {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        **dict(sub_config),
    }
    return build_openai_compatible(merged, name="groq")
