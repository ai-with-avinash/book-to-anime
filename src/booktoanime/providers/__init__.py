"""Pluggable provider interfaces and adapters.

The pipeline never imports a concrete provider directly. Instead it accepts
:class:`LanguageProvider`, :class:`AudioProvider`, and :class:`VisualProvider`
instances obtained from :func:`registry.build_language_provider` (and friends)
based on the user's ``config.yaml``.

This module re-exports the abstract interfaces and the value objects that
travel across them.
"""

from __future__ import annotations

from .base import (
    AudioProvider,
    ChatMessage,
    CompletionRequest,
    GeneratedAudio,
    GeneratedImage,
    ImageExplanation,
    ImageGenRequest,
    LanguageProvider,
    Role,
    TTSRequest,
    VisionInput,
    VisualProvider,
)
from .registry import (
    build_audio_provider,
    build_language_provider,
    build_visual_provider,
    register_audio_provider,
    register_language_provider,
    register_visual_provider,
)

__all__ = [
    "AudioProvider",
    "ChatMessage",
    "CompletionRequest",
    "GeneratedAudio",
    "GeneratedImage",
    "ImageExplanation",
    "ImageGenRequest",
    "LanguageProvider",
    "Role",
    "TTSRequest",
    "VisionInput",
    "VisualProvider",
    "build_audio_provider",
    "build_language_provider",
    "build_visual_provider",
    "register_audio_provider",
    "register_language_provider",
    "register_visual_provider",
]
