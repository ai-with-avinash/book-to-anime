"""Provider registry and config-driven instantiation.

Adapters self-register via the ``register_*_provider`` decorators below; the
orchestrator only ever calls ``build_*_provider(config)`` and gets a fully
configured instance. New adapters drop in as a single file under
``providers/{language,audio,visual}/`` plus an entry in ``config.yaml`` —
no edits to pipeline code required.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .base import AudioProvider, LanguageProvider, VisualProvider

T_Lang = TypeVar("T_Lang", bound=LanguageProvider)
T_Audio = TypeVar("T_Audio", bound=AudioProvider)
T_Visual = TypeVar("T_Visual", bound=VisualProvider)

_LANGUAGE_REGISTRY: dict[str, Callable[[Mapping[str, Any]], LanguageProvider]] = {}
_AUDIO_REGISTRY: dict[str, Callable[[Mapping[str, Any]], AudioProvider]] = {}
_VISUAL_REGISTRY: dict[str, Callable[[Mapping[str, Any]], VisualProvider]] = {}

# Built-in adapter modules to import on first registry use. They self-register
# via decorators in their module bodies. Optional native SDKs are guarded by
# ImportError handling so missing extras don't break first-run.
_BUILTIN_LANGUAGE_MODULES: tuple[str, ...] = (
    "booktoanime.providers.language.openai_compatible",
    "booktoanime.providers.language.anthropic",
    "booktoanime.providers.language.gemini",
    "booktoanime.providers.language.groq",
    "booktoanime.providers.language.together",
    "booktoanime.providers.language.fireworks",
    "booktoanime.providers.language.deepseek",
    "booktoanime.providers.language.mistral",
)

_BUILTIN_AUDIO_MODULES: tuple[str, ...] = (
    "booktoanime.providers.audio.kokoro",
)

_BUILTIN_VISUAL_MODULES: tuple[str, ...] = (
    "booktoanime.providers.visual.sdxl_diffusers",
)


# --------------------------------------------------------- registration helpers


def register_language_provider(
    name: str,
) -> Callable[[Callable[[Mapping[str, Any]], T_Lang]], Callable[[Mapping[str, Any]], T_Lang]]:
    """Decorator to register a :class:`LanguageProvider` factory under ``name``."""

    def decorator(
        factory: Callable[[Mapping[str, Any]], T_Lang],
    ) -> Callable[[Mapping[str, Any]], T_Lang]:
        if name in _LANGUAGE_REGISTRY:
            raise ValueError(f"Language provider already registered: {name!r}")
        _LANGUAGE_REGISTRY[name] = factory
        return factory

    return decorator


def register_audio_provider(
    name: str,
) -> Callable[[Callable[[Mapping[str, Any]], T_Audio]], Callable[[Mapping[str, Any]], T_Audio]]:
    """Decorator to register an :class:`AudioProvider` factory under ``name``."""

    def decorator(
        factory: Callable[[Mapping[str, Any]], T_Audio],
    ) -> Callable[[Mapping[str, Any]], T_Audio]:
        if name in _AUDIO_REGISTRY:
            raise ValueError(f"Audio provider already registered: {name!r}")
        _AUDIO_REGISTRY[name] = factory
        return factory

    return decorator


def register_visual_provider(
    name: str,
) -> Callable[[Callable[[Mapping[str, Any]], T_Visual]], Callable[[Mapping[str, Any]], T_Visual]]:
    """Decorator to register a :class:`VisualProvider` factory under ``name``."""

    def decorator(
        factory: Callable[[Mapping[str, Any]], T_Visual],
    ) -> Callable[[Mapping[str, Any]], T_Visual]:
        if name in _VISUAL_REGISTRY:
            raise ValueError(f"Visual provider already registered: {name!r}")
        _VISUAL_REGISTRY[name] = factory
        return factory

    return decorator


# --------------------------------------------------------- builders


def build_language_provider(config: Mapping[str, Any]) -> LanguageProvider:
    """Instantiate a :class:`LanguageProvider` from a parsed config block.

    Args:
        config: A mapping with at least ``{"active": "<name>"}`` plus a
            sibling block matching that name (``config["openai_compatible"]``,
            etc.). Extra keys (such as ``"vision_fallback"``) are ignored here
            and consumed by the orchestrator.

    Raises:
        ValueError: ``active`` missing/unknown or its sub-block missing.
    """

    _ensure_language_modules_loaded()
    name, sub_config = _resolve_active(config, _LANGUAGE_REGISTRY, kind="language")
    return _LANGUAGE_REGISTRY[name](sub_config)


def build_audio_provider(config: Mapping[str, Any]) -> AudioProvider:
    _ensure_audio_modules_loaded()
    name, sub_config = _resolve_active(config, _AUDIO_REGISTRY, kind="audio")
    return _AUDIO_REGISTRY[name](sub_config)


def build_visual_provider(config: Mapping[str, Any]) -> VisualProvider:
    _ensure_visual_modules_loaded()
    name, sub_config = _resolve_active(config, _VISUAL_REGISTRY, kind="visual")
    return _VISUAL_REGISTRY[name](sub_config)


# --------------------------------------------------------- internals


def _resolve_active(
    config: Mapping[str, Any],
    registry: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    *,
    kind: str,
) -> tuple[str, Mapping[str, Any]]:
    if "active" not in config:
        raise ValueError(
            f"{kind} provider config is missing required key 'active'. "
            f"Set e.g. `{kind}.active: openai_compatible` in config.yaml."
        )

    name = str(config["active"])
    if name not in registry:
        registered = ", ".join(sorted(registry)) or "(none registered)"
        raise ValueError(
            f"unknown {kind} provider: {name!r}. Registered providers: {registered}."
        )

    sub_config = config.get(name)
    if sub_config is None:
        raise ValueError(
            f"{kind} provider {name!r} is selected but config has no `{name}:` block."
        )
    if not isinstance(sub_config, Mapping):
        raise ValueError(f"{kind} provider {name!r} sub-config must be a mapping.")

    return name, sub_config


def _ensure_language_modules_loaded() -> None:
    """Lazily import built-in language adapters so they self-register.

    Modules whose optional SDK is not installed import-fail silently — the
    user only gets an error if they actually select that provider in config.
    """

    for mod_name in _BUILTIN_LANGUAGE_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            # Native SDK not installed for this provider; skip silently.
            continue


def _ensure_audio_modules_loaded() -> None:
    """Lazily import built-in audio adapters so they self-register."""

    for mod_name in _BUILTIN_AUDIO_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue


def _ensure_visual_modules_loaded() -> None:
    """Lazily import built-in visual adapters so they self-register."""

    for mod_name in _BUILTIN_VISUAL_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue


# Helpers used by tests to start from a clean registry.
def _reset_registries_for_tests() -> None:
    _LANGUAGE_REGISTRY.clear()
    _AUDIO_REGISTRY.clear()
    _VISUAL_REGISTRY.clear()
