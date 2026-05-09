"""Run booktoanime locally with persona-only visuals (fast).

Stack:
    * LLM: Ollama (http://localhost:11434, model `llama3.2:3b`)
    * TTS: Kokoro (real audio narration)
    * Visual: persona reuse — one PNG per job, copied to every shot
    * Assembly: real ffmpeg

Persona source priority:
    1. ``BOOKTOANIME_PERSONA_PATH`` env var → reuse that PNG (no model download).
    2. SDXL `prepare()` → generates one anime portrait (~7GB download first run,
       then ~30-60s per job on MPS).

Per-shot ``render()`` copies the persona file. Image stage drops from ~35min
(46 SDXL renders) to ~5s (46 file copies).

Usage:
    python scripts/run_local.py
    BOOKTOANIME_PERSONA_PATH=~/my_anime_face.png python scripts/run_local.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from booktoanime._dotenv import load_dotenv
from booktoanime.api import AppSettings, ProviderFactory, create_app
from booktoanime.parsing import PDFParser
from booktoanime.pipeline.manifest import ProvidersConfig
from booktoanime.providers.audio.kokoro import _factory as kokoro_factory
from booktoanime.providers.base import GeneratedImage, LipSyncProvider, VisualProvider
from booktoanime.providers.language.openai_compatible import build_openai_compatible
from booktoanime.providers.lipsync.replicate_hosted import (
    _factory as replicate_lipsync_factory,
)
from booktoanime.providers.visual.sdxl_diffusers import _factory as sdxl_factory


class PersonaOnlyVisual(VisualProvider):
    """Wrap a base VisualProvider but emit one persona for every shot.

    Removes the per-shot SDXL inference cost; only the persona generation
    runs through the wrapped provider. If ``user_persona_path`` is set, even
    that step is skipped — the file is copied straight into the persona dir.
    """

    name = "persona_only"

    def __init__(
        self,
        base: VisualProvider,
        persona_dir: Path,
        user_persona_path: Path | None = None,
    ) -> None:
        self._base = base
        self._persona_dir = persona_dir
        self._user_persona_path = user_persona_path
        self._persona_path: Path | None = None

    async def prepare(self, *, anime_style: str, narrator_seed: int) -> Path:
        self._persona_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._persona_dir / f"{anime_style}__{narrator_seed}.png"

        if not cache_path.is_file():
            if self._user_persona_path is not None:
                shutil.copyfile(self._user_persona_path, cache_path)
            else:
                source = await self._base.prepare(
                    anime_style=anime_style, narrator_seed=narrator_seed
                )
                if source != cache_path:
                    shutil.copyfile(source, cache_path)

        self._persona_path = cache_path
        return cache_path

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        if self._persona_path is None:
            raise RuntimeError(
                "PersonaOnlyVisual.render() called before prepare(); "
                "pipeline must call prepare() first."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._persona_path, out_path)
        return GeneratedImage(
            path=out_path,
            seed=request.seed,
            width=request.width,
            height=request.height,
        )

    async def close(self) -> None:
        await self._base.close()


def _env(key: str, default: str) -> str:
    """Read env var with default. Empty string → default."""

    value = os.environ.get(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def _build_language() -> Any:
    return build_openai_compatible(
        {
            "base_url": _env("BOOKTOANIME_LLM_BASE_URL", "http://localhost:11434/v1"),
            "model": _env("BOOKTOANIME_LLM_MODEL", "llama3.2:3b"),
            "api_key": _env("BOOKTOANIME_LLM_API_KEY", "ollama"),
            "request_timeout_s": _env_int("BOOKTOANIME_LLM_TIMEOUT_S", 300),
        }
    )


def _build_audio() -> Any:
    return kokoro_factory(
        {
            "voice_id": _env("BOOKTOANIME_TTS_VOICE_ID", "af_bella"),
            "language": _env("BOOKTOANIME_TTS_LANGUAGE", "en-US"),
            "sample_rate": _env_int("BOOKTOANIME_TTS_SAMPLE_RATE", 24000),
        }
    )


def _build_visual_factory(persona_dir: Path) -> Any:
    user_persona_env = os.environ.get("BOOKTOANIME_PERSONA_PATH")
    user_persona = Path(user_persona_env).expanduser() if user_persona_env else None
    if user_persona is not None and not user_persona.is_file():
        raise FileNotFoundError(
            f"BOOKTOANIME_PERSONA_PATH points to {user_persona!r}, but no file there."
        )

    def _factory() -> VisualProvider:
        base = sdxl_factory(
            {
                "checkpoint": _env(
                    "BOOKTOANIME_SDXL_CHECKPOINT",
                    "stabilityai/stable-diffusion-xl-base-1.0",
                ),
                "ip_adapter_repo": _env("BOOKTOANIME_SDXL_IP_ADAPTER_REPO", "h94/IP-Adapter"),
                "ip_adapter_subfolder": _env(
                    "BOOKTOANIME_SDXL_IP_ADAPTER_SUBFOLDER", "sdxl_models"
                ),
                "ip_adapter_weights": _env(
                    "BOOKTOANIME_SDXL_IP_ADAPTER_WEIGHTS",
                    "ip-adapter-plus_sdxl_vit-h.safetensors",
                ),
                "width": _env_int("BOOKTOANIME_SDXL_WIDTH", 1024),
                "height": _env_int("BOOKTOANIME_SDXL_HEIGHT", 1024),
                "steps": _env_int("BOOKTOANIME_SDXL_STEPS", 20),
                "guidance": _env_float("BOOKTOANIME_SDXL_GUIDANCE", 5.5),
            }
        )
        return PersonaOnlyVisual(base, persona_dir, user_persona)

    return _factory


def _build_lipsync() -> LipSyncProvider | None:
    """Replicate-hosted SadTalker if ``REPLICATE_API_TOKEN`` is set, else None.

    Returning None is a soft default — the orchestrator skips
    MOUTH_ANIMATION when ``lipsync.enabled`` is false in the JobConfig and
    raises a clear error if it's enabled but no provider is wired.
    """

    if not os.environ.get("REPLICATE_API_TOKEN"):
        return None
    return replicate_lipsync_factory({"api_key_env": "REPLICATE_API_TOKEN"})


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    data_dir = REPO_ROOT / ".local-data"
    data_dir.mkdir(exist_ok=True)
    persona_dir = data_dir / "personas"

    factory = ProviderFactory(
        language_factory=_build_language,
        audio_factory=_build_audio,
        visual_factory=_build_visual_factory(persona_dir),
        lipsync_factory=_build_lipsync,
    )
    settings = AppSettings(
        data_dir=data_dir,
        provider_factory=factory,
        parser_factory=PDFParser,
        config_overrides={
            "providers_obj": ProvidersConfig(
                language="openai_compatible",
                audio="kokoro",
                visual="sdxl_diffusers",
            ),
            "providers": {
                "language": "openai_compatible",
                "audio": "kokoro",
                "visual": "sdxl_diffusers",
            },
        },
    )

    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="info", loop="asyncio")
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
