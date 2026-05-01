"""Kokoro-82M TTS adapter (default audio provider).

Kokoro is an Apache-2.0 licensed 82M-parameter TTS model that runs on CPU at
real-time-ish speeds and benefits modestly from GPU. The upstream ``kokoro``
package exposes a ``KPipeline`` whose ``__call__`` yields ``(graphemes,
phonemes, audio_tensor)`` tuples per sentence chunk.

This adapter:

* Defers the heavy model load until the first synthesis call (the orchestrator
  may build many providers up-front; we don't want to download model weights
  before the user has even confirmed a job).
* Wraps the synchronous pipeline call in ``asyncio.to_thread`` so it doesn't
  block the event loop.
* Concatenates all chunks into one waveform and writes a single WAV via
  ``soundfile``.
* Returns a :class:`GeneratedAudio` with the *measured* duration so the video
  assembler can time shots correctly.

The actual Kokoro engine is hidden behind a small :class:`KokoroEngine`
Protocol so tests can inject a stub without importing torch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import soundfile as sf

from ...errors import ProviderError
from ..base import AudioProvider, GeneratedAudio, TTSRequest
from ..registry import register_audio_provider

_logger = logging.getLogger(__name__)

# Kokoro v0.x ships at 24 kHz. We expose this as a class-level default so
# tests and high-quality profiles can override.
_DEFAULT_SAMPLE_RATE = 24_000

# Voice ids supported by upstream Kokoro v0.x. Kept as a small allow-list so
# config typos surface as ValueError early rather than as a confusing model
# load failure mid-pipeline. Update when upstream adds voices.
_KOKORO_VOICES_BY_LANG: Mapping[str, tuple[str, ...]] = {
    "en-US": (
        "af_bella",
        "af_nicole",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_michael",
    ),
    "en-GB": ("bf_emma", "bf_isabella", "bm_george", "bm_lewis"),
}

# A friendly default per language. Used when the user only sets `language` in
# config and leaves voice_id blank.
_KOKORO_DEFAULT_VOICE: Mapping[str, str] = {
    "en-US": "af_bella",
    "en-GB": "bf_emma",
}


@runtime_checkable
class KokoroEngine(Protocol):
    """Minimal duck-typed interface to the Kokoro pipeline.

    Real instances are ``kokoro.KPipeline``; tests pass a stub that yields
    pre-baked numpy waveforms.
    """

    def __call__(
        self,
        text: str,
        *,
        voice: str,
        speed: float = 1.0,
    ) -> Iterable[Any]:
        """Yield per-chunk results. Each item is a tuple whose last element
        is a 1-D numpy float audio array, matching the upstream contract.
        """


class KokoroProvider(AudioProvider):
    name = "kokoro"

    def __init__(
        self,
        *,
        default_voice: str = "af_bella",
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        lang_code: str = "a",
        engine: KokoroEngine | None = None,
        engine_factory: type | None = None,
    ) -> None:
        """Construct a Kokoro provider.

        Args:
            default_voice: Voice id to use if a request omits ``voice_id``.
            sample_rate: Output sample rate. Kokoro emits 24 kHz today; bumping
                this would force a resample we don't currently apply.
            lang_code: Kokoro language flag (``"a"`` = American English,
                ``"b"`` = British English). See upstream README.
            engine: An already-loaded engine. If supplied, ``engine_factory``
                is ignored and no model load happens.
            engine_factory: Callable that returns a :class:`KokoroEngine`. Used
                for lazy initialization. Defaults to ``kokoro.KPipeline``.
        """

        self._default_voice = default_voice
        self._sample_rate = sample_rate
        self._lang_code = lang_code
        self._engine: KokoroEngine | None = engine
        self._engine_factory = engine_factory
        self._engine_lock = asyncio.Lock()

    # ------------------------------------------------------ AudioProvider API

    async def list_voices(self, language: str | None = None) -> Sequence[str]:
        if language is None:
            return tuple(v for voices in _KOKORO_VOICES_BY_LANG.values() for v in voices)
        return _KOKORO_VOICES_BY_LANG.get(language, ())

    async def synthesize(self, request: TTSRequest, out_path: Path) -> GeneratedAudio:
        if not request.text.strip():
            raise ProviderError("TTS request has empty text.")

        voice = request.voice_id or self._default_voice
        await self._validate_voice(voice, request.language)
        engine = await self._get_engine()

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            chunks = await asyncio.to_thread(
                _run_kokoro,
                engine,
                request.text,
                voice,
                request.speed,
            )
        except Exception as exc:
            raise ProviderError(f"Kokoro synthesis failed: {exc}") from exc

        if not chunks:
            raise ProviderError("Kokoro produced no audio chunks for this text.")

        waveform = np.concatenate(chunks).astype(np.float32, copy=False)
        # Kokoro can occasionally over-shoot [-1, 1]; soft-clip to avoid distortion on save.
        np.clip(waveform, -1.0, 1.0, out=waveform)

        await asyncio.to_thread(
            sf.write,
            str(out_path),
            waveform,
            self._sample_rate,
            "PCM_16",
        )

        duration = float(len(waveform)) / float(self._sample_rate)
        return GeneratedAudio(
            path=out_path,
            duration_seconds=duration,
            sample_rate=self._sample_rate,
        )

    async def close(self) -> None:
        # Kokoro has no resources to release; we keep the engine ref alive in
        # case the orchestrator calls synthesize() again on the same instance.
        return None

    # ------------------------------------------------------ internals

    async def _get_engine(self) -> KokoroEngine:
        if self._engine is not None:
            return self._engine

        async with self._engine_lock:
            if self._engine is not None:
                return self._engine

            factory = self._engine_factory or _import_kokoro_factory
            try:
                engine = await asyncio.to_thread(factory, self._lang_code)
            except ImportError as exc:
                raise ImportError(
                    "the 'kokoro' package is required for the Kokoro provider. "
                    "Install with `pip install booktoanime[kokoro]` (this also pulls torch)."
                ) from exc
            except Exception as exc:
                raise ProviderError(f"Kokoro engine init failed: {exc}") from exc

            self._engine = cast(KokoroEngine, engine)
            return self._engine

    async def _validate_voice(self, voice: str, language: str) -> None:
        # Be permissive: if the language isn't one we know about, trust the user.
        known = _KOKORO_VOICES_BY_LANG.get(language)
        if known is None:
            return
        if voice not in known:
            raise ProviderError(
                f"voice_id {voice!r} is not in Kokoro's {language} voice set: "
                f"{', '.join(known)}."
            )


def _run_kokoro(
    engine: KokoroEngine,
    text: str,
    voice: str,
    speed: float,
) -> list[np.ndarray]:
    """Iterate the engine, returning a list of float32 audio arrays."""

    chunks: list[np.ndarray] = []
    for result in engine(text, voice=voice, speed=speed):
        audio = _audio_from_result(result)
        if audio is None or audio.size == 0:
            continue
        chunks.append(audio.astype(np.float32, copy=False))
    return chunks


def _audio_from_result(result: Any) -> np.ndarray | None:
    """Pull the audio array out of a Kokoro yield item.

    Upstream's chunk shape is ``(graphemes, phonemes, audio)`` but we don't
    rely on it — we just take the last array-like field.
    """

    if isinstance(result, np.ndarray):
        return result

    last = result[-1] if isinstance(result, (tuple, list)) else result
    if hasattr(last, "detach"):  # torch.Tensor
        last = last.detach().cpu().numpy()
    if isinstance(last, np.ndarray):
        return last
    return None


def _import_kokoro_factory(lang_code: str) -> KokoroEngine:
    """Default factory: load ``kokoro.KPipeline`` for the requested language."""

    from kokoro import KPipeline

    return cast(KokoroEngine, KPipeline(lang_code=lang_code))


def _resolve_voice_and_lang(sub_config: Mapping[str, Any]) -> tuple[str, str, str]:
    language = str(sub_config.get("language", "en-US"))
    voice = sub_config.get("voice_id") or _KOKORO_DEFAULT_VOICE.get(language)
    if not voice:
        raise ValueError(
            f"kokoro requires `voice_id:` for language {language!r}; "
            "no default is configured."
        )
    lang_code = "a" if language.startswith("en-US") else "b"
    return str(voice), language, lang_code


@register_audio_provider("kokoro")
def _factory(sub_config: Mapping[str, Any]) -> KokoroProvider:
    voice, _language, lang_code = _resolve_voice_and_lang(sub_config)
    sample_rate = int(sub_config.get("sample_rate", _DEFAULT_SAMPLE_RATE))
    return KokoroProvider(
        default_voice=voice,
        sample_rate=sample_rate,
        lang_code=lang_code,
    )


__all__ = ["KokoroEngine", "KokoroProvider"]
