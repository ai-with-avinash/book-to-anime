"""Tests for the Kokoro TTS adapter using a stub engine."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from booktoanime.errors import ProviderError
from booktoanime.providers import registry
from booktoanime.providers.audio.kokoro import (
    KokoroProvider,
    _resolve_voice_and_lang,
)
from booktoanime.providers.base import TTSRequest


class _StubEngine:
    """Stub Kokoro pipeline that yields one short sine-wave chunk per call."""

    def __init__(self, *, sample_rate: int = 24_000, duration: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, text: str, *, voice: str, speed: float = 1.0
    ) -> Iterable[tuple[str, str, np.ndarray]]:
        self.calls.append({"text": text, "voice": voice, "speed": speed})
        n = int(self.sample_rate * self.duration)
        wave = 0.1 * np.sin(2 * np.pi * 440.0 * np.arange(n) / self.sample_rate)
        yield ("graphemes", "phonemes", wave.astype(np.float32))


@pytest.mark.asyncio
async def test_synthesize_writes_wav_with_correct_duration(tmp_path: Path) -> None:
    engine = _StubEngine(duration=0.5)
    provider = KokoroProvider(default_voice="af_bella", engine=engine)

    out_path = tmp_path / "shot_0001.wav"
    result = await provider.synthesize(
        TTSRequest(text="Hello world.", voice_id="af_bella", language="en-US"),
        out_path,
    )

    assert result.path == out_path
    assert out_path.is_file()
    assert result.sample_rate == 24_000
    assert result.duration_seconds == pytest.approx(0.5, abs=0.01)

    data, sr = sf.read(str(out_path))
    assert sr == 24_000
    assert len(data) == int(0.5 * 24_000)

    assert engine.calls == [{"text": "Hello world.", "voice": "af_bella", "speed": 1.0}]


@pytest.mark.asyncio
async def test_uses_default_voice_when_request_voice_blank(tmp_path: Path) -> None:
    engine = _StubEngine()
    provider = KokoroProvider(default_voice="af_sarah", engine=engine)

    await provider.synthesize(
        TTSRequest(text="ok", voice_id="", language="en-US"),
        tmp_path / "out.wav",
    )
    assert engine.calls[0]["voice"] == "af_sarah"


@pytest.mark.asyncio
async def test_unknown_voice_for_known_language_raises(tmp_path: Path) -> None:
    provider = KokoroProvider(default_voice="af_bella", engine=_StubEngine())
    with pytest.raises(ProviderError, match="not in Kokoro's en-US voice set"):
        await provider.synthesize(
            TTSRequest(text="ok", voice_id="xx_invalid", language="en-US"),
            tmp_path / "out.wav",
        )


@pytest.mark.asyncio
async def test_empty_text_raises(tmp_path: Path) -> None:
    provider = KokoroProvider(default_voice="af_bella", engine=_StubEngine())
    with pytest.raises(ProviderError, match="empty text"):
        await provider.synthesize(
            TTSRequest(text="   ", voice_id="af_bella", language="en-US"),
            tmp_path / "out.wav",
        )


@pytest.mark.asyncio
async def test_engine_returning_no_chunks_raises(tmp_path: Path) -> None:
    class _EmptyEngine:
        def __call__(self, text: str, *, voice: str, speed: float = 1.0):
            return iter(())

    provider = KokoroProvider(default_voice="af_bella", engine=_EmptyEngine())
    with pytest.raises(ProviderError, match="no audio chunks"):
        await provider.synthesize(
            TTSRequest(text="hi", voice_id="af_bella", language="en-US"),
            tmp_path / "out.wav",
        )


@pytest.mark.asyncio
async def test_engine_factory_invoked_on_first_use(tmp_path: Path) -> None:
    fake_engine = _StubEngine()
    factory_calls: list[str] = []

    def factory(lang_code: str):
        factory_calls.append(lang_code)
        return fake_engine

    provider = KokoroProvider(
        default_voice="af_bella",
        engine_factory=factory,
        lang_code="a",
    )

    # Not loaded yet.
    assert factory_calls == []

    await provider.synthesize(
        TTSRequest(text="hi", voice_id="af_bella", language="en-US"),
        tmp_path / "out.wav",
    )
    assert factory_calls == ["a"]

    # Subsequent calls reuse the loaded engine.
    await provider.synthesize(
        TTSRequest(text="again", voice_id="af_bella", language="en-US"),
        tmp_path / "out2.wav",
    )
    assert factory_calls == ["a"]


@pytest.mark.asyncio
async def test_factory_import_error_surfaces_install_hint() -> None:
    def factory(_lang: str):
        raise ImportError("no kokoro")

    provider = KokoroProvider(default_voice="af_bella", engine_factory=factory)
    with pytest.raises(ImportError, match=r"booktoanime\[kokoro\]"):
        await provider.synthesize(
            TTSRequest(text="hi", voice_id="af_bella", language="en-US"),
            Path("/tmp/never_used.wav"),
        )


@pytest.mark.asyncio
async def test_list_voices_filters_by_language() -> None:
    provider = KokoroProvider(default_voice="af_bella", engine=_StubEngine())
    en_us = await provider.list_voices(language="en-US")
    assert "af_bella" in en_us
    assert "bf_emma" not in en_us

    all_voices = await provider.list_voices()
    assert "af_bella" in all_voices
    assert "bf_emma" in all_voices


def test_resolve_voice_and_lang_picks_default_for_known_language() -> None:
    voice, lang, lang_code = _resolve_voice_and_lang({"language": "en-GB"})
    assert voice == "bf_emma"
    assert lang == "en-GB"
    assert lang_code == "b"


def test_resolve_voice_and_lang_uses_explicit_voice() -> None:
    voice, _lang, _code = _resolve_voice_and_lang({"language": "en-US", "voice_id": "am_adam"})
    assert voice == "am_adam"


def test_resolve_voice_and_lang_unknown_language_no_default() -> None:
    with pytest.raises(ValueError, match="no default is configured"):
        _resolve_voice_and_lang({"language": "ja-JP"})


def test_kokoro_self_registers_via_audio_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "active": "kokoro",
        "kokoro": {"language": "en-US", "voice_id": "af_bella"},
    }
    provider = registry.build_audio_provider(config)
    assert provider.name == "kokoro"
