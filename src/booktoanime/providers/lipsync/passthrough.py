"""No-op lip-sync provider.

Wraps the source PNG into a real H.264 mp4 of the requested duration. The
mouth doesn't move — this provider exists so the rest of the pipeline can
treat lip-sync output uniformly (per-shot mp4 → assembler) regardless of
whether mouth animation is enabled.

It is also the test-friendly default: ``passthrough`` requires only ffmpeg,
no model downloads, no GPU, no network.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import wave
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...errors import ProviderError
from ..base import AnimatedShot, LipSyncProvider
from ..registry import register_lipsync_provider

_DEFAULT_FPS = 30


class PassthroughLipSyncProvider(LipSyncProvider):
    name = "passthrough"

    def __init__(
        self,
        *,
        fps: int = _DEFAULT_FPS,
        ffmpeg_binary: str | None = None,
    ) -> None:
        self._fps = max(1, fps)
        self._binary = ffmpeg_binary or shutil.which("ffmpeg") or "ffmpeg"

    async def animate(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        out_path: Path,
    ) -> AnimatedShot:
        if not image_path.is_file():
            raise ProviderError(f"persona image missing: {image_path}")
        if not audio_path.is_file():
            raise ProviderError(f"shot audio missing: {audio_path}")

        duration = _read_wav_duration(audio_path)
        if duration <= 0.0:
            raise ProviderError(f"shot audio has zero duration: {audio_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        argv = [
            self._binary,
            "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1",
            "-t", f"{duration:.3f}",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-r", str(self._fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp_path),
        ]
        try:
            await _run_ffmpeg(argv)
        except Exception as exc:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
            raise ProviderError(f"passthrough lip-sync ffmpeg failed: {exc}") from exc

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise ProviderError("passthrough lip-sync produced an empty file")
        tmp_path.replace(out_path)

        return AnimatedShot(
            path=out_path,
            duration_seconds=duration,
            fps=float(self._fps),
        )

    async def close(self) -> None:
        return None


def _read_wav_duration(path: Path) -> float:
    """Return WAV duration in seconds; 0.0 if the file is unreadable."""

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, OSError):
        return 0.0
    if rate <= 0:
        return 0.0
    return float(frames) / float(rate)


async def _run_ffmpeg(argv: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise ProviderError(
            f"ffmpeg exited {process.returncode}: {shlex.join(argv)} stderr={stderr.decode(errors='replace')}"
        )


@register_lipsync_provider("passthrough")
def _factory(sub_config: Mapping[str, Any]) -> PassthroughLipSyncProvider:
    fps = int(sub_config.get("fps", _DEFAULT_FPS))
    return PassthroughLipSyncProvider(fps=fps)


__all__ = ["PassthroughLipSyncProvider"]
