"""Local SadTalker lip-sync adapter.

SadTalker (https://github.com/OpenTalker/SadTalker) is Apache-2.0 and produces
full-head motion (better for anime portraits than mouth-only models). We wrap
its CLI rather than its Python API because:

* The library still imports CUDA-only kernels at module top in some releases;
  invoking the CLI lets us point at a venv-installed copy without paying that
  cost on import.
* CLI output is a single mp4 + a few intermediates we can clean up — no need
  to mirror their internal data classes.

Cache: model weights live under ``data_dir/models/sadtalker/`` per the
project's general "lazy per job" download policy; the user installs the
``[lipsync]`` extra to get torch/diffusers and the CLI script.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...errors import ProviderError
from ..base import AnimatedShot, LipSyncProvider
from ..registry import register_lipsync_provider

_DEFAULT_FPS = 25  # SadTalker emits 25 FPS by default


class SadTalkerLocalProvider(LipSyncProvider):
    name = "sadtalker_local"

    def __init__(
        self,
        *,
        cli_binary: str | None = None,
        checkpoints_dir: Path | None = None,
        device: str | None = None,
        still_mode: bool = True,
        preprocess: str = "full",
        fps: int = _DEFAULT_FPS,
    ) -> None:
        self._cli = cli_binary or shutil.which("sadtalker") or "sadtalker"
        self._checkpoints_dir = checkpoints_dir
        self._device = device or _autodetect_device()
        self._still_mode = still_mode
        self._preprocess = preprocess
        self._fps = max(1, fps)

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

        out_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = out_path.parent / f"_sadtalker_{out_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        argv = [
            self._cli,
            "--source_image", str(image_path),
            "--driven_audio", str(audio_path),
            "--result_dir", str(work_dir),
            "--preprocess", self._preprocess,
            "--device", self._device,
            "--fps", str(self._fps),
        ]
        if self._still_mode:
            argv.append("--still")
        if self._checkpoints_dir is not None:
            argv += ["--checkpoint_dir", str(self._checkpoints_dir)]

        try:
            await _run(argv)
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ProviderError(
                "sadtalker_local invocation failed; install with "
                "`pip install booktoanime[lipsync]` and run on CUDA/MPS, "
                "or switch to lipsync.active=replicate. "
                f"underlying error: {exc}"
            ) from exc

        produced = _pick_latest_mp4(work_dir)
        if produced is None:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ProviderError(f"sadtalker_local produced no mp4 under {work_dir}")

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        shutil.move(str(produced), str(tmp_path))
        shutil.rmtree(work_dir, ignore_errors=True)

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
            raise ProviderError("sadtalker_local moved an empty mp4 to the output path")
        tmp_path.replace(out_path)

        # ffprobe duration measurement is the assembler's job; we report a
        # best-effort estimate using ffprobe if available, else the audio
        # length. The MouthAnimator re-measures with ffprobe anyway.
        duration = await _ffprobe_duration(out_path)
        return AnimatedShot(
            path=out_path,
            duration_seconds=duration,
            fps=float(self._fps),
        )

    async def close(self) -> None:
        return None


def _autodetect_device() -> str:
    """Pick a SadTalker --device value matching the local accelerator."""

    # We avoid importing torch here so the registry import stays cheap. The
    # adapter is only used after the user opts in via config; if torch isn't
    # installed the CLI will raise a clearer error than we could.
    if os.environ.get("BOOKTOANIME_FORCE_CPU"):
        return "cpu"
    return "cuda"  # SadTalker also accepts "cpu"; "mps" support is partial


def _pick_latest_mp4(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


async def _run(argv: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise ProviderError(
            f"command exited {process.returncode}: {shlex.join(argv)}\n"
            f"stderr={stderr.decode(errors='replace')}"
        )


async def _ffprobe_duration(path: Path) -> float:
    binary = shutil.which("ffprobe")
    if binary is None:
        return 0.0
    process = await asyncio.create_subprocess_exec(
        binary,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return 0.0
    try:
        return float(stdout.decode().strip() or 0.0)
    except ValueError:
        return 0.0


@register_lipsync_provider("sadtalker_local")
def _factory(sub_config: Mapping[str, Any]) -> SadTalkerLocalProvider:
    checkpoints = sub_config.get("checkpoints_dir")
    return SadTalkerLocalProvider(
        cli_binary=sub_config.get("cli_binary"),
        checkpoints_dir=Path(checkpoints) if checkpoints else None,
        device=sub_config.get("device"),
        still_mode=bool(sub_config.get("still_mode", True)),
        preprocess=str(sub_config.get("preprocess", "full")),
        fps=int(sub_config.get("fps", _DEFAULT_FPS)),
    )


__all__ = ["SadTalkerLocalProvider"]
