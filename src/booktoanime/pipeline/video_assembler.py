"""ffmpeg-driven video assembly.

For each shot we build:

* A still ``image2`` input loop with duration = measured audio duration.
* A zoompan filter implementing Ken Burns motion (the storyboard tells us
  the from/to anchor + zoom values).
* The shot's WAV file as the audio stream.

Shots are concatenated with an ``xfade`` chain (video) and a parallel
``acrossfade`` chain (audio). The final stream is encoded as H.264 + AAC
into ``output.mp4``.

We do NOT depend on the ``ffmpeg-python`` package. It's a thin wrapper that
adds little over building the argv ourselves and pulls in extra surface for
no obvious win. The :class:`VideoAssembler` accepts an injectable runner so
tests can stub the subprocess without ffmpeg installed.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError
from .artifacts import (
    AudioIndex,
    ImagesIndex,
    Storyboard,
)
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .srt_sidecar import write_srt
from .stages import Stage

_logger = logging.getLogger(__name__)

FFmpegRunner = Callable[[Sequence[str], Path], Awaitable[None]]


class FFmpegError(BookToAnimeError):
    user_message = "Video assembly failed. Check the per-job log for ffmpeg's stderr."


@dataclass(frozen=True)
class VideoAssemblerConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crossfade_default_ms: int = 400
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "medium"


class VideoAssembler:
    """Build the final ``output.mp4`` (and ``output.srt``) from a finished job."""

    def __init__(
        self,
        config: VideoAssemblerConfig,
        *,
        bus: ProgressEventBus,
        runner: FFmpegRunner | None = None,
        ffmpeg_binary: str | None = None,
        write_subtitles: bool = True,
    ) -> None:
        self._config = config
        self._bus = bus
        self._runner = runner or _default_runner
        self._binary = ffmpeg_binary or shutil.which("ffmpeg") or "ffmpeg"
        self._write_subtitles = write_subtitles

    async def assemble(
        self,
        *,
        storyboard: Storyboard,
        audio_index: AudioIndex,
        images_index: ImagesIndex,
        job_dir: Path,
    ) -> Path:
        out_path = job_dir / "output.mp4"
        if out_path.is_file() and out_path.stat().st_size > 0:
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.INFO,
                    stage=Stage.ASSEMBLY.value,
                    message=f"reusing existing {out_path.name}",
                )
            )
            return out_path

        self._validate_inputs(storyboard, audio_index, images_index, job_dir)

        if self._write_subtitles:
            write_srt(
                storyboard=storyboard,
                audio_index=audio_index,
                out_path=job_dir / "output.srt",
            )
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.INFO,
                    stage=Stage.ASSEMBLY.value,
                    message="wrote output.srt",
                )
            )

        argv = self._build_command(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
            out_path=out_path,
        )
        log_path = job_dir / "logs" / "ffmpeg.log"
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.ASSEMBLY.value,
                message=f"running ffmpeg ({len(storyboard.shots)} shots)",
            )
        )
        try:
            await self._runner(argv, log_path)
        except FFmpegError:
            raise
        except Exception as exc:
            raise FFmpegError(f"ffmpeg invocation failed: {exc}") from exc

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise FFmpegError(f"ffmpeg completed but {out_path.name} was not produced")

        return out_path

    # ----------------------------------------------------------- internals

    def _validate_inputs(
        self,
        storyboard: Storyboard,
        audio_index: AudioIndex,
        images_index: ImagesIndex,
        job_dir: Path,
    ) -> None:
        if not storyboard.shots:
            raise FFmpegError("storyboard contains no shots; cannot assemble video")

        image_files = {item.shot_id: job_dir / item.file for item in images_index.items}
        audio_files = {item.shot_id: job_dir / item.file for item in audio_index.items}

        missing: list[str] = []
        for shot in storyboard.shots:
            if shot.id not in image_files or not image_files[shot.id].is_file():
                missing.append(f"image:{shot.id}")
            if shot.id not in audio_files or not audio_files[shot.id].is_file():
                missing.append(f"audio:{shot.id}")
        if missing:
            raise FFmpegError(
                "missing required shot artifacts before assembly: "
                + ", ".join(missing[:10])
                + ("…" if len(missing) > 10 else "")
            )

    def _build_command(
        self,
        *,
        storyboard: Storyboard,
        audio_index: AudioIndex,
        images_index: ImagesIndex,
        job_dir: Path,
        out_path: Path,
    ) -> list[str]:
        durations = {item.shot_id: max(0.5, item.duration_seconds) for item in audio_index.items}
        for shot in storyboard.shots:
            durations.setdefault(shot.id, max(0.5, shot.duration_seconds_target))

        image_paths = {item.shot_id: job_dir / item.file for item in images_index.items}
        audio_paths = {item.shot_id: job_dir / item.file for item in audio_index.items}

        argv: list[str] = [self._binary, "-y", "-hide_banner", "-loglevel", "error"]

        for shot in storyboard.shots:
            argv += [
                "-loop", "1",
                "-t", f"{durations[shot.id]:.3f}",
                "-i", str(image_paths[shot.id]),
            ]
        for shot in storyboard.shots:
            argv += ["-i", str(audio_paths[shot.id])]

        filter_complex = self._build_filtergraph(
            storyboard=storyboard,
            durations=durations,
        )
        argv += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]"]
        argv += [
            "-r", str(self._config.fps),
            "-c:v", self._config.video_codec,
            "-pix_fmt", "yuv420p",
            "-preset", self._config.preset,
            "-crf", str(self._config.crf),
            "-c:a", self._config.audio_codec,
            "-movflags", "+faststart",
            "-shortest",
            str(out_path),
        ]
        return argv

    def _build_filtergraph(
        self,
        *,
        storyboard: Storyboard,
        durations: dict[str, float],
    ) -> str:
        width = self._config.width
        height = self._config.height
        fps = self._config.fps

        video_filters: list[str] = []
        # Per-shot video pre-processing: scale + zoompan for Ken Burns.
        for idx, shot in enumerate(storyboard.shots):
            frames = max(1, int(durations[shot.id] * fps))
            frame_span = max(1, frames - 1)
            from_zoom = max(1.0, shot.ken_burns.from_[2])
            to_zoom = max(1.0, shot.ken_burns.to[2])
            zoom_expr = (
                f"{from_zoom:.4f}+(({to_zoom:.4f}-{from_zoom:.4f})*on/{frame_span})"
            )
            from_x = max(0.0, min(0.999, shot.ken_burns.from_[0]))
            to_x = max(0.0, min(0.999, shot.ken_burns.to[0]))
            from_y = max(0.0, min(0.999, shot.ken_burns.from_[1]))
            to_y = max(0.0, min(0.999, shot.ken_burns.to[1]))
            x_expr = f"iw*({from_x:.4f}+({to_x:.4f}-{from_x:.4f})*on/{frame_span})"
            y_expr = f"ih*({from_y:.4f}+({to_y:.4f}-{from_y:.4f})*on/{frame_span})"
            video_filters.append(
                # Contain-fit: scale within frame, pad with black so the full
                # image (including face on portrait sources) is visible.
                f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
                f"d={frames}:s={width}x{height}:fps={fps},"
                f"trim=duration={durations[shot.id]:.3f},setpts=PTS-STARTPTS"
                f"[v{idx}]"
            )

        # Crossfade chain on video.
        n = len(storyboard.shots)

        if n == 1:
            # Single-shot run: no chain, no concat — wire v0 + a0 directly.
            video_filters.append("[v0]copy[vout]")
            video_filters.append(f"[{n}:a]anull[aout]")
            return ";".join(video_filters)

        # xfade offset must be the running length of the *already-faded*
        # output stream, not the raw cumulative input duration. Track it
        # separately so non-uniform fades stay aligned.
        current_label = "v0"
        rendered_so_far = durations[storyboard.shots[0].id]
        for idx in range(1, n):
            fade_ms = max(
                storyboard.shots[idx - 1].crossfade_in_ms,
                storyboard.shots[idx].crossfade_out_ms,
                0,
            )
            fade_seconds = min(
                fade_ms / 1000.0,
                max(0.05, durations[storyboard.shots[idx - 1].id] - 0.05),
                max(0.05, durations[storyboard.shots[idx].id] - 0.05),
            )
            offset = max(0.0, rendered_so_far - fade_seconds)
            rendered_so_far = offset + durations[storyboard.shots[idx].id]
            out_label = "vout" if idx == n - 1 else f"vx{idx}"
            video_filters.append(
                f"[{current_label}][v{idx}]xfade=transition=fade:"
                f"duration={fade_seconds:.3f}:offset={offset:.3f}[{out_label}]"
            )
            current_label = out_label

        # Audio: concat (the audio has already been timed to each shot — TTS
        # output length is what drives shot duration). Use ``concat`` filter
        # rather than ``acrossfade`` to keep the implementation small; we accept
        # hard audio cuts in v1.
        audio_inputs = "".join(f"[{n + idx}:a]" for idx in range(n))
        video_filters.append(f"{audio_inputs}concat=n={n}:v=0:a=1[aout]")

        return ";".join(video_filters)


# -------------------------------------------------------------- default runner


async def _default_runner(argv: Sequence[str], log_path: Path) -> None:
    """Run ffmpeg via subprocess, capture stderr to ``log_path``."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    log_path.write_bytes(
        b"-- argv --\n"
        + shlex.join(argv).encode("utf-8", errors="replace")
        + b"\n-- stdout --\n"
        + (stdout or b"")
        + b"\n-- stderr --\n"
        + (stderr or b"")
    )
    if process.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited {process.returncode}; see {log_path} for full output"
        )
