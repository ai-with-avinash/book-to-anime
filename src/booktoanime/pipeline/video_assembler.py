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
    ChapterRecord,
    ChaptersIndex,
    ImagesIndex,
    Shot,
    ShotAudioRecord,
    ShotImageRecord,
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

        chapters_dir = job_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        chapter_groups = _group_by_topic(storyboard)
        chapter_records: list[ChapterRecord] = []

        for chapter_idx, (topic_id, shots) in enumerate(chapter_groups, start=1):
            chapter_filename = f"chapter_{chapter_idx:03d}.mp4"
            chapter_path = chapters_dir / chapter_filename
            chapter_srt_filename = f"chapter_{chapter_idx:03d}.srt"
            chapter_srt_path = chapters_dir / chapter_srt_filename

            sub_storyboard = _subset_storyboard(storyboard, shots, audio_index)
            sub_audio = _subset_audio(audio_index, shots)
            sub_images = _subset_images(images_index, shots)
            chapter_duration = sum(item.duration_seconds for item in sub_audio.items)

            if self._write_subtitles:
                write_srt(
                    storyboard=sub_storyboard,
                    audio_index=sub_audio,
                    out_path=chapter_srt_path,
                )

            await self._render_one(
                storyboard=sub_storyboard,
                audio_index=sub_audio,
                images_index=sub_images,
                job_dir=job_dir,
                out_path=chapter_path,
                log_path=job_dir / "logs" / f"ffmpeg_{chapter_filename}.log",
                label=f"chapter {chapter_idx}/{len(chapter_groups)} ({topic_id})",
            )

            chapter_records.append(
                ChapterRecord(
                    topic_id=topic_id,
                    order=chapter_idx,
                    file=str(chapter_path.relative_to(job_dir).as_posix()),
                    srt_file=str(chapter_srt_path.relative_to(job_dir).as_posix()),
                    duration_seconds=chapter_duration,
                )
            )

        ChaptersIndex(items=chapter_records).save(chapters_dir / "index.json")

        await self._concat_chapters(
            chapter_paths=[chapters_dir / f"chapter_{rec.order:03d}.mp4" for rec in chapter_records],
            out_path=out_path,
            log_path=job_dir / "logs" / "ffmpeg_concat.log",
        )

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise FFmpegError(f"ffmpeg completed but {out_path.name} was not produced")

        return out_path

    async def _render_one(
        self,
        *,
        storyboard: Storyboard,
        audio_index: AudioIndex,
        images_index: ImagesIndex,
        job_dir: Path,
        out_path: Path,
        log_path: Path,
        label: str,
    ) -> None:
        if out_path.is_file() and out_path.stat().st_size > 0:
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.INFO,
                    stage=Stage.ASSEMBLY.value,
                    message=f"reusing {label} ({out_path.name})",
                )
            )
            return

        argv = self._build_command(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
            out_path=out_path,
        )
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.ASSEMBLY.value,
                message=f"running ffmpeg for {label} ({len(storyboard.shots)} shots)",
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

    async def _concat_chapters(
        self,
        *,
        chapter_paths: Sequence[Path],
        out_path: Path,
        log_path: Path,
    ) -> None:
        if not chapter_paths:
            raise FFmpegError("no chapters to concat; cannot produce output.mp4")

        # ffmpeg's concat demuxer reads a text manifest. Using stream-copy
        # avoids re-encoding because every chapter shares the same codec
        # parameters from VideoAssemblerConfig.
        list_path = out_path.parent / "chapters" / "_concat.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text(
            "\n".join(f"file '{path.resolve().as_posix()}'" for path in chapter_paths) + "\n",
            encoding="utf-8",
        )

        argv = [
            self._binary,
            "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.ASSEMBLY.value,
                message=f"concatenating {len(chapter_paths)} chapter(s) into {out_path.name}",
            )
        )
        try:
            await self._runner(argv, log_path)
        except FFmpegError:
            raise
        except Exception as exc:
            raise FFmpegError(f"ffmpeg concat failed: {exc}") from exc

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
        # Per-shot video pre-processing: scale + zoompan for Ken Burns. All
        # shots use static images as of phase 1 — the lip-sync mp4 branch
        # was removed when the project pivoted away from character narration.
        for idx, shot in enumerate(storyboard.shots):
            apply_zoompan = True

            base_filter = (
                f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                f"fps={fps}"
            )

            if apply_zoompan:
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
                base_filter += (
                    f",zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
                    f"d={frames}:s={width}x{height}:fps={fps}"
                )

            video_filters.append(
                base_filter
                + f",trim=duration={durations[shot.id]:.3f},setpts=PTS-STARTPTS"
                + f"[v{idx}]"
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


# -------------------------------------------------------------- chapter helpers


def _group_by_topic(storyboard: Storyboard) -> list[tuple[str, list[Shot]]]:
    """Group shots by ``topic_id`` while preserving storyboard order.

    Returns a list of ``(topic_id, shots_in_order)`` tuples. Topics appear
    in the order their first shot appears, so the chapter sequence matches
    the original narrative flow.
    """

    groups: dict[str, list[Shot]] = {}
    order: list[str] = []
    for shot in storyboard.shots:
        if shot.topic_id not in groups:
            groups[shot.topic_id] = []
            order.append(shot.topic_id)
        groups[shot.topic_id].append(shot)
    return [(topic_id, groups[topic_id]) for topic_id in order]


def _subset_storyboard(
    storyboard: Storyboard,
    shots: Sequence[Shot],
    audio_index: AudioIndex,
) -> Storyboard:
    shot_ids = {shot.id for shot in shots}
    measured = {item.shot_id: item.duration_seconds for item in audio_index.items}
    total = 0.0
    for shot in shots:
        total += measured.get(shot.id, shot.duration_seconds_target)
    return Storyboard(
        schema_version=storyboard.schema_version,
        shots=[shot for shot in storyboard.shots if shot.id in shot_ids],
        total_duration_seconds_target=max(0.0, total),
    )


def _subset_audio(audio_index: AudioIndex, shots: Sequence[Shot]) -> AudioIndex:
    shot_ids = {shot.id for shot in shots}
    return AudioIndex(
        schema_version=audio_index.schema_version,
        items=[
            ShotAudioRecord(
                shot_id=item.shot_id,
                file=item.file,
                duration_seconds=item.duration_seconds,
                sample_rate=item.sample_rate,
            )
            for item in audio_index.items
            if item.shot_id in shot_ids
        ],
    )


def _subset_images(images_index: ImagesIndex, shots: Sequence[Shot]) -> ImagesIndex:
    shot_ids = {shot.id for shot in shots}
    return ImagesIndex(
        schema_version=images_index.schema_version,
        items=[
            ShotImageRecord(
                shot_id=item.shot_id,
                file=item.file,
                seed=item.seed,
                width=item.width,
                height=item.height,
            )
            for item in images_index.items
            if item.shot_id in shot_ids
        ],
    )


# -------------------------------------------------------------- default runner


async def _default_runner(argv: Sequence[str], log_path: Path) -> None:
    """Run ffmpeg via subprocess, streaming stderr directly to ``log_path``.

    ffmpeg's verbose stderr can grow to hundreds of megabytes on a long
    encode. Using :pyfunc:`asyncio.subprocess.Process.communicate` buffers
    the entire stderr stream in process RAM, which can OOM the orchestrator.
    Instead we wire ffmpeg's stderr fd straight at the open log file so the
    bytes never round-trip through Python.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = await asyncio.to_thread(open, str(log_path), "wb")
    try:
        log_handle.write(b"-- argv --\n")
        log_handle.write(shlex.join(argv).encode("utf-8", errors="replace"))
        log_handle.write(b"\n-- stderr --\n")
        log_handle.flush()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=log_handle,
        )
        returncode = await process.wait()
    finally:
        await asyncio.to_thread(log_handle.close)

    if returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited {returncode}; see {log_path} for full output"
        )
