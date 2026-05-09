"""Per-shot mouth animation with bounded parallelism + filesystem-truthful resume.

Mirrors :mod:`booktoanime.pipeline.image_renderer` so the resume / failure
semantics match the rest of the pipeline:

* a shot is "done" iff ``mouth/<shot_id>.mp4`` exists **and** ``index.json``
  lists it,
* on entry the index is reconciled against the filesystem (deleted files
  re-render; orphaned files are adopted via ffprobe),
* on partial failure we collect what we have, persist it, and re-raise the
  first :class:`BookToAnimeError`.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ProviderError
from ..providers.base import LipSyncProvider
from .artifacts import (
    AudioIndex,
    ImagesIndex,
    MouthIndex,
    MouthShotRecord,
    Shot,
    Storyboard,
)
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .stages import Stage

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MouthAnimatorConfig:
    concurrency: int = 1
    fps: float = 25.0


class MouthAnimator:
    def __init__(
        self,
        provider: LipSyncProvider,
        config: MouthAnimatorConfig,
        *,
        bus: ProgressEventBus,
    ) -> None:
        self._provider = provider
        self._config = config
        self._bus = bus

    async def animate(
        self,
        *,
        storyboard: Storyboard,
        images_index: ImagesIndex,
        audio_index: AudioIndex,
        job_dir: Path,
    ) -> MouthIndex:
        mouth_dir = job_dir / "mouth"
        mouth_dir.mkdir(parents=True, exist_ok=True)
        index_path = mouth_dir / "index.json"

        completed = _reconcile_existing_index(index_path, storyboard, mouth_dir, self._config.fps)
        existing_ids = {record.shot_id for record in completed}

        image_paths = _index_paths(images_index, job_dir)
        audio_paths = _index_paths(audio_index, job_dir)
        missing: list[str] = []
        for shot in storyboard.shots:
            if shot.id in existing_ids:
                continue
            if shot.id not in image_paths or not image_paths[shot.id].is_file():
                missing.append(f"image:{shot.id}")
            if shot.id not in audio_paths or not audio_paths[shot.id].is_file():
                missing.append(f"audio:{shot.id}")
        if missing:
            raise ProviderError(
                "missing inputs for mouth animation: "
                + ", ".join(missing[:10])
                + ("…" if len(missing) > 10 else "")
            )

        semaphore = asyncio.Semaphore(max(1, self._config.concurrency))
        total = len(storyboard.shots)
        completed_count = len(existing_ids)
        new_records: dict[str, MouthShotRecord] = {}
        first_error: BookToAnimeError | None = None

        async def animate_one(shot: Shot) -> None:
            nonlocal first_error
            if shot.id in existing_ids:
                return
            async with semaphore:
                if first_error is not None:
                    raise asyncio.CancelledError()
                try:
                    record = await self._animate_one(shot, image_paths[shot.id], audio_paths[shot.id], mouth_dir)
                except BookToAnimeError as exc:
                    if first_error is None:
                        first_error = exc
                    await self._bus.emit(
                        ProgressEvent(
                            kind=ProgressKind.SHOT_FAILED,
                            stage=Stage.MOUTH_ANIMATION.value,
                            message=str(exc),
                            shot_id=shot.id,
                            user_message=exc.user_message,
                        )
                    )
                    raise
            new_records[shot.id] = record
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.SHOT_COMPLETED,
                    stage=Stage.MOUTH_ANIMATION.value,
                    message=f"animated {shot.id}",
                    shot_id=shot.id,
                )
            )

        tasks = [asyncio.create_task(animate_one(shot)) for shot in storyboard.shots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for shot, result in zip(storyboard.shots, results, strict=True):
            if isinstance(result, BookToAnimeError):
                if first_error is None:
                    first_error = result
                continue
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                if first_error is None:
                    first_error = ProviderError(
                        f"unexpected error animating shot {shot.id}: {result}"
                    )
                continue

        for shot in storyboard.shots:
            if shot.id in new_records:
                completed.append(new_records[shot.id])
                completed_count += 1

        MouthIndex(items=completed).save(index_path)
        if completed_count and total:
            await self._emit_progress_ratio(completed_count, total)

        if first_error is not None:
            raise first_error

        return MouthIndex(items=completed)

    async def _animate_one(
        self,
        shot: Shot,
        image_path: Path,
        audio_path: Path,
        mouth_dir: Path,
    ) -> MouthShotRecord:
        out_path = mouth_dir / f"{shot.id}.mp4"
        try:
            animated = await self._provider.animate(
                image_path=image_path,
                audio_path=audio_path,
                out_path=out_path,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"unhandled lip-sync error on shot {shot.id}: {exc}") from exc

        # Providers may report best-effort durations (or zero). Re-measure
        # with ffprobe so the assembler's xfade math stays accurate.
        measured = await _ffprobe_duration(out_path)
        duration = measured if measured > 0.0 else animated.duration_seconds
        return MouthShotRecord(
            shot_id=shot.id,
            file=f"mouth/{out_path.name}",
            duration_seconds=max(0.0, duration),
            fps=animated.fps if animated.fps > 0 else self._config.fps,
        )

    async def _emit_progress_ratio(self, done: int, total: int) -> None:
        ratio = (done / total) if total else 1.0
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.MOUTH_ANIMATION.value,
                message=f"mouth_animation: {done}/{total}",
                progress=ratio,
            )
        )


def _index_paths(
    index: ImagesIndex | AudioIndex,
    job_dir: Path,
) -> dict[str, Path]:
    return {item.shot_id: job_dir / item.file for item in index.items}


def _reconcile_existing_index(
    index_path: Path,
    storyboard: Storyboard,
    mouth_dir: Path,
    default_fps: float,
) -> list[MouthShotRecord]:
    if not index_path.is_file():
        loaded: list[MouthShotRecord] = []
    else:
        try:
            loaded = list(MouthIndex.from_path(index_path).items)
        except Exception as exc:
            _logger.warning(
                "ignoring unreadable mouth index (will rebuild from filesystem): %s",
                exc,
            )
            loaded = []

    by_id = {record.shot_id: record for record in loaded}
    storyboard_ids = {shot.id for shot in storyboard.shots}

    survived: list[MouthShotRecord] = []
    for record in loaded:
        if record.shot_id not in storyboard_ids:
            continue
        full_path = mouth_dir.parent / record.file
        if full_path.is_file() and full_path.stat().st_size > 0:
            survived.append(record)

    survived_ids = {record.shot_id for record in survived}
    for shot in storyboard.shots:
        if shot.id in survived_ids or shot.id in by_id:
            continue
        on_disk = mouth_dir / f"{shot.id}.mp4"
        if on_disk.is_file() and on_disk.stat().st_size > 0:
            survived.append(
                MouthShotRecord(
                    shot_id=shot.id,
                    file=f"mouth/{on_disk.name}",
                    duration_seconds=0.0,
                    fps=default_fps,
                )
            )
    return survived


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
