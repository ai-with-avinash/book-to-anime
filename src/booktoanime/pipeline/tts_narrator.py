"""Per-shot TTS synthesis with bounded parallelism + filesystem-truthful resume.

Resume rule mirrors the image renderer: a shot is "done" if and only if both
``audio/<shot_id>.wav`` exists on disk **and** ``index.json`` lists it.
Crashes between file write and index persistence are healed at next start.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ProviderError
from ..providers import AudioProvider, TTSRequest
from .artifacts import AudioIndex, Shot, ShotAudioRecord, Storyboard
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .stages import Stage

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSNarratorConfig:
    voice_id: str
    language: str
    speed: float = 1.0
    concurrency: int = 2


class TTSNarrator:
    def __init__(
        self,
        provider: AudioProvider,
        config: TTSNarratorConfig,
        *,
        bus: ProgressEventBus,
    ) -> None:
        self._provider = provider
        self._config = config
        self._bus = bus

    async def synthesize(
        self,
        *,
        storyboard: Storyboard,
        job_dir: Path,
    ) -> AudioIndex:
        audio_dir = job_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        index_path = audio_dir / "index.json"

        completed = _reconcile_existing_index(index_path, storyboard, audio_dir)
        existing_ids = {record.shot_id for record in completed}

        semaphore = asyncio.Semaphore(max(1, self._config.concurrency))
        total = len(storyboard.shots)
        completed_count = len(existing_ids)
        new_records: dict[str, ShotAudioRecord] = {}
        first_error: BookToAnimeError | None = None

        async def synth_one(shot: Shot) -> None:
            nonlocal first_error
            if shot.id in existing_ids:
                return
            async with semaphore:
                if first_error is not None:
                    raise asyncio.CancelledError()
                try:
                    record = await self._synthesize_one(shot, audio_dir)
                except BookToAnimeError as exc:
                    if first_error is None:
                        first_error = exc
                    await self._bus.emit(
                        ProgressEvent(
                            kind=ProgressKind.SHOT_FAILED,
                            stage=Stage.AUDIO.value,
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
                    stage=Stage.AUDIO.value,
                    message=f"narrated {shot.id}",
                    shot_id=shot.id,
                )
            )

        tasks = [asyncio.create_task(synth_one(shot)) for shot in storyboard.shots]
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
                        f"unexpected error narrating shot {shot.id}: {result}"
                    )
                continue

        for shot in storyboard.shots:
            if shot.id in new_records:
                completed.append(new_records[shot.id])
                completed_count += 1

        AudioIndex(items=completed).save(index_path)
        if completed_count and total:
            await self._emit_progress_ratio(completed_count, total)

        if first_error is not None:
            raise first_error
        return AudioIndex(items=completed)

    async def _synthesize_one(self, shot: Shot, audio_dir: Path) -> ShotAudioRecord:
        out_path = audio_dir / f"{shot.id}.wav"
        request = TTSRequest(
            text=shot.narration_text,
            voice_id=self._config.voice_id,
            language=self._config.language,
            speed=self._config.speed,
        )
        try:
            generated = await self._provider.synthesize(request, out_path)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"unhandled audio error on shot {shot.id}: {exc}") from exc

        rel_file = f"audio/{out_path.name}"
        return ShotAudioRecord(
            shot_id=shot.id,
            file=rel_file,
            duration_seconds=generated.duration_seconds,
            sample_rate=generated.sample_rate,
        )

    async def _emit_progress_ratio(self, done: int, total: int) -> None:
        ratio = (done / total) if total else 1.0
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.AUDIO.value,
                message=f"audio: {done}/{total}",
                progress=ratio,
            )
        )


def _reconcile_existing_index(
    index_path: Path,
    storyboard: Storyboard,
    audio_dir: Path,
) -> list[ShotAudioRecord]:
    """Drop records whose file is missing; adopt orphan files on disk."""

    if not index_path.is_file():
        loaded: list[ShotAudioRecord] = []
    else:
        try:
            loaded = list(AudioIndex.from_path(index_path).items)
        except Exception as exc:
            _logger.warning(
                "ignoring unreadable audio index (will rebuild from filesystem): %s", exc
            )
            loaded = []

    by_id = {record.shot_id: record for record in loaded}
    storyboard_ids = {shot.id for shot in storyboard.shots}

    survived: list[ShotAudioRecord] = []
    for record in loaded:
        if record.shot_id not in storyboard_ids:
            continue
        full_path = audio_dir.parent / record.file
        if full_path.is_file():
            survived.append(record)

    survived_ids = {record.shot_id for record in survived}
    for shot in storyboard.shots:
        if shot.id in survived_ids or shot.id in by_id:
            continue
        on_disk = audio_dir / f"{shot.id}.wav"
        if on_disk.is_file():
            # Adopt the file with placeholder metadata; assembly will probe
            # actual duration via soundfile if it needs it.
            survived.append(
                ShotAudioRecord(
                    shot_id=shot.id,
                    file=f"audio/{on_disk.name}",
                    duration_seconds=0.0,
                    sample_rate=0,
                )
            )
    return survived
