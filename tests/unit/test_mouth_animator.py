"""Unit tests for ``MouthAnimator`` using a stub :class:`LipSyncProvider`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from booktoanime.errors import ProviderError
from booktoanime.pipeline.artifacts import (
    AudioIndex,
    ImagesIndex,
    KenBurns,
    MouthIndex,
    Shot,
    ShotAudioRecord,
    ShotImageRecord,
    Storyboard,
)
from booktoanime.pipeline.events import ProgressEventBus
from booktoanime.pipeline.mouth_animator import MouthAnimator, MouthAnimatorConfig
from booktoanime.providers.base import AnimatedShot, LipSyncProvider


class _StubLipSync(LipSyncProvider):
    """Writes a fake mp4 file at ``out_path`` and reports a fixed duration."""

    name = "stub_lipsync"

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.calls: list[tuple[Path, Path, Path]] = []
        self._fail_on = fail_on or set()

    async def animate(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        out_path: Path,
    ) -> AnimatedShot:
        # Identify which shot this is by output filename suffix.
        shot_id = out_path.stem
        if shot_id in self._fail_on:
            raise ProviderError(f"forced failure for {shot_id}")
        self.calls.append((image_path, audio_path, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00\x00\x00 ftypisomFAKEMOUTHMP4")
        return AnimatedShot(path=out_path, duration_seconds=4.2, fps=25.0)

    async def close(self) -> None:
        return None


def _shot(idx: int) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text=f"narration {idx}",
        duration_seconds_target=4.0,
        image_prompt=f"prompt {idx}",
        seed=idx,
        ken_burns=KenBurns.model_validate(
            {"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}
        ),
    )


def _populate_inputs(job_dir: Path, n: int) -> tuple[ImagesIndex, AudioIndex]:
    (job_dir / "images").mkdir(parents=True, exist_ok=True)
    (job_dir / "audio").mkdir(parents=True, exist_ok=True)
    images = []
    audios = []
    for i in range(1, n + 1):
        (job_dir / "images" / f"shot_{i:04d}.png").write_bytes(b"\x89PNG fake")
        (job_dir / "audio" / f"shot_{i:04d}.wav").write_bytes(b"RIFF fake WAV")
        images.append(
            ShotImageRecord(
                shot_id=f"shot_{i:04d}",
                file=f"images/shot_{i:04d}.png",
                seed=i,
                width=128,
                height=128,
            )
        )
        audios.append(
            ShotAudioRecord(
                shot_id=f"shot_{i:04d}",
                file=f"audio/shot_{i:04d}.wav",
                duration_seconds=4.0,
                sample_rate=24_000,
            )
        )
    return ImagesIndex(items=images), AudioIndex(items=audios)


@pytest.mark.asyncio
async def test_animate_calls_provider_per_shot_and_writes_index(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 2)
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2)],
        total_duration_seconds_target=8.0,
    )

    provider = _StubLipSync()
    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        provider,
        MouthAnimatorConfig(concurrency=1, fps=25.0),
        bus=bus,
    )

    index = await animator.animate(
        storyboard=storyboard,
        images_index=images_index,
        audio_index=audio_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert len(provider.calls) == 2
    assert {p.shot_id for p in index.items} == {"shot_0001", "shot_0002"}
    persisted = MouthIndex.from_path(job_dir / "mouth" / "index.json")
    assert {p.shot_id for p in persisted.items} == {"shot_0001", "shot_0002"}
    for record in persisted.items:
        assert (job_dir / record.file).is_file()
        # Provider returned 4.2; ffprobe is not installed in tests so the
        # fallback to the provider's reported duration must apply.
        assert record.duration_seconds in (4.2, pytest.approx(4.2))


@pytest.mark.asyncio
async def test_animate_skips_already_rendered_shots(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 2)
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2)],
        total_duration_seconds_target=8.0,
    )
    mouth_dir = job_dir / "mouth"
    mouth_dir.mkdir(parents=True)
    pre = mouth_dir / "shot_0001.mp4"
    pre.write_bytes(b"already-animated")
    # Persist matching index entry so reconcile keeps it.
    MouthIndex.model_validate(
        {
            "schema_version": 1,
            "items": [
                {
                    "shot_id": "shot_0001",
                    "file": "mouth/shot_0001.mp4",
                    "duration_seconds": 4.0,
                    "fps": 25.0,
                }
            ],
        }
    ).save(mouth_dir / "index.json")

    provider = _StubLipSync()
    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        provider,
        MouthAnimatorConfig(concurrency=1, fps=25.0),
        bus=bus,
    )
    index = await animator.animate(
        storyboard=storyboard,
        images_index=images_index,
        audio_index=audio_index,
        job_dir=job_dir,
    )
    await bus.close()

    # Provider was only called for the missing shot.
    assert len(provider.calls) == 1
    assert provider.calls[0][2].name == "shot_0002.mp4"
    assert {p.shot_id for p in index.items} == {"shot_0001", "shot_0002"}


@pytest.mark.asyncio
async def test_animate_missing_inputs_raises(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 1)
    # Storyboard references shot_0002 which has no inputs.
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2)],
        total_duration_seconds_target=8.0,
    )
    provider = _StubLipSync()
    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        provider,
        MouthAnimatorConfig(concurrency=1),
        bus=bus,
    )
    with pytest.raises(ProviderError, match="missing inputs for mouth animation"):
        await animator.animate(
            storyboard=storyboard,
            images_index=images_index,
            audio_index=audio_index,
            job_dir=job_dir,
        )
    await bus.close()


@pytest.mark.asyncio
async def test_animate_persists_partial_progress_on_failure(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 3)
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2), _shot(3)],
        total_duration_seconds_target=12.0,
    )

    provider = _StubLipSync(fail_on={"shot_0002"})
    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        provider,
        MouthAnimatorConfig(concurrency=1),
        bus=bus,
    )
    with pytest.raises(ProviderError, match="forced failure for shot_0002"):
        await animator.animate(
            storyboard=storyboard,
            images_index=images_index,
            audio_index=audio_index,
            job_dir=job_dir,
        )
    await bus.close()

    # shot_0001 must be persisted in the index even though shot_0002 failed.
    persisted = MouthIndex.from_path(job_dir / "mouth" / "index.json")
    saved_ids = {p.shot_id for p in persisted.items}
    assert "shot_0001" in saved_ids
    # Failure on shot_0002 cancels the rest, so shot_0003 may or may not appear.


@pytest.mark.asyncio
async def test_animate_propagates_unexpected_error_as_provider_error(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 1)
    storyboard = Storyboard(
        shots=[_shot(1)],
        total_duration_seconds_target=4.0,
    )

    class _RaisingProvider(LipSyncProvider):
        name = "raising"

        async def animate(self, *, image_path: Path, audio_path: Path, out_path: Path) -> AnimatedShot:
            raise RuntimeError("kaboom")

        async def close(self) -> None:
            return None

    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        _RaisingProvider(),
        MouthAnimatorConfig(concurrency=1),
        bus=bus,
    )
    with pytest.raises(ProviderError, match="unhandled lip-sync error"):
        await animator.animate(
            storyboard=storyboard,
            images_index=images_index,
            audio_index=audio_index,
            job_dir=job_dir,
        )
    await bus.close()


@pytest.mark.asyncio
async def test_animate_concurrency_respects_semaphore(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _populate_inputs(job_dir, 4)
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2), _shot(3), _shot(4)],
        total_duration_seconds_target=16.0,
    )

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    class _CountingProvider(LipSyncProvider):
        name = "counting"

        async def animate(self, *, image_path: Path, audio_path: Path, out_path: Path) -> AnimatedShot:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.01)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"\x00\x00\x00 ftypisomFAKE")
                return AnimatedShot(path=out_path, duration_seconds=4.0, fps=25.0)
            finally:
                async with lock:
                    in_flight -= 1

        async def close(self) -> None:
            return None

    bus = ProgressEventBus(job_dir / "events.log")
    animator = MouthAnimator(
        _CountingProvider(),
        MouthAnimatorConfig(concurrency=2),
        bus=bus,
    )
    await animator.animate(
        storyboard=storyboard,
        images_index=images_index,
        audio_index=audio_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert peak <= 2
