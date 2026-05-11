"""Integration test exercising the real ffmpeg binary on a synthetic job.

These tests are gated behind ``@pytest.mark.real_ffmpeg`` so they only
run when explicitly requested (e.g. ``pytest -m real_ffmpeg``). When the
marker IS selected and ffmpeg is missing, the module fails loudly via
:pyfunc:`pytest.fail` rather than skipping — keeping CI honest.

What we verify:

* For ``n == 1`` shots the filtergraph fix from phase 4 holds — ffmpeg
  must produce a real MP4 instead of failing on ``concat=n=1``.
* For ``n == 3`` shots with non-uniform durations the xfade offset math
  produces an output whose duration matches the closed-form expected
  value within a 200 ms tolerance.
* For ``n == 10`` shots the cumulative crossfade chain stays in sync.

The duration assertion is the gate: if xfade offsets drift the encoded
video either freezes (offset too large) or skips (offset too small) and
its total runtime stops matching the expected ``sum(durations) - sum(fades)``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from booktoanime.pipeline.artifacts import (
    AudioIndex,
    ImagesIndex,
    KenBurns,
    Shot,
    ShotAudioRecord,
    ShotImageRecord,
    Storyboard,
)
from booktoanime.pipeline.events import ProgressEventBus
from booktoanime.pipeline.video_assembler import VideoAssembler, VideoAssemblerConfig

pytestmark = pytest.mark.real_ffmpeg


@pytest.fixture(autouse=True)
def _require_ffmpeg() -> None:
    """Hard-fail (not skip) when ffmpeg/ffprobe are missing.

    The marker keeps these tests deselected by default so a developer box
    without ffmpeg can still run the unit suite. But once the marker is
    selected the binary is non-optional — skipping would let CI quietly
    pass without exercising the real assembler.
    """

    if shutil.which("ffmpeg") is None:
        pytest.fail(
            "ffmpeg binary required for tests in this module. Install ffmpeg "
            "and ensure it is on PATH (e.g. `brew install ffmpeg` or "
            "`apt-get install ffmpeg`).",
            pytrace=False,
        )
    if shutil.which("ffprobe") is None:
        pytest.fail(
            "ffprobe binary required for tests in this module. Install ffmpeg "
            "(which ships ffprobe alongside).",
            pytrace=False,
        )


_SAMPLE_RATE = 24_000
_CROSSFADE_DEFAULT_MS = 400  # matches VideoAssemblerConfig default


def _write_black_png(path: Path, width: int = 320, height: int = 180) -> None:
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_silent_wav(path: Path, duration_seconds: float) -> None:
    n_samples = int(duration_seconds * _SAMPLE_RATE)
    samples = np.zeros(n_samples, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, _SAMPLE_RATE)


def _make_shot(idx: int, duration: float) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text=f"narration {idx}",
        duration_seconds_target=duration,
        image_prompt=f"prompt {idx}",
        seed=idx,
        ken_burns=KenBurns.model_validate(
            {"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}
        ),
    )


def _scaffold_job(job_dir: Path, durations: list[float]) -> tuple[Storyboard, AudioIndex, ImagesIndex]:
    """Materialise the on-disk image/audio fixtures for a job."""

    shots: list[Shot] = []
    image_records: list[ShotImageRecord] = []
    audio_records: list[ShotAudioRecord] = []

    for idx, dur in enumerate(durations, start=1):
        shot = _make_shot(idx, dur)
        shots.append(shot)

        img_path = job_dir / "images" / f"{shot.id}.png"
        _write_black_png(img_path)
        wav_path = job_dir / "audio" / f"{shot.id}.wav"
        _write_silent_wav(wav_path, dur)

        image_records.append(
            ShotImageRecord(
                shot_id=shot.id,
                file=f"images/{shot.id}.png",
                seed=shot.seed,
                width=320,
                height=180,
            )
        )
        audio_records.append(
            ShotAudioRecord(
                shot_id=shot.id,
                file=f"audio/{shot.id}.wav",
                duration_seconds=dur,
                sample_rate=_SAMPLE_RATE,
            )
        )

    storyboard = Storyboard(
        shots=shots,
        total_duration_seconds_target=sum(durations),
    )
    return (
        storyboard,
        AudioIndex(items=audio_records),
        ImagesIndex(items=image_records),
    )


def _probe_duration(mp4_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _expected_duration(durations: list[float]) -> float:
    """Closed-form expected duration of the xfade chain.

    Each adjacent pair of shots overlaps by the per-pair crossfade. With
    uniform fades, total = sum(durations) - (n-1) * fade.
    """

    total = sum(durations)
    fade_seconds = _CROSSFADE_DEFAULT_MS / 1000.0
    overlap = max(0, len(durations) - 1) * fade_seconds
    return total - overlap


@pytest.mark.asyncio
async def test_real_ffmpeg_single_shot(tmp_path: Path) -> None:
    """n=1 must produce a valid MP4 without invoking ``concat=n=1``."""

    job_dir = tmp_path / "job_n1"
    job_dir.mkdir()
    durations = [1.0]
    storyboard, audio_index, images_index = _scaffold_job(job_dir, durations)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=320, height=180, fps=24, crf=28, preset="ultrafast"),
        bus=bus,
    )
    out_path = await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    actual = _probe_duration(out_path)
    expected = _expected_duration(durations)
    assert abs(actual - expected) < 0.2, (
        f"n=1 duration drift: expected {expected:.3f}s, got {actual:.3f}s"
    )


@pytest.mark.asyncio
async def test_real_ffmpeg_three_shots(tmp_path: Path) -> None:
    """n=3 with non-uniform durations stays in sync under the fixed xfade math."""

    job_dir = tmp_path / "job_n3"
    job_dir.mkdir()
    durations = [1.0, 2.0, 3.0]
    storyboard, audio_index, images_index = _scaffold_job(job_dir, durations)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=320, height=180, fps=24, crf=28, preset="ultrafast"),
        bus=bus,
    )
    out_path = await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    actual = _probe_duration(out_path)
    expected = _expected_duration(durations)
    assert abs(actual - expected) < 0.2, (
        f"n=3 duration drift: expected {expected:.3f}s, got {actual:.3f}s"
    )


@pytest.mark.asyncio
async def test_real_ffmpeg_ten_shots(tmp_path: Path) -> None:
    """n=10 stresses the xfade chain across many shots."""

    job_dir = tmp_path / "job_n10"
    job_dir.mkdir()
    # Mixed durations so xfade math is exercised, not just uniform.
    durations = [1.0, 1.5, 2.0, 1.2, 1.8, 2.5, 1.0, 1.3, 1.7, 2.0]
    storyboard, audio_index, images_index = _scaffold_job(job_dir, durations)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=320, height=180, fps=24, crf=28, preset="ultrafast"),
        bus=bus,
    )
    out_path = await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    actual = _probe_duration(out_path)
    expected = _expected_duration(durations)
    assert abs(actual - expected) < 0.2, (
        f"n=10 duration drift: expected {expected:.3f}s, got {actual:.3f}s"
    )
