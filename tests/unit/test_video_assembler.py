"""Unit tests for ``VideoAssembler`` using a stub ffmpeg runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

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
from booktoanime.pipeline.video_assembler import (
    FFmpegError,
    VideoAssembler,
    VideoAssemblerConfig,
)


def _shot(idx: int, *, duration: float = 4.0) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text=f"narration {idx}",
        duration_seconds_target=duration,
        image_prompt=f"prompt {idx}",
        seed=idx,
        ken_burns=KenBurns.model_validate({"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}),
    )


def _make_dirs(job_dir: Path, n: int) -> tuple[ImagesIndex, AudioIndex]:
    (job_dir / "images").mkdir(parents=True, exist_ok=True)
    (job_dir / "audio").mkdir(parents=True, exist_ok=True)
    image_records = []
    audio_records = []
    for i in range(1, n + 1):
        img_path = job_dir / "images" / f"shot_{i:04d}.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")
        wav_path = job_dir / "audio" / f"shot_{i:04d}.wav"
        wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
        image_records.append(
            ShotImageRecord(
                shot_id=f"shot_{i:04d}",
                file=f"images/shot_{i:04d}.png",
                seed=i,
                width=128,
                height=128,
            )
        )
        audio_records.append(
            ShotAudioRecord(
                shot_id=f"shot_{i:04d}",
                file=f"audio/shot_{i:04d}.wav",
                duration_seconds=3.0 + i,
                sample_rate=24_000,
            )
        )
    return ImagesIndex(items=image_records), AudioIndex(items=audio_records)


@pytest.mark.asyncio
async def test_assemble_invokes_runner_and_writes_subtitles(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 2)
    storyboard = Storyboard(
        shots=[_shot(1, duration=4.0), _shot(2, duration=5.0)],
        total_duration_seconds_target=9.0,
    )

    captured_argv: list[str] = []

    async def runner(argv: Sequence[str], log_path: Path) -> None:
        captured_argv.extend(argv)
        out_path = Path(argv[-1])
        out_path.write_bytes(b"\x00\x00\x00 ftypisom" + b"X" * 64)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=128, height=128, fps=24, crf=23, preset="ultrafast"),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )

    out_path = await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert out_path == job_dir / "output.mp4"
    assert out_path.is_file() and out_path.stat().st_size > 0
    # SRT sidecar produced.
    srt = (job_dir / "output.srt").read_text("utf-8")
    assert "narration 1" in srt and "narration 2" in srt
    assert "00:00:00,000" in srt
    # ffmpeg argv shape sanity.
    assert captured_argv[0] == "ffmpeg-stub"
    assert "-filter_complex" in captured_argv
    assert captured_argv[-1] == str(out_path)
    assert "-c:v" in captured_argv and "libx264" in captured_argv


@pytest.mark.asyncio
async def test_assemble_skips_when_output_exists(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 1)
    storyboard = Storyboard(
        shots=[_shot(1, duration=4.0)],
        total_duration_seconds_target=4.0,
    )
    pre_existing = job_dir / "output.mp4"
    pre_existing.write_bytes(b"already here" * 100)

    runner_calls = 0

    async def runner(argv: Sequence[str], log_path: Path) -> None:
        nonlocal runner_calls
        runner_calls += 1

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=64, height=64),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    out = await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    assert out == pre_existing
    assert runner_calls == 0


@pytest.mark.asyncio
async def test_assemble_missing_artifact_raises(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 1)
    # Storyboard references shot_0002, which has no on-disk artifacts.
    storyboard = Storyboard(
        shots=[_shot(1), _shot(2)],
        total_duration_seconds_target=8.0,
    )

    async def runner(argv: Sequence[str], log_path: Path) -> None:
        raise AssertionError("runner should not be invoked when inputs are missing")

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    with pytest.raises(FFmpegError, match="missing required shot artifacts"):
        await assembler.assemble(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
        )
    await bus.close()


@pytest.mark.asyncio
async def test_assemble_runner_failure_propagates(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 1)
    storyboard = Storyboard(
        shots=[_shot(1)],
        total_duration_seconds_target=4.0,
    )

    async def runner(argv: Sequence[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("simulated stderr", encoding="utf-8")
        raise FFmpegError("ffmpeg exited 1; see logs/ffmpeg.log")

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    with pytest.raises(FFmpegError):
        await assembler.assemble(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
        )
    await bus.close()


@pytest.mark.asyncio
async def test_filtergraph_n1_uses_anull(tmp_path: Path) -> None:
    """n=1 must NOT emit `concat=n=1` — that filter requires >=2 inputs."""

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 1)
    storyboard = Storyboard(
        shots=[_shot(1, duration=4.0)],
        total_duration_seconds_target=4.0,
    )

    captured: list[str] = []

    async def runner(argv, log_path):
        captured.extend(argv)
        out_path = Path(argv[-1])
        out_path.write_bytes(b"\x00\x00\x00 ftypisom" + b"X" * 64)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=128, height=128, fps=24, crf=23, preset="ultrafast"),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    fc_index = captured.index("-filter_complex") + 1
    filter_complex = captured[fc_index]
    assert "[v0]copy[vout]" in filter_complex
    assert "anull[aout]" in filter_complex
    assert "concat=n=1" not in filter_complex


@pytest.mark.asyncio
async def test_filtergraph_xfade_offsets_use_rendered_duration(tmp_path: Path) -> None:
    """For 3 shots with non-uniform fades, the xfade offset for shot 3 must
    equal `rendered_after_xfade_2 - fade_2_3`, not raw cumulative input time.
    """

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 3)
    # Override audio durations to known values.
    audio_index = AudioIndex(
        items=[
            ShotAudioRecord(shot_id="shot_0001", file="audio/shot_0001.wav", duration_seconds=10.0, sample_rate=24000),
            ShotAudioRecord(shot_id="shot_0002", file="audio/shot_0002.wav", duration_seconds=8.0, sample_rate=24000),
            ShotAudioRecord(shot_id="shot_0003", file="audio/shot_0003.wav", duration_seconds=6.0, sample_rate=24000),
        ]
    )
    storyboard = Storyboard(
        shots=[
            _shot(1, duration=10.0),
            _shot(2, duration=8.0),
            _shot(3, duration=6.0),
        ],
        total_duration_seconds_target=24.0,
    )

    captured: list[str] = []

    async def runner(argv, log_path):
        captured.extend(argv)
        Path(argv[-1]).write_bytes(b"X" * 100)

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(width=128, height=128, fps=24),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    await assembler.assemble(
        storyboard=storyboard,
        audio_index=audio_index,
        images_index=images_index,
        job_dir=job_dir,
    )
    await bus.close()

    fc_index = captured.index("-filter_complex") + 1
    filter_complex = captured[fc_index]
    # xfade_1 between v0 + v1: offset = dur(0) - fade = 10 - 0.4 = 9.6
    assert "offset=9.600" in filter_complex
    # xfade_2 between vx1 + v2: rendered_so_far = 9.6 + 8.0 = 17.6,
    #   then offset = 17.6 - fade(1,2) = 17.6 - 0.4 = 17.2
    assert "offset=17.200" in filter_complex


@pytest.mark.asyncio
async def test_assemble_fails_when_runner_succeeds_but_no_output(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    images_index, audio_index = _make_dirs(job_dir, 1)
    storyboard = Storyboard(
        shots=[_shot(1)],
        total_duration_seconds_target=4.0,
    )

    async def runner(argv: Sequence[str], log_path: Path) -> None:
        # Pretend success but never produce output.
        return None

    bus = ProgressEventBus(job_dir / "events.log")
    assembler = VideoAssembler(
        VideoAssemblerConfig(),
        bus=bus,
        runner=runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    with pytest.raises(FFmpegError, match="not produced"):
        await assembler.assemble(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
        )
    await bus.close()
