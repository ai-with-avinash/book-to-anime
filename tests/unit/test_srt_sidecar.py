"""Unit tests for the SRT sidecar writer."""

from __future__ import annotations

from pathlib import Path

from booktoanime.pipeline.artifacts import (
    AudioIndex,
    KenBurns,
    Shot,
    ShotAudioRecord,
    Storyboard,
)
from booktoanime.pipeline.srt_sidecar import _format_timestamp, write_srt


def _shot(idx: int, text: str, target: float = 5.0) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text=text,
        duration_seconds_target=target,
        image_prompt="p",
        seed=idx,
        ken_burns=KenBurns.model_validate({"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}),
    )


def test_format_timestamp() -> None:
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(1.5) == "00:00:01,500"
    assert _format_timestamp(3661.123) == "01:01:01,123"


def test_write_srt_uses_measured_durations(tmp_path: Path) -> None:
    storyboard = Storyboard(
        shots=[_shot(1, "Hello world", target=999.0), _shot(2, "Second line", target=999.0)],
        total_duration_seconds_target=10.0,
    )
    audio_index = AudioIndex(
        items=[
            ShotAudioRecord(shot_id="shot_0001", file="audio/shot_0001.wav", duration_seconds=2.5, sample_rate=24000),
            ShotAudioRecord(shot_id="shot_0002", file="audio/shot_0002.wav", duration_seconds=3.0, sample_rate=24000),
        ]
    )
    out = tmp_path / "out.srt"
    write_srt(storyboard=storyboard, audio_index=audio_index, out_path=out)

    text = out.read_text("utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in text
    assert "2\n00:00:02,500 --> 00:00:05,500\nSecond line" in text


def test_write_srt_falls_back_to_target_when_audio_missing(tmp_path: Path) -> None:
    storyboard = Storyboard(
        shots=[_shot(1, "Only shot", target=4.0)],
        total_duration_seconds_target=4.0,
    )
    audio_index = AudioIndex(items=[])  # no audio for this shot

    out = tmp_path / "out.srt"
    write_srt(storyboard=storyboard, audio_index=audio_index, out_path=out)

    text = out.read_text("utf-8")
    assert "00:00:00,000 --> 00:00:04,000" in text
