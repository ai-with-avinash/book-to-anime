"""Generate a ``.srt`` subtitle sidecar from the storyboard + audio index.

We use the *measured* per-shot audio duration (the TTS provider returns the
real length) rather than the storyboard's target so subtitles stay in sync
even when narration runs long or short.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import AudioIndex, Storyboard


def write_srt(
    *,
    storyboard: Storyboard,
    audio_index: AudioIndex,
    out_path: Path,
) -> None:
    """Write a UTF-8 SRT file at ``out_path``."""

    durations = {item.shot_id: item.duration_seconds for item in audio_index.items}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cursor = 0.0
    blocks: list[str] = []
    for index, shot in enumerate(storyboard.shots, start=1):
        duration = durations.get(shot.id) or shot.duration_seconds_target
        start = cursor
        end = cursor + max(0.05, duration)
        cursor = end
        blocks.append(
            f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n"
            f"{shot.narration_text.strip()}\n"
        )

    out_path.write_text("\n".join(blocks), encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
