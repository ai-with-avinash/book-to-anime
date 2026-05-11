"""Storyboard builder.

Converts per-topic summaries into a flat list of shots. Each shot carries:

* The exact narration text the TTS stage will speak.
* An image-generation prompt baked with the chosen anime style fragment.
* A deterministic seed so re-runs produce identical images.
* A Ken Burns motion path used by the assembly stage.

We aim for ~7-9 second shots (per the approved plan) by splitting topic
narration into roughly that-sized text chunks at sentence boundaries, then
rounding word counts to seconds at 165 wpm.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .artifacts import (
    KenBurns,
    Shot,
    Storyboard,
    TopicSection,
)

_TARGET_SHOT_SECONDS = 8.0
_MIN_SHOT_SECONDS = 4.0
_MAX_SHOT_SECONDS = 14.0
_WORDS_PER_MINUTE = 165.0
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


_KEN_BURNS_PATTERNS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 1.0), (0.05, 0.05, 1.10)),
    ((0.05, 0.05, 1.10), (0.0, 0.0, 1.0)),
    ((0.0, 0.05, 1.05), (0.05, 0.0, 1.12)),
    ((0.03, 0.03, 1.08), (0.07, 0.07, 1.0)),
)


@dataclass(frozen=True)
class StoryboardConfig:
    panel_style: str
    base_seed: int = 12345
    crossfade_in_ms: int = 400
    crossfade_out_ms: int = 400


class StoryboardBuilder:
    def __init__(self, config: StoryboardConfig) -> None:
        self._config = config

    def build(
        self,
        topics: Sequence[TopicSection],
    ) -> Storyboard:
        shots: list[Shot] = []
        order = 1
        seed = self._config.base_seed
        total_seconds = 0.0

        for topic in topics:
            chunks = list(_chunk_narration(topic.summary))
            if not chunks:
                continue
            for chunk_idx, chunk in enumerate(chunks):
                duration = _seconds_for(chunk)
                total_seconds += duration
                shot = Shot(
                    id=f"shot_{order:04d}",
                    topic_id=topic.id,
                    order=order,
                    narration_text=chunk,
                    duration_seconds_target=duration,
                    image_prompt=self._image_prompt(topic, chunk, chunk_idx),
                    negative_prompt=None,
                    use_persona_reference=True,
                    ip_adapter_strength=0.65,
                    seed=seed,
                    ken_burns=self._ken_burns_for(order),
                    crossfade_in_ms=self._config.crossfade_in_ms,
                    crossfade_out_ms=self._config.crossfade_out_ms,
                    explains_image_id=topic.image_refs[0] if topic.image_refs else None,
                )
                shots.append(shot)
                order += 1
                seed += 1

        return Storyboard(
            shots=shots,
            total_duration_seconds_target=total_seconds,
        )

    # -------------------------------------------------------------- helpers

    def _image_prompt(self, topic: TopicSection, chunk: str, chunk_idx: int) -> str:
        excerpt = " ".join(chunk.split()[:30])
        first_keypoint = topic.key_points[0] if topic.key_points else ""
        focus = first_keypoint if chunk_idx == 0 and first_keypoint else excerpt
        return (
            f"{self._config.panel_style}, anime explainer scene illustrating "
            f"{topic.title.lower()}: {focus}"
        )

    def _ken_burns_for(self, order: int) -> KenBurns:
        pattern_index = (order - 1) % len(_KEN_BURNS_PATTERNS)
        from_, to = _KEN_BURNS_PATTERNS[pattern_index]
        return KenBurns.model_validate({"from": from_, "to": to})


def _chunk_narration(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_seconds = 0.0
    for sentence in sentences:
        seconds = _seconds_for(sentence)
        if (
            current
            and current_seconds + seconds > _MAX_SHOT_SECONDS
        ):
            chunks.append(" ".join(current))
            current = []
            current_seconds = 0.0
        current.append(sentence)
        current_seconds += seconds
        if current_seconds >= _TARGET_SHOT_SECONDS:
            chunks.append(" ".join(current))
            current = []
            current_seconds = 0.0

    if current:
        # Merge a too-short trailing chunk into the previous one when possible.
        if (
            chunks
            and current_seconds < _MIN_SHOT_SECONDS
        ):
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


def _seconds_for(text: str) -> float:
    words = max(1, len(text.split()))
    return max(_MIN_SHOT_SECONDS, words * 60.0 / _WORDS_PER_MINUTE)
