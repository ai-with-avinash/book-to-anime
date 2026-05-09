"""Stage enum + ordering."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Pipeline stages, in the order they run.

    Resume rule: a job picks up at the first stage whose status is not
    ``completed`` according to ``manifest.json``. Each stage is responsible
    for its own per-shot resume (e.g. the images stage skips shots that
    already have a file in ``images/``).
    """

    PARSING = "parsing"
    STRUCTURING = "structuring"
    STORYBOARD = "storyboard"
    IMAGES = "images"
    AUDIO = "audio"
    MOUTH_ANIMATION = "mouth_animation"
    ASSEMBLY = "assembly"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.PARSING,
    Stage.STRUCTURING,
    Stage.STORYBOARD,
    Stage.IMAGES,
    Stage.AUDIO,
    Stage.MOUTH_ANIMATION,
    Stage.ASSEMBLY,
)


def next_stage(stage: Stage) -> Stage | None:
    """Return the stage that follows ``stage`` in the canonical order."""

    idx = STAGE_ORDER.index(stage)
    if idx + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[idx + 1]
