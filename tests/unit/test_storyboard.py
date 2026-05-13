"""Storyboard builder coverage for phase-2 prompt + visual_kind rewrite."""

from __future__ import annotations

import pytest

from booktoanime.pipeline.artifacts import TopicSection, VisualKind
from booktoanime.pipeline.storyboard import (
    StoryboardBuilder,
    StoryboardConfig,
    _sentence_clean,
)


def _topic(
    *,
    title: str = "Quantum Entanglement",
    summary: str,
    key_points: list[str] | None = None,
    image_refs: list[str] | None = None,
) -> TopicSection:
    return TopicSection(
        id="topic_001",
        title=title,
        page_range=(0, 0),
        summary=summary,
        key_points=key_points or [],
        image_refs=image_refs or [],
        estimated_narration_seconds=30.0,
    )


def _build(panel_style: str = "clean-linework") -> StoryboardBuilder:
    return StoryboardBuilder(StoryboardConfig(panel_style=panel_style))


def _multi_sentence_summary(sentence_count: int) -> str:
    # Each sentence is sized so the storyboard chunker emits one shot per
    # sentence at the 8-second target (with ~22 words per sentence ≈ 8s).
    chunk = (
        "We carefully explore one key idea in depth here using sustained prose "
        "phrasing across many concrete words and clauses today as the chapter."
    )
    return " ".join(f"{chunk}" for _ in range(sentence_count))


# --------------------------------------------------------------- prompt format


def test_image_prompt_uses_educational_illustration_template() -> None:
    builder = _build("clean-linework")
    topic = _topic(
        title="Quantum Entanglement.",
        summary=_multi_sentence_summary(1),
        key_points=["pairs are correlated"],
    )

    storyboard = builder.build([topic])

    assert storyboard.shots, "expected at least one shot"
    prompt = storyboard.shots[0].image_prompt
    assert prompt.startswith("educational illustration of "), prompt
    # Title is sentence-cleaned (trailing period stripped) + lowercased.
    assert "quantum entanglement:" in prompt
    assert "anime" not in prompt.lower()
    # Style fragment is baked in.
    assert "clean line art illustration" in prompt


def test_image_prompt_truncates_long_title_to_60_chars() -> None:
    builder = _build()
    long_title = (
        "An extraordinarily verbose chapter title that keeps going beyond the "
        "sixty character truncation limit "
    )
    topic = _topic(
        title=long_title,
        summary=_multi_sentence_summary(1),
    )

    storyboard = builder.build([topic])
    prompt = storyboard.shots[0].image_prompt
    # Grab the segment between the literal "of " and ":".
    head, _, _ = prompt.partition(":")
    title_chunk = head.removeprefix("educational illustration of ")
    assert len(title_chunk) <= 60


# --------------------------------------------------------------- visual_kind


def test_three_shots_with_image_refs_yields_title_card_then_figure_then_illustration() -> None:
    builder = _build()
    topic = _topic(
        summary=_multi_sentence_summary(3),
        image_refs=["fig_alpha"],
    )

    storyboard = builder.build([topic])
    assert len(storyboard.shots) == 3
    kinds = [s.visual_kind for s in storyboard.shots]
    assert kinds[0] == VisualKind.TITLE_CARD
    assert kinds[1] == VisualKind.FIGURE
    assert kinds[2] == VisualKind.ILLUSTRATION
    # figure_id points at the stable ExtractedImage.image_id, not a list index.
    assert storyboard.shots[1].figure_id == "fig_alpha"
    assert storyboard.shots[0].figure_id is None
    assert storyboard.shots[2].figure_id is None


def test_two_shots_no_image_refs_are_both_illustration() -> None:
    builder = _build()
    topic = _topic(summary=_multi_sentence_summary(2))

    storyboard = builder.build([topic])
    assert len(storyboard.shots) == 2
    assert all(s.visual_kind == VisualKind.ILLUSTRATION for s in storyboard.shots)
    assert all(s.figure_id is None for s in storyboard.shots)


def test_two_shots_with_image_refs_yields_figure_then_illustration_no_title_card() -> None:
    builder = _build()
    topic = _topic(
        summary=_multi_sentence_summary(2),
        image_refs=["fig_only"],
    )

    storyboard = builder.build([topic])
    assert len(storyboard.shots) == 2
    # Topic with ≤2 shots skips the title card.
    assert VisualKind.TITLE_CARD not in {s.visual_kind for s in storyboard.shots}
    assert storyboard.shots[0].visual_kind == VisualKind.FIGURE
    assert storyboard.shots[0].figure_id == "fig_only"
    assert storyboard.shots[1].visual_kind == VisualKind.ILLUSTRATION


def test_five_shots_with_two_image_refs_yields_one_title_two_figures_two_illustrations() -> None:
    builder = _build()
    topic = _topic(
        summary=_multi_sentence_summary(5),
        image_refs=["fig_a", "fig_b"],
    )

    storyboard = builder.build([topic])
    assert len(storyboard.shots) == 5

    kinds = [s.visual_kind for s in storyboard.shots]
    assert kinds[0] == VisualKind.TITLE_CARD
    assert kinds[1] == VisualKind.FIGURE
    assert kinds[2] == VisualKind.FIGURE
    assert kinds[3] == VisualKind.ILLUSTRATION
    assert kinds[4] == VisualKind.ILLUSTRATION

    figure_ids = [s.figure_id for s in storyboard.shots]
    assert figure_ids == [None, "fig_a", "fig_b", None, None]


def test_each_image_ref_only_assigned_once() -> None:
    """Same ExtractedImage.image_id never appears on two shots."""

    builder = _build()
    topic = _topic(
        summary=_multi_sentence_summary(4),
        image_refs=["fig_one"],
    )

    storyboard = builder.build([topic])
    figure_ids = [s.figure_id for s in storyboard.shots if s.figure_id]
    assert figure_ids == ["fig_one"]


# --------------------------------------------------------------- _sentence_clean


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello world.", "Hello world"),
        ("   Trailing spaces   ", "Trailing spaces"),
        ("Multiple\n\nlines\tand\ttabs", "Multiple lines and tabs"),
        ("No period", "No period"),
        ("Multi.period..ends...", "Multi.period..ends"),
        ("", ""),
    ],
)
def test_sentence_clean(raw: str, expected: str) -> None:
    assert _sentence_clean(raw) == expected
