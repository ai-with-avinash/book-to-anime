"""Tests for the Pillow-based panel composer.

Covers:

* aspect-aware dispatch (landscape → bottom-strip; portrait → side-panel),
* title truncation + sentence-cleaning,
* palette correctness via background-pixel spot check,
* font-load failure surfaces as :class:`RenderError`,
* unknown ``panel_style`` raises :class:`RenderError`.

No pixel-hash snapshots — output is sampled by region + colour bucket.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageFont

from booktoanime.errors import RenderError
from booktoanime.pipeline import panel_composer
from booktoanime.pipeline.panel_composer import (
    _sentence_clean,
    compose_figure_panel,
    compose_title_card,
)

_PALETTE_BG = {
    "clean-linework": (255, 255, 255),
    "chalkboard-sketch": (26, 58, 46),
    "watercolor-technical": (245, 235, 214),
    "flat-vector-infographic": (232, 232, 232),
}


def _make_figure(path: Path, size: tuple[int, int]) -> None:
    """Write a flat-coloured PNG to ``path``. Colour stays away from any palette BG."""

    Image.new("RGB", size, (200, 100, 50)).save(path)


# ----------------------------------------------------------- figure panel layout


def test_compose_figure_panel_landscape_layout(tmp_path: Path) -> None:
    fig_path = tmp_path / "land.png"
    _make_figure(fig_path, (320, 180))

    out = compose_figure_panel(
        figure_path=fig_path,
        caption="A short caption describing the figure.",
        title="Newton's First Law",
        panel_style="clean-linework",
        target_size=(1920, 1080),
    )
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (1920, 1080)

    # Bottom-strip layout: figure occupies the top ``height - 280`` pixels.
    # A pixel deep in the figure region should NOT be the background colour
    # because the figure colour (200, 100, 50) is far from white.
    fig_region_pixel = out.getpixel((960, 200))
    assert fig_region_pixel != _PALETTE_BG["clean-linework"]

    # The caption strip background must be the panel-style BG colour.
    bottom_pixel = out.getpixel((10, 1075))
    assert bottom_pixel == _PALETTE_BG["clean-linework"]


def test_compose_figure_panel_portrait_layout(tmp_path: Path) -> None:
    fig_path = tmp_path / "port.png"
    _make_figure(fig_path, (180, 320))

    out = compose_figure_panel(
        figure_path=fig_path,
        caption="Side-panel caption text.",
        title="Conservation of Energy",
        panel_style="watercolor-technical",
        target_size=(1920, 1080),
    )
    assert out.size == (1920, 1080)

    # Side-panel layout: figure on the left square block.
    fig_side = min(1920, 1080)  # 1080
    # Right of the figure block should be the BG colour.
    right_pixel = out.getpixel((fig_side + 100, 540))
    assert right_pixel == _PALETTE_BG["watercolor-technical"]


def test_compose_figure_panel_unknown_panel_style_raises(tmp_path: Path) -> None:
    fig_path = tmp_path / "fig.png"
    _make_figure(fig_path, (256, 256))

    with pytest.raises(RenderError):
        compose_figure_panel(
            figure_path=fig_path,
            caption="x",
            title="t",
            panel_style="nonexistent-style",
        )


def test_compose_figure_panel_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RenderError):
        compose_figure_panel(
            figure_path=tmp_path / "missing.png",
            caption="x",
            title="t",
            panel_style="clean-linework",
        )


# ----------------------------------------------------------- title card


def test_compose_title_card_basic_dimensions() -> None:
    out = compose_title_card(
        title="Chapter One",
        subtitle="An introduction",
        panel_style="chalkboard-sketch",
        target_size=(1920, 1080),
    )
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (1920, 1080)


def test_compose_title_card_unknown_style_raises() -> None:
    with pytest.raises(RenderError):
        compose_title_card(
            title="t", subtitle="s", panel_style="bogus", target_size=(320, 180)
        )


# ----------------------------------------------------------- palette spot checks


@pytest.mark.parametrize(
    "style,expected_bg",
    [
        ("clean-linework", (255, 255, 255)),
        ("chalkboard-sketch", (26, 58, 46)),
        ("watercolor-technical", (245, 235, 214)),
        ("flat-vector-infographic", (232, 232, 232)),
    ],
)
def test_title_card_bg_palette(style: str, expected_bg: tuple[int, int, int]) -> None:
    out = compose_title_card(
        title="x", subtitle="y", panel_style=style, target_size=(320, 180)
    )
    # Top-left corner is far from the centred title text — must be BG.
    assert out.getpixel((0, 0)) == expected_bg


# ----------------------------------------------------------- text helpers


def test_sentence_clean_strips_and_truncates() -> None:
    out = _sentence_clean("  hello    world.   ")
    assert out == "hello world"


def test_sentence_clean_truncates_overlong() -> None:
    long = "a" * 100
    out = _sentence_clean(long, max_chars=60)
    assert len(out) <= 60
    assert out.endswith("…")


# ----------------------------------------------------------- font load failure


def test_font_load_failure_raises_render_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_kw: object) -> ImageFont.FreeTypeFont:
        raise OSError("font corrupt")

    monkeypatch.setattr(panel_composer.ImageFont, "truetype", boom)

    with pytest.raises(RenderError):
        compose_title_card(
            title="x",
            subtitle="y",
            panel_style="clean-linework",
            target_size=(320, 180),
        )
