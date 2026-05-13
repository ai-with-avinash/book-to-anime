"""Pillow-based panel composition for figure shots + title cards.

Phase 3 — figure-first rendering. Renders real extracted figures as
comic-style panels with caption strips; renders title cards. SDXL path
in :mod:`pipeline.image_renderer` handles only :class:`VisualKind.ILLUSTRATION`.

Two layouts based on source-figure aspect:

* ``aspect >= 1.2`` (landscape) → bottom caption strip. Figure occupies
  the top ``width x 800`` region, caption occupies the bottom ``width x
  (height - 800)`` region.
* ``aspect < 1.2`` (portrait / square) → side panel. Figure occupies a
  square block at the left of the canvas, caption occupies the right.

Both helpers raise :class:`booktoanime.errors.RenderError` on font-load
failure or unknown panel-style key, so the image-renderer stage can fail
cleanly without leaking ``KeyError`` / ``IOError`` to the user.
"""

from __future__ import annotations

import re
from importlib.resources import as_file, files
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..errors import RenderError

# Per-panel-style background colour (R, G, B).
_PALETTE: dict[str, tuple[int, int, int]] = {
    "clean-linework": (255, 255, 255),
    "chalkboard-sketch": (26, 58, 46),  # #1a3a2e
    "watercolor-technical": (245, 235, 214),  # #f5ebd6
    "flat-vector-infographic": (232, 232, 232),  # #e8e8e8
}

# Foreground (text) colour per style — contrast-checked against the BG above.
_TEXT_COLOR: dict[str, tuple[int, int, int]] = {
    "clean-linework": (30, 30, 30),
    "chalkboard-sketch": (255, 255, 255),
    "watercolor-technical": (60, 50, 30),
    "flat-vector-infographic": (40, 40, 40),
}

# ``importlib.resources`` anchor for the bundled OFL-1.1 sans-serif TTF
# pair (Inter Regular / Bold). The font files ship inside the wheel under
# ``src/booktoanime/web/static/fonts/`` so installs from the wheel can
# locate them without a filesystem-walk fallback.
_FONTS_PACKAGE = "booktoanime.web.static.fonts"


def _load_font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Return an :class:`ImageFont.FreeTypeFont` for the given weight + size.

    Resolves the TTF via :func:`importlib.resources.files` so the call
    works whether the package is loaded from source or from an installed
    wheel. Surfaces :class:`RenderError` (not :class:`OSError`) on failure
    so the image-renderer stage can fail with a user-actionable message.
    """

    fname = "Inter-Bold.ttf" if weight == "bold" else "Inter-Regular.ttf"
    try:
        resource = files(_FONTS_PACKAGE) / fname
        with as_file(resource) as path:
            return ImageFont.truetype(str(path), size=size)
    except Exception as exc:
        raise RenderError(f"failed to load bundled font {fname}: {exc}") from exc


_WHITESPACE = re.compile(r"\s+")


def _sentence_clean(text: str, max_chars: int = 60) -> str:
    """Normalize whitespace, drop trailing periods, truncate with ellipsis.

    Used for both the caption-strip title and the title-card title/subtitle
    so PDF artefacts (double spaces, line-break hyphenation residue, etc.)
    don't bleed into the rendered panel.
    """

    collapsed = _WHITESPACE.sub(" ", text).strip().rstrip(".").strip()
    if len(collapsed) > max_chars:
        collapsed = collapsed[: max_chars - 1].rstrip() + "…"
    return collapsed


def _wrap_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """Greedy word-wrap. Truncates the final line with an ellipsis on overflow."""

    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
            continue

        # New line needed.
        if current:
            lines.append(current)
        # If we've hit the cap, truncate this final line + bail.
        if len(lines) >= max_lines:
            # Pop the last line we just appended (it doesn't include `word`)
            # and rebuild it with ellipsis so overflow is visible.
            last = lines.pop()
            truncated = last
            while truncated:
                bbox = draw.textbbox((0, 0), truncated + "…", font=font)
                if bbox[2] - bbox[0] <= max_width:
                    break
                truncated = truncated[:-1]
            lines.append((truncated + "…") if truncated else "…")
            return lines
        current = word

    if current:
        lines.append(current)
    return lines


def _letterbox(
    img: Image.Image,
    area: tuple[int, int],
    bg: tuple[int, int, int],
) -> Image.Image:
    """Fit ``img`` inside ``area`` preserving aspect, fill remainder with ``bg``."""

    aw, ah = area
    src_aspect = img.width / max(img.height, 1)
    area_aspect = aw / max(ah, 1)
    if src_aspect > area_aspect:
        new_w = aw
        new_h = max(1, round(aw / src_aspect))
    else:
        new_h = ah
        new_w = max(1, round(ah * src_aspect))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", area, bg)
    out.paste(resized, ((aw - new_w) // 2, (ah - new_h) // 2))
    return out


def compose_figure_panel(
    figure_path: Path,
    caption: str,
    title: str,
    panel_style: str,
    target_size: tuple[int, int] = (1920, 1080),
) -> Image.Image:
    """Compose a figure panel.

    Layout is chosen by source aspect:

    * ``aspect >= 1.2`` (landscape): figure across the top
      (``width x 800`` letterboxed), caption strip across the bottom.
    * ``aspect < 1.2``  (portrait / square): figure on the left as a
      square block (``height x height``), caption on the right.

    Raises:
        RenderError: Font load failure, unknown ``panel_style``, or
            unreadable ``figure_path``.
    """

    if panel_style not in _PALETTE:
        raise RenderError(f"unknown panel_style: {panel_style!r}")

    try:
        figure = Image.open(figure_path).convert("RGB")
    except Exception as exc:
        raise RenderError(f"failed to open figure {figure_path}: {exc}") from exc

    bg = _PALETTE[panel_style]
    fg = _TEXT_COLOR[panel_style]
    title_font = _load_font("bold", 36)
    caption_font = _load_font("regular", 22)

    width, height = target_size
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    src_aspect = figure.width / max(figure.height, 1)
    cleaned_title = _sentence_clean(title, 60)

    if src_aspect >= 1.2:
        # Bottom-strip layout: figure on top, caption beneath.
        fig_area = (width, max(1, height - 280))
        fig_resized = _letterbox(figure, fig_area, bg)
        canvas.paste(fig_resized, (0, 0))

        cap_x_pad = 60
        title_y = fig_area[1] + 30
        draw.text((cap_x_pad, title_y), cleaned_title, fill=fg, font=title_font)
        cap_y = title_y + 50
        cap_lines = _wrap_caption(
            draw,
            caption,
            caption_font,
            max_width=width - 2 * cap_x_pad,
            max_lines=4,
        )
        for i, line in enumerate(cap_lines):
            draw.text((cap_x_pad, cap_y + i * 30), line, fill=fg, font=caption_font)
    else:
        # Side-panel layout: figure left, caption right.
        fig_side = min(width, height)
        fig_area = (fig_side, height)
        fig_resized = _letterbox(figure, fig_area, bg)
        canvas.paste(fig_resized, (0, 0))

        cap_x = fig_side + 40
        cap_w = max(1, width - fig_side - 80)
        title_y = 80
        draw.text((cap_x, title_y), cleaned_title, fill=fg, font=title_font)
        cap_y = title_y + 70
        cap_lines = _wrap_caption(
            draw,
            caption,
            caption_font,
            max_width=cap_w,
            max_lines=10,
        )
        for i, line in enumerate(cap_lines):
            draw.text((cap_x, cap_y + i * 32), line, fill=fg, font=caption_font)

    return canvas


def compose_title_card(
    title: str,
    subtitle: str,
    panel_style: str,
    target_size: tuple[int, int] = (1920, 1080),
) -> Image.Image:
    """Compose a title card with a centred title + subtitle.

    Used for shots whose :class:`VisualKind` is :data:`VisualKind.TITLE_CARD`
    — typically the bookend shot at the start of a multi-shot topic.

    Raises:
        RenderError: Unknown ``panel_style`` or font load failure.
    """

    if panel_style not in _PALETTE:
        raise RenderError(f"unknown panel_style: {panel_style!r}")

    bg = _PALETTE[panel_style]
    fg = _TEXT_COLOR[panel_style]
    title_font = _load_font("bold", 56)
    sub_font = _load_font("regular", 28)

    width, height = target_size
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    cleaned_title = _sentence_clean(title, 80)
    cleaned_sub = _sentence_clean(subtitle, 100)

    title_bbox = draw.textbbox((0, 0), cleaned_title, font=title_font)
    sub_bbox = draw.textbbox((0, 0), cleaned_sub, font=sub_font)

    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    sub_w = sub_bbox[2] - sub_bbox[0]

    title_x = (width - title_w) // 2
    title_y = height // 2 - title_h - 20
    sub_x = (width - sub_w) // 2
    sub_y = height // 2 + 20

    draw.text((title_x, title_y), cleaned_title, fill=fg, font=title_font)
    if cleaned_sub:
        draw.text((sub_x, sub_y), cleaned_sub, fill=fg, font=sub_font)
    return canvas


__all__ = [
    "compose_figure_panel",
    "compose_title_card",
]
