"""Panel-style prompt fragments shared by the visual provider + style seeder.

A panel style is a short, evocative prompt fragment that gets appended to every
SDXL prompt (per shot) and to the style-anchor reference image (per job). Both
the storyboard prompt builder and :mod:`pipeline.style_seeder` import the same
mapping so the per-job anchor and per-shot prompts share an aesthetic.

If you add a new style, ALSO add it to:

* :mod:`web.templates.index.html` (the upload-form dropdown)
* the panel-style section in ``README.md``
* the ``defaults.panel_style`` comment in ``config.example.yaml``
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

STYLE_FRAGMENTS: Final[Mapping[str, str]] = {
    "clean-linework": (
        "clean line art illustration, minimal color palette, technical diagram "
        "aesthetic, neutral background, sharp edges, no shading"
    ),
    "chalkboard-sketch": (
        "chalkboard illustration, white chalk on dark green background, "
        "hand-drawn diagram style, classroom aesthetic"
    ),
    "watercolor-technical": (
        "soft watercolor technical illustration, muted earth palette, "
        "hand-painted scientific drawing, light pencil outline"
    ),
    "flat-vector-infographic": (
        "flat vector infographic, bold geometric shapes, limited four-color "
        "palette, modern educational design"
    ),
}


# Base prompt used only by ``StyleSeeder`` to produce the per-job IP-Adapter
# anchor. It deliberately mentions no subject so the resulting image is a
# pure style swatch the per-shot renders can hew to.
STYLE_ANCHOR_BASE: Final[str] = (
    "abstract style reference, no subject, neutral composition, "
    "single textural sample"
)


__all__ = ["STYLE_ANCHOR_BASE", "STYLE_FRAGMENTS"]
