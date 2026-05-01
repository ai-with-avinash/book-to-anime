"""Narrator-persona seed selection.

The structuring stage decides who the narrator is — a stable seed and a short
descriptor — and the images stage materialises that into an actual reference
image via :meth:`VisualProvider.prepare`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .artifacts import NarratorPersona


@dataclass(frozen=True)
class PersonaSeederConfig:
    anime_style: str
    narration_language: str
    voice_id: str


def derive_persona(config: PersonaSeederConfig) -> NarratorPersona:
    """Derive a deterministic persona from style + voice + language.

    Same inputs → same seed across machines and across resumes.
    """

    digest = hashlib.sha256(
        "|".join((config.anime_style, config.narration_language, config.voice_id)).encode("utf-8")
    ).digest()
    # Truncate to 31 bits — well within torch.Generator.manual_seed range.
    seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    descriptor = (
        f"{config.anime_style} narrator persona, voice {config.voice_id} "
        f"in {config.narration_language}"
    )
    return NarratorPersona(seed=seed, style_descriptor=descriptor, reference_image=None)
