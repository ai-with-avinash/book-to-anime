"""Per-image VLM explanation grounded in surrounding text.

When the language provider lacks vision (or no fallback was configured) we
silently degrade by synthesizing a caption-only explanation from the surrounding
text — the user still gets *something* to narrate, just not a vision-grounded
description.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..errors import CapabilityNotSupportedError
from ..parsing.models import ExtractedImage
from ..providers import ImageExplanation, LanguageProvider, VisionInput

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExplainedImage:
    image_id: str
    explanation: ImageExplanation


class ImageExplainer:
    """Run vision explanation on the embedded images we want narrated.

    The orchestrator decides which images to send (typically only those
    referenced by a topic). The explainer just routes calls through the
    provider, with a fallback when vision is unsupported.
    """

    def __init__(
        self,
        primary: LanguageProvider,
        *,
        vision_fallback: LanguageProvider | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = vision_fallback

    async def explain_many(
        self,
        images: Iterable[ExtractedImage],
        *,
        job_dir: Path,
    ) -> list[ExplainedImage]:
        results: list[ExplainedImage] = []
        for image in images:
            absolute_path = (job_dir / image.file).resolve()
            vision_input = VisionInput(
                image_path=absolute_path,
                surrounding_text=image.surrounding_text,
                caption_hint=image.caption_hint,
            )
            explanation = await self._explain(vision_input, image=image)
            results.append(ExplainedImage(image_id=image.id, explanation=explanation))
        return results

    async def _explain(
        self,
        vision_input: VisionInput,
        *,
        image: ExtractedImage,
    ) -> ImageExplanation:
        for provider in self._candidates():
            try:
                return await provider.explain_image(vision_input)
            except CapabilityNotSupportedError:
                continue
        # No vision-capable provider available; degrade gracefully.
        return _synthesize_from_caption(image)

    def _candidates(self) -> list[LanguageProvider]:
        candidates = [self._primary]
        if self._fallback is not None and self._fallback is not self._primary:
            candidates.append(self._fallback)
        return candidates


def _synthesize_from_caption(image: ExtractedImage) -> ImageExplanation:
    caption = image.caption_hint or "an embedded figure"
    surrounding = image.surrounding_text.strip()
    snippet = surrounding[:280] if surrounding else "no surrounding context available"
    summary = f"{caption.capitalize()} appears here: {snippet[:120]}"
    detail = (
        f"{caption}\n\nNo vision model was available to inspect this image. "
        "The narrator references the surrounding text instead:\n"
        f"{snippet}"
    )
    _logger.info(
        "synthesized text-only explanation for image %s (no vision provider)",
        image.id,
    )
    return ImageExplanation(summary=summary, detail=detail)
