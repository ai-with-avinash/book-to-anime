"""Embedded raster-image extraction from a PDF using ``pypdf``.

We deliberately avoid PyMuPDF (AGPL). ``pypdf`` covers the common case of
images embedded as XObjects with standard filters (DCTDecode/FlateDecode/
LZW/CCITTFax). Anything pypdf cannot decode is skipped with a warning rather
than aborting the whole job; downstream stages still see whatever images
were successfully extracted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import ExtractedImage

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExtractionResult:
    page_index: int
    images: list[ExtractedImage]


class ImageExtractor:
    """Extracts embedded raster images and writes them under a target directory."""

    def __init__(self, *, min_width: int = 32, min_height: int = 32) -> None:
        # Filter out tiny decorative images (icons, separators) that add no narrative value.
        self._min_width = min_width
        self._min_height = min_height

    def extract_page(
        self,
        reader: PdfReader,
        page_index: int,
        out_dir: Path,
        *,
        rel_dir: str = "extracted",
    ) -> list[ExtractedImage]:
        """Extract images from a single page.

        Args:
            reader: An open ``pypdf.PdfReader`` for the source PDF.
            page_index: Zero-based page index to extract from.
            out_dir: Absolute directory the images should be written to.
                Created if missing.
            rel_dir: Path prefix stored in ``ExtractedImage.file`` so the JSON
                artifact remains portable between machines (paths are stored
                relative to the job directory).

        Returns:
            A list of :class:`ExtractedImage` records — one per image successfully
            written to disk. Failures are logged and skipped.
        """

        out_dir.mkdir(parents=True, exist_ok=True)
        page = reader.pages[page_index]

        results: list[ExtractedImage] = []
        # ``pypdf`` exposes images on each page; iteration is ordered by appearance.
        for image_index, image in enumerate(page.images):
            try:
                width, height = self._dimensions(image)
            except Exception as exc:
                _logger.warning(
                    "skipping image on page %d (#%d): could not read size (%s)",
                    page_index,
                    image_index,
                    exc,
                )
                continue

            if width < self._min_width or height < self._min_height:
                continue

            ext = self._suffix_for(image.name)
            image_id = f"img_{page_index}_{image_index}"
            out_name = f"{image_id}{ext}"
            out_path = out_dir / out_name

            try:
                out_path.write_bytes(image.data)
            except OSError as exc:
                _logger.warning(
                    "skipping image on page %d (#%d): write failed (%s)",
                    page_index,
                    image_index,
                    exc,
                )
                continue

            rel_file = f"{rel_dir.rstrip('/')}/{out_name}" if rel_dir else out_name
            results.append(
                ExtractedImage(
                    id=image_id,
                    file=rel_file,
                    width=width,
                    height=height,
                )
            )

        return results

    @staticmethod
    def _dimensions(image: Any) -> tuple[int, int]:
        """Return the ``(width, height)`` of a pypdf image record.

        Different pypdf versions expose either ``image.image`` (a Pillow image)
        or only the raw bytes, so we try both.
        """

        pil = getattr(image, "image", None)
        if pil is not None and hasattr(pil, "size"):
            width, height = pil.size
            return int(width), int(height)

        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(image.data)) as opened:
            return int(opened.width), int(opened.height)

    @staticmethod
    def _suffix_for(name: str) -> str:
        """Pick a filesystem suffix from the image name pypdf reports."""

        # pypdf names look like "image1.png", "Im0.jpg", etc. Fall back to .png if unknown.
        lower = name.lower()
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"):
            if lower.endswith(ext):
                return ext
        return ".png"
