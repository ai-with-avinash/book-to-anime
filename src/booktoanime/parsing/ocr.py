"""OCR fallback using Tesseract (via ``pytesseract``).

We use OCR only when a page has effectively no extractable text layer. The
high-quality profile may swap this implementation for PaddleOCR, but the
abstract entry point is the same: ``OCREngine.recognize(image)`` returning a
plain text string.
"""

from __future__ import annotations

import shutil
from typing import Protocol

from PIL import Image

from ..errors import OCRUnavailableError


class OCREngine(Protocol):
    """Minimal OCR interface; allows us to swap engines per profile."""

    def recognize(self, image: Image.Image, *, language: str = "eng") -> str:
        ...


class TesseractOCR:
    """Tesseract-backed implementation of :class:`OCREngine`.

    The Tesseract binary must be installed on the system. The constructor
    verifies its presence and raises :class:`OCRUnavailableError` if missing,
    so callers fail fast at pipeline-init time rather than mid-page.
    """

    def __init__(self, *, binary: str | None = None) -> None:
        binary_path = binary or shutil.which("tesseract")
        if not binary_path:
            raise OCRUnavailableError()

        # Bind lazily so we don't import pytesseract on machines that never use OCR.
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = binary_path
        self._pytesseract = pytesseract

    def recognize(self, image: Image.Image, *, language: str = "eng") -> str:
        """Run OCR on a Pillow image and return the recognized text."""

        return str(self._pytesseract.image_to_string(image, lang=language))
