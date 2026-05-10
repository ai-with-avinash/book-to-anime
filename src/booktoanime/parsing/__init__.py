"""PDF parsing and asset extraction.

Public API:

- :class:`PDFParser` — orchestrates text + table + image extraction with optional OCR.
- :class:`ParsedDocument`, :class:`ParsedPage`, :class:`ExtractedImage`, :class:`ExtractedTable` —
  data model used as the parsing-stage artifact (`extracted/parsed.json`).
"""

from __future__ import annotations

from .models import (
    ExtractedImage,
    ExtractedTable,
    ParsedDocument,
    ParsedPage,
    PDFMetadata,
)
from .pdf_parser import ParserConfig, PDFParser

__all__ = [
    "ExtractedImage",
    "ExtractedTable",
    "PDFMetadata",
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "ParserConfig",
]
