"""High-level PDF parser that produces a :class:`ParsedDocument`.

Responsibilities:
    * Open the PDF safely, refusing encrypted or corrupted files with clear errors.
    * Extract the text layer and tables via ``pdfplumber``.
    * Extract embedded raster images via :class:`ImageExtractor` (pypdf-backed).
    * Optionally run OCR on pages with no text layer.
    * Attach surrounding-text context and caption hints to each extracted image
      so the downstream VLM stage has something to ground on.

The parser writes images under ``<job_dir>/extracted/`` and returns a
:class:`ParsedDocument` that the caller serializes to ``parsed.json``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.pdf import PDF as PlumberPDF  # noqa: N811 - upstream class is named PDF
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..errors import (
    CorruptedPDFError,
    EncryptedPDFError,
    ParsingError,
    UnparseableImageOnlyPDFError,
)
from .image_extractor import ImageExtractor
from .models import (
    ExtractedTable,
    ParsedDocument,
    ParsedPage,
    PDFMetadata,
)
from .ocr import OCREngine, TesseractOCR

_logger = logging.getLogger(__name__)

# Heuristic: pages with fewer than this many non-whitespace chars are treated
# as having no text layer for OCR-fallback purposes.
_MIN_TEXT_LAYER_CHARS = 12

_CAPTION_PATTERN = re.compile(
    r"^\s*((?:Figure|Fig\.|Table|Chart|Diagram)\s*[\d\.\-A-Z]+[\.:\)]?)\s*(.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParserConfig:
    """Knobs that control parser behavior.

    Attributes:
        ocr_enabled: Run OCR on pages with no text layer.
        ocr_language: Tesseract language code (e.g. ``"eng"``, ``"jpn"``).
        ocr_resolution: DPI used to rasterize pages for OCR. 300 is a good default.
        min_image_width / min_image_height: Skip embedded images smaller than this.
    """

    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_resolution: int = 300
    min_image_width: int = 32
    min_image_height: int = 32


class PDFParser:
    """Parse a single PDF into a :class:`ParsedDocument`."""

    def __init__(
        self,
        config: ParserConfig | None = None,
        *,
        ocr_engine: OCREngine | None = None,
    ) -> None:
        self._config = config or ParserConfig()
        self._image_extractor = ImageExtractor(
            min_width=self._config.min_image_width,
            min_height=self._config.min_image_height,
        )
        # Lazily construct the OCR engine so machines without Tesseract still
        # work for text-only PDFs even if `ocr_enabled=True`.
        self._ocr_engine_override = ocr_engine
        self._ocr_engine: OCREngine | None = None

    # ------------------------------------------------------------------ public

    def parse(self, pdf_path: Path, *, job_dir: Path) -> ParsedDocument:
        """Parse ``pdf_path`` and write extracted images into ``job_dir``.

        Args:
            pdf_path: Absolute path to the source PDF.
            job_dir: Job working directory. Images are written under
                ``job_dir / "extracted"``.

        Returns:
            A fully-populated :class:`ParsedDocument`.

        Raises:
            EncryptedPDFError: PDF is password-protected.
            CorruptedPDFError: PDF cannot be opened.
            UnparseableImageOnlyPDFError: PDF has no text layer and OCR is disabled.
            ParsingError: Any other parsing failure.
        """

        if not pdf_path.is_file():
            raise CorruptedPDFError(f"PDF not found at {pdf_path}")

        reader = self._open_pypdf(pdf_path)
        extracted_dir = job_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        try:
            with pdfplumber.open(str(pdf_path)) as plumber:
                metadata = self._build_metadata(plumber, reader)
                pages = [
                    self._parse_page(plumber, reader, idx, extracted_dir)
                    for idx in range(len(plumber.pages))
                ]
        except (EncryptedPDFError, CorruptedPDFError, UnparseableImageOnlyPDFError):
            raise
        except Exception as exc:
            raise ParsingError(f"pdfplumber failed: {exc}") from exc

        if all(not p.text.strip() for p in pages):
            # We tried text layer + (optionally) OCR and got nothing.
            raise UnparseableImageOnlyPDFError()

        return ParsedDocument(pages=pages, metadata=metadata)

    # ------------------------------------------------------------------ helpers

    def _open_pypdf(self, pdf_path: Path) -> PdfReader:
        try:
            reader = PdfReader(str(pdf_path))
        except PdfReadError as exc:
            raise CorruptedPDFError(str(exc)) from exc
        except OSError as exc:
            raise CorruptedPDFError(str(exc)) from exc

        if reader.is_encrypted:
            # Try empty password (some PDFs are technically "encrypted" with no password).
            try:
                if reader.decrypt("") == 0:
                    raise EncryptedPDFError()
            except NotImplementedError as exc:  # unsupported encryption algorithm
                raise EncryptedPDFError(str(exc)) from exc

        return reader

    def _build_metadata(self, plumber: PlumberPDF, reader: PdfReader) -> PDFMetadata:
        plumber_meta: Mapping[str, Any] = plumber.metadata or {}
        pypdf_meta: Mapping[str, Any] = reader.metadata or {}

        def _pick(*keys: str) -> str | None:
            for key in keys:
                value = plumber_meta.get(key) or pypdf_meta.get(key)
                if value:
                    return str(value)
            return None

        return PDFMetadata(
            title=_pick("Title", "/Title"),
            author=_pick("Author", "/Author"),
            subject=_pick("Subject", "/Subject"),
            creator=_pick("Creator", "/Creator"),
            producer=_pick("Producer", "/Producer"),
            page_count=len(plumber.pages),
        )

    def _parse_page(
        self,
        plumber: PlumberPDF,
        reader: PdfReader,
        index: int,
        extracted_dir: Path,
    ) -> ParsedPage:
        plumber_page = plumber.pages[index]

        text = (plumber_page.extract_text() or "").strip()
        ocr_used = False
        if len(text) < _MIN_TEXT_LAYER_CHARS and self._config.ocr_enabled:
            ocr_text = self._ocr_page(plumber_page)
            if ocr_text:
                text = ocr_text
                ocr_used = True

        tables = self._extract_tables(plumber_page, page_index=index)
        raw_images = self._image_extractor.extract_page(reader, index, extracted_dir)
        caption_hint = self._caption_hint(text)
        page_context = text[:1500]
        images = [
            image.model_copy(
                update={
                    "caption_hint": caption_hint,
                    "surrounding_text": page_context,
                }
            )
            for image in raw_images
        ]

        return ParsedPage(
            index=index,
            text=text,
            tables=tables,
            images=images,
            ocr_used=ocr_used,
        )

    def _ocr_page(self, plumber_page: object) -> str:
        engine = self._get_ocr_engine()
        if engine is None:
            return ""

        try:
            # ``page.to_image`` is a pdfplumber method; type-check at runtime.
            page_image = plumber_page.to_image(resolution=self._config.ocr_resolution)  # type: ignore[attr-defined]
            pil = page_image.original
        except Exception as exc:
            _logger.warning("OCR rasterization failed: %s", exc)
            return ""

        try:
            return engine.recognize(pil, language=self._config.ocr_language).strip()
        except Exception as exc:
            _logger.warning("OCR recognition failed: %s", exc)
            return ""

    def _get_ocr_engine(self) -> OCREngine | None:
        if self._ocr_engine is not None:
            return self._ocr_engine
        if self._ocr_engine_override is not None:
            self._ocr_engine = self._ocr_engine_override
            return self._ocr_engine
        try:
            self._ocr_engine = TesseractOCR()
        except Exception as exc:
            _logger.info("OCR engine unavailable: %s", exc)
            self._ocr_engine = None
        return self._ocr_engine

    def _extract_tables(self, plumber_page: object, *, page_index: int) -> list[ExtractedTable]:
        try:
            raw_tables = plumber_page.extract_tables() or []  # type: ignore[attr-defined]
        except Exception as exc:
            _logger.warning("table extraction failed on page %d: %s", page_index, exc)
            return []

        tables: list[ExtractedTable] = []
        for tbl_idx, rows in enumerate(raw_tables):
            cleaned: list[list[str]] = []
            for row in rows:
                cleaned.append([(cell or "").strip() for cell in row])
            tables.append(
                ExtractedTable(
                    id=f"t_{page_index}_{tbl_idx}",
                    rows=cleaned,
                )
            )
        return tables

    @staticmethod
    def _caption_hint(text: str) -> str | None:
        for line in text.splitlines():
            match = _CAPTION_PATTERN.match(line)
            if match:
                head, tail = match.group(1), match.group(2)
                hint = f"{head} {tail}".strip()
                return hint or head
        return None
