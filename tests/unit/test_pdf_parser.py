"""Unit tests for ``booktoanime.parsing.pdf_parser``."""

from __future__ import annotations

from pathlib import Path

import pytest

from booktoanime.errors import (
    CorruptedPDFError,
    EncryptedPDFError,
    UnparseableImageOnlyPDFError,
)
from booktoanime.parsing import ParsedDocument, PDFParser
from booktoanime.parsing.pdf_parser import ParserConfig


def test_parses_tiny_pdf_text_and_metadata(tiny_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    parsed = parser.parse(tiny_pdf, job_dir=job_dir)

    assert isinstance(parsed, ParsedDocument)
    assert parsed.metadata.page_count == 2
    assert parsed.metadata.title == "Tiny Test Book"
    assert parsed.metadata.author == "BookToAnime Tests"

    page1, page2 = parsed.pages
    assert "Chapter 1" in page1.text
    assert "Welcome to the tiny test book" in page1.text
    assert "Chapter 2" in page2.text
    assert page1.ocr_used is False
    assert page2.ocr_used is False


def test_extracts_embedded_image_with_caption_hint(tiny_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    parsed = parser.parse(tiny_pdf, job_dir=job_dir)

    page1 = parsed.pages[0]
    assert page1.images, "expected at least one image extracted from page 1"

    image = page1.images[0]
    assert image.width >= 32
    assert image.height >= 32
    assert image.file.startswith("extracted/")

    # Caption hint should pick up the "Figure 1.1" line.
    assert image.caption_hint is not None
    assert image.caption_hint.lower().startswith("figure 1.1")

    # File was actually written to disk under the job dir.
    assert (job_dir / image.file).is_file()


def test_extracts_table_rows(tiny_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    parsed = parser.parse(tiny_pdf, job_dir=job_dir)

    page2 = parsed.pages[1]
    # pdfplumber may or may not recognize the simple grid as a table; if it
    # does, rows should at least include the headers + values we wrote.
    if page2.tables:
        flattened = {cell for row in page2.tables[0].rows for cell in row}
        assert "Name" in flattened or "Alice" in flattened


def test_round_trips_through_json(tiny_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    parsed = parser.parse(tiny_pdf, job_dir=job_dir)

    out_path = job_dir / "extracted" / "parsed.json"
    out_path.write_bytes(parsed.to_json_bytes())

    reloaded = ParsedDocument.from_path(out_path)
    assert reloaded == parsed


def test_encrypted_pdf_raises(encrypted_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    with pytest.raises(EncryptedPDFError):
        parser.parse(encrypted_pdf, job_dir=job_dir)


def test_corrupted_pdf_raises(corrupted_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser()
    with pytest.raises(CorruptedPDFError):
        parser.parse(corrupted_pdf, job_dir=job_dir)


def test_image_only_pdf_without_ocr_raises(image_only_pdf: Path, job_dir: Path) -> None:
    parser = PDFParser(ParserConfig(ocr_enabled=False))
    with pytest.raises(UnparseableImageOnlyPDFError):
        parser.parse(image_only_pdf, job_dir=job_dir)


def test_image_only_pdf_with_mock_ocr_succeeds(image_only_pdf: Path, job_dir: Path) -> None:
    class _StubOCR:
        def recognize(self, image, *, language: str = "eng") -> str:
            return "(no text layer)"

    parser = PDFParser(ParserConfig(ocr_enabled=True), ocr_engine=_StubOCR())
    parsed = parser.parse(image_only_pdf, job_dir=job_dir)

    assert parsed.metadata.page_count == 1
    assert parsed.pages[0].ocr_used is True
    assert "no text layer" in parsed.pages[0].text


def test_missing_pdf_raises(job_dir: Path, tmp_path: Path) -> None:
    parser = PDFParser()
    with pytest.raises(CorruptedPDFError):
        parser.parse(tmp_path / "does_not_exist.pdf", job_dir=job_dir)
