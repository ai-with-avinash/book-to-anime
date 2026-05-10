"""Unit tests for ``booktoanime.parsing.image_extractor``."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from booktoanime.parsing.image_extractor import ImageExtractor


def test_extract_page_writes_files_and_records(tiny_pdf: Path, job_dir: Path) -> None:
    reader = PdfReader(str(tiny_pdf))
    out_dir = job_dir / "extracted"

    extractor = ImageExtractor()
    images = extractor.extract_page(reader, 0, out_dir, rel_dir="extracted")

    assert len(images) >= 1
    record = images[0]
    assert record.id.startswith("img_0_")
    assert record.file.startswith("extracted/")
    assert (job_dir / record.file).is_file()
    assert record.width > 0 and record.height > 0


def test_filters_below_minimum_dimensions(tiny_pdf: Path, job_dir: Path) -> None:
    extractor = ImageExtractor(min_width=10_000, min_height=10_000)
    reader = PdfReader(str(tiny_pdf))
    images = extractor.extract_page(reader, 0, job_dir / "extracted")
    assert images == []
