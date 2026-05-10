"""Unit tests for parsing pydantic models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from booktoanime.parsing.models import (
    ExtractedImage,
    ExtractedTable,
    ParsedDocument,
    ParsedPage,
    PDFMetadata,
)


def _document() -> ParsedDocument:
    return ParsedDocument(
        pages=[
            ParsedPage(
                index=0,
                text="hello",
                tables=[ExtractedTable(id="t_0_0", rows=[["h"], ["v"]])],
                images=[
                    ExtractedImage(
                        id="img_0_0",
                        file="extracted/img_0_0.png",
                        width=100,
                        height=80,
                        caption_hint="Figure 1.1",
                        surrounding_text="hello",
                    )
                ],
                ocr_used=False,
            )
        ],
        metadata=PDFMetadata(page_count=1, title="t"),
    )


def test_round_trip(tmp_path: Path) -> None:
    doc = _document()
    out = tmp_path / "parsed.json"
    out.write_bytes(doc.to_json_bytes())
    reloaded = ParsedDocument.from_path(out)
    assert reloaded == doc


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ParsedPage.model_validate({"index": 0, "unexpected_field": True})


def test_image_dimensions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExtractedImage(id="x", file="f.png", width=0, height=10)


def test_metadata_page_count_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        PDFMetadata(page_count=-1)
