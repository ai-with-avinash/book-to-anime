"""Path-traversal guards on every JobRelPath field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from booktoanime.parsing.models import ExtractedImage, ParsedDocument, ParsedPage, PDFMetadata
from booktoanime.pipeline.artifacts import (
    NarratorPersona,
    ShotAudioRecord,
    ShotImageRecord,
)


@pytest.mark.parametrize(
    "bad_value",
    [
        "/etc/passwd",
        "../../../etc/passwd",
        "extracted/../../etc/passwd",
        "with\x00nul",
    ],
)
def test_extracted_image_rejects_unsafe_paths(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedImage(id="x", file=bad_value, width=10, height=10)


def test_extracted_image_accepts_safe_relative() -> None:
    img = ExtractedImage(id="x", file="extracted/img_0_0.png", width=10, height=10)
    assert img.file == "extracted/img_0_0.png"


@pytest.mark.parametrize("bad_value", ["/abs/path", "../escape"])
def test_narrator_persona_rejects_unsafe_paths(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        NarratorPersona(seed=1, style_descriptor="x", reference_image=bad_value)


@pytest.mark.parametrize("bad_value", ["/abs", "../escape", "ok/../../escape"])
def test_shot_image_record_rejects_unsafe_paths(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        ShotImageRecord(shot_id="s", file=bad_value, seed=1, width=10, height=10)


@pytest.mark.parametrize("bad_value", ["/abs", "../escape"])
def test_shot_audio_record_rejects_unsafe_paths(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        ShotAudioRecord(shot_id="s", file=bad_value, duration_seconds=1.0, sample_rate=24000)


def test_parsed_document_rejects_unsafe_source_pdf() -> None:
    with pytest.raises(ValidationError):
        ParsedDocument(
            source_pdf="/etc/passwd",
            pages=[ParsedPage(index=0, text="x")],
            metadata=PDFMetadata(page_count=1),
        )
