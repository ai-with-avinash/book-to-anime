"""Data model for parsed PDF artifacts.

These pydantic models define the on-disk JSON contract written by the parsing
stage to ``<job_dir>/extracted/parsed.json`` and consumed by the structuring
stage. The model is intentionally framework-agnostic so it can be re-loaded by
other tools (e.g. a debugging script) without importing the rest of the
pipeline.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _validate_job_relative_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        raise ValueError(f"path must be relative to the job directory: {value!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"path must not traverse parent directories: {value!r}")
    if "\x00" in value:
        raise ValueError("path must not contain NUL bytes")
    return value


JobRelPath = Annotated[str, AfterValidator(_validate_job_relative_path)]


class PDFMetadata(BaseModel):
    """Top-level metadata pulled from the PDF info dictionary."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    page_count: int = Field(ge=0)


class ExtractedTable(BaseModel):
    """A single table extracted from a page."""

    model_config = ConfigDict(extra="forbid")

    id: str
    rows: list[list[str]] = Field(
        default_factory=list,
        description="Row-major table data; each row is a list of cell strings.",
    )
    caption: str | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="(x0, top, x1, bottom) on the source page in PDF user-space units.",
    )


class ExtractedImage(BaseModel):
    """A single raster image extracted from a page.

    ``file`` is a path **relative** to the job directory so artifacts remain
    portable between machines. The parser writes the actual bytes to
    ``<job_dir>/extracted/<file>``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    file: JobRelPath
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bbox: tuple[float, float, float, float] | None = None
    caption_hint: str | None = Field(
        default=None,
        description="Likely caption found near the image (e.g. 'Figure 1.2').",
    )
    surrounding_text: str = Field(
        default="",
        description="Text near the image used to ground later VLM explanations.",
    )


class ParsedPage(BaseModel):
    """One page of the parsed document."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Zero-based page index.")
    text: str = ""
    tables: list[ExtractedTable] = Field(default_factory=list)
    images: list[ExtractedImage] = Field(default_factory=list)
    ocr_used: bool = False


class ParsedDocument(BaseModel):
    """Top-level parsing-stage artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    source_pdf: JobRelPath = Field(
        default="source.pdf",
        description="Path relative to the job directory, e.g. 'source.pdf'.",
    )
    pages: list[ParsedPage]
    metadata: PDFMetadata

    def to_json_bytes(self) -> bytes:
        """Return a deterministic UTF-8 JSON encoding suitable for atomic writes."""

        return self.model_dump_json(indent=2).encode("utf-8")

    @classmethod
    def from_path(cls, path: Path) -> ParsedDocument:
        """Load and validate a previously written ``parsed.json`` file."""

        return cls.model_validate_json(path.read_bytes())
