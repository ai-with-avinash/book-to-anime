"""Pydantic models for inter-stage artifact files (``structured.json``,
``storyboard.json``, per-stage index files).

Path safety:
    Every field documented as "path relative to the job directory" is wrapped
    in :data:`JobRelPath` and validated to reject absolute paths and parent-
    directory escapes (``..``). This is the project's defense-in-depth against
    a tampered artifact JSON pointing the pipeline (or downstream consumers
    that read these files) at arbitrary filesystem locations.
"""

from __future__ import annotations

from collections.abc import Sequence
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

# --------------------------------------------------------------- structured.json


class TopicSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    page_range: tuple[int, int]
    summary: str
    key_points: list[str] = Field(default_factory=list)
    image_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    estimated_narration_seconds: float = Field(ge=0.0)


class NarratorPersona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    style_descriptor: str
    reference_image: JobRelPath | None = Field(
        default=None,
        description="Path relative to the job directory; populated by the images stage.",
    )


class StructuredDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    topics: list[TopicSection]
    narrator_persona: NarratorPersona

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> StructuredDocument:
        return cls.model_validate_json(path.read_bytes())


# --------------------------------------------------------------- storyboard.json


class KenBurns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_: tuple[float, float, float] = Field(alias="from")
    to: tuple[float, float, float]


class Shot(BaseModel):
    """One shot in the storyboard.

    A shot is the smallest unit of work the images and audio stages care
    about. It has its own narration text, image prompt, and timing data.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    topic_id: str
    order: int = Field(ge=1)
    narration_text: str
    duration_seconds_target: float = Field(ge=0.5)
    image_prompt: str
    negative_prompt: str | None = None
    use_persona_reference: bool = True
    ip_adapter_strength: float = Field(default=0.65, ge=0.0, le=1.0)
    seed: int
    ken_burns: KenBurns
    crossfade_in_ms: int = Field(default=400, ge=0)
    crossfade_out_ms: int = Field(default=400, ge=0)
    explains_image_id: str | None = None


class Storyboard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    shots: list[Shot]
    total_duration_seconds_target: float = Field(ge=0.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2, by_alias=True).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> Storyboard:
        return cls.model_validate_json(path.read_bytes())


# --------------------------------------------------------------- per-stage index files


class ShotImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: str
    file: JobRelPath
    seed: int
    width: int
    height: int


class ImagesIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    items: list[ShotImageRecord] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> ImagesIndex:
        return cls.model_validate_json(path.read_bytes())


class ShotAudioRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: str
    file: JobRelPath
    duration_seconds: float
    sample_rate: int


class AudioIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    items: list[ShotAudioRecord] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> AudioIndex:
        return cls.model_validate_json(path.read_bytes())


class MouthShotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: str
    file: JobRelPath
    duration_seconds: float
    fps: float


class MouthIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    items: list[MouthShotRecord] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> MouthIndex:
        return cls.model_validate_json(path.read_bytes())


class ChapterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str
    order: int = Field(ge=1)
    file: JobRelPath
    srt_file: JobRelPath
    duration_seconds: float = Field(ge=0.0)


class ChaptersIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    items: list[ChapterRecord] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))

    @classmethod
    def from_path(cls, path: Path) -> ChaptersIndex:
        return cls.model_validate_json(path.read_bytes())


# --------------------------------------------------------------- helpers exported for stage code


def shot_ids(shots: Sequence[Shot]) -> list[str]:
    return [shot.id for shot in shots]
