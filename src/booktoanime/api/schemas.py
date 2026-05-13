"""Request / response models for the JSON API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..pipeline.manifest import (
    AspectRatio,
    Depth,
    JobConfig,
    LengthPreset,
    NarrationConfig,
    Profile,
    ProvidersConfig,
    StageStatus,
)


class ChapterSummary(BaseModel):
    """One per-topic mp4 + srt that a completed job exposes."""

    model_config = ConfigDict(extra="forbid")

    order: int
    topic_id: str
    duration_seconds: float
    mp4_url: str
    srt_url: str


class CreateJobRequest(BaseModel):
    """Form fields posted by the upload UI (multipart, sans the PDF file)."""

    model_config = ConfigDict(extra="forbid")

    panel_style: str = "clean-linework"
    voice_id: str
    language: str = "en-US"
    speed: float = 1.0
    depth: Depth = "undergraduate"
    length_preset: LengthPreset = "standard"
    minutes_per_topic: float | None = Field(default=None, gt=0.0)
    aspect_ratio: AspectRatio = "16:9"
    profile: Profile = "default"

    def to_job_config(self, providers: ProvidersConfig) -> JobConfig:
        return JobConfig(
            panel_style=self.panel_style,
            narration=NarrationConfig(
                voice_id=self.voice_id,
                language=self.language,
                speed=self.speed,
            ),
            depth=self.depth,
            length_preset=self.length_preset,
            minutes_per_topic=self.minutes_per_topic,
            aspect_ratio=self.aspect_ratio,
            profile=self.profile,
            providers=providers,
        )


class JobSummary(BaseModel):
    """Compact view returned by ``GET /jobs`` and ``GET /jobs/{id}``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    current_stage: str | None
    created_at: datetime
    updated_at: datetime
    source_pdf: str
    error_message: str | None = None
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    chapters: list[ChapterSummary] = Field(default_factory=list)


class JobCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    events_url: str
    job_url: str


class JobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[JobSummary]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    version: str
    providers: dict[str, str]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    extra: dict[str, Any] = Field(default_factory=dict)
