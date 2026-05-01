"""``manifest.json`` — the root descriptor for a job directory."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .stages import STAGE_ORDER, Stage


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    error_message: str | None = None


class NarrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_id: str
    language: str
    speed: float = 1.0


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    audio: str
    visual: str


Depth = Literal["eli5", "undergraduate", "expert"]
LengthPreset = Literal["short", "standard", "in_depth"]
AspectRatio = Literal["16:9", "9:16", "1:1"]
Profile = Literal["default", "high_quality", "low_vram"]


class JobConfig(BaseModel):
    """Per-job configuration captured from the upload form."""

    model_config = ConfigDict(extra="forbid")

    anime_style: str = "shounen-bright"
    narration: NarrationConfig
    depth: Depth = "undergraduate"
    length_preset: LengthPreset = "standard"
    minutes_per_topic: float | None = Field(default=None, gt=0.0)
    aspect_ratio: AspectRatio = "16:9"
    profile: Profile = "default"
    providers: ProvidersConfig


class JobManifest(BaseModel):
    """Top-level descriptor stored as ``<job_dir>/manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    job_id: str
    created_at: str
    source_pdf: Annotated[str, Field(description="Path relative to the job directory.")] = (
        "source.pdf"
    )
    config: JobConfig
    stages: dict[str, StageState] = Field(default_factory=dict)

    # ------------------------------------------------------------- factories

    @classmethod
    def for_new_job(
        cls,
        *,
        job_id: str,
        config: JobConfig,
        source_pdf: str = "source.pdf",
    ) -> JobManifest:
        return cls(
            job_id=job_id,
            created_at=_now_iso(),
            source_pdf=source_pdf,
            config=config,
            stages={stage.value: StageState() for stage in STAGE_ORDER},
        )

    @classmethod
    def from_path(cls, path: Path) -> JobManifest:
        return cls.model_validate_json(path.read_bytes())

    # ------------------------------------------------------------- mutations

    def mark_started(self, stage: Stage) -> None:
        self.stages[stage.value] = StageState(
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
        )

    def mark_completed(self, stage: Stage) -> None:
        prior = self.stages.get(stage.value, StageState())
        self.stages[stage.value] = prior.model_copy(
            update={
                "status": StageStatus.COMPLETED,
                "completed_at": _now_iso(),
                "failed_at": None,
                "error_message": None,
            }
        )

    def mark_failed(self, stage: Stage, error_message: str) -> None:
        prior = self.stages.get(stage.value, StageState())
        self.stages[stage.value] = prior.model_copy(
            update={
                "status": StageStatus.FAILED,
                "failed_at": _now_iso(),
                "error_message": error_message,
            }
        )

    # ------------------------------------------------------------- queries

    def first_unfinished_stage(self) -> Stage | None:
        for stage in STAGE_ORDER:
            state = self.stages.get(stage.value, StageState())
            if state.status != StageStatus.COMPLETED:
                return stage
        return None

    def stage_status(self, stage: Stage) -> StageStatus:
        return self.stages.get(stage.value, StageState()).status

    # ------------------------------------------------------------- IO

    def save(self, manifest_path: Path) -> None:
        """Atomically write the manifest to ``manifest_path``."""

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_bytes(self.model_dump_json(indent=2).encode("utf-8"))
        tmp_path.replace(manifest_path)

    def to_public_dict(self) -> Mapping[str, object]:
        """A lightweight representation suitable for the SSE/UI layer."""

        return self.model_dump(mode="json")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
