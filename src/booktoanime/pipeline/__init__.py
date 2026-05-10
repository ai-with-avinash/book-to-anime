"""End-to-end PDF -> structured -> storyboard -> images -> audio orchestration.

Stage order is intentionally rigid (see :class:`Stage`). Each stage writes a
versioned artifact to disk inside the job directory; the orchestrator never
holds large intermediate state in memory and so resumes any job from the
last completed stage by inspecting the artifact files alone.
"""

from __future__ import annotations

from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .manifest import (
    JobConfig,
    JobManifest,
    NarrationConfig,
    ProvidersConfig,
    StageStatus,
)
from .orchestrator import PipelineDependencies, PipelineOrchestrator
from .stages import Stage

__all__ = [
    "JobConfig",
    "JobManifest",
    "NarrationConfig",
    "PipelineDependencies",
    "PipelineOrchestrator",
    "ProgressEvent",
    "ProgressEventBus",
    "ProgressKind",
    "ProvidersConfig",
    "Stage",
    "StageStatus",
]
