"""Dependency wiring for the FastAPI app.

The provider stack is built once per app via :class:`ProviderFactory`, then
reused across requests. Long-running background jobs are tracked in
:class:`JobRunner` so SSE subscribers can attach to a live bus and resume
endpoints know whether a job is already running.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..parsing import PDFParser
from ..pipeline.events import ProgressEventBus
from ..pipeline.orchestrator import PipelineDependencies, PipelineOrchestrator
from ..pipeline.video_assembler import FFmpegRunner
from ..providers import (
    AudioProvider,
    LanguageProvider,
    VisualProvider,
    build_audio_provider,
    build_language_provider,
    build_visual_provider,
)
from ..providers.base import LipSyncProvider
from ..state import JobRepository, JobStatus

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderFactory:
    """Builds the three provider instances on demand.

    The defaults read from a ``config.yaml``-style mapping; tests substitute
    a closure that returns pre-built fakes.
    """

    language_factory: Callable[[], LanguageProvider]
    audio_factory: Callable[[], AudioProvider]
    visual_factory: Callable[[], VisualProvider]
    vision_fallback_factory: Callable[[], LanguageProvider | None] = lambda: None
    # Optional lip-sync provider builder. Returns None when lip-sync isn't
    # wired; the orchestrator skips MOUTH_ANIMATION when the JobConfig
    # disables it, and raises a clear error if it's enabled but no provider
    # was supplied.
    lipsync_factory: Callable[[], LipSyncProvider | None] = lambda: None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ProviderFactory:
        """Build factory closures from a parsed ``config.yaml`` dict."""

        language_cfg = config.get("language") or {}
        audio_cfg = config.get("audio") or {}
        visual_cfg = config.get("visual") or {}
        vision_fallback_name = (
            config.get("language", {}).get("vision_fallback")
            if isinstance(config.get("language"), Mapping)
            else None
        )

        def _language() -> LanguageProvider:
            return build_language_provider(language_cfg)

        def _audio() -> AudioProvider:
            return build_audio_provider(audio_cfg)

        def _visual() -> VisualProvider:
            return build_visual_provider(visual_cfg)

        def _vision_fallback() -> LanguageProvider | None:
            if not vision_fallback_name:
                return None
            sub = language_cfg.get(vision_fallback_name) if isinstance(language_cfg, Mapping) else None
            if not sub:
                return None
            return build_language_provider({"active": vision_fallback_name, vision_fallback_name: sub})

        return cls(
            language_factory=_language,
            audio_factory=_audio,
            visual_factory=_visual,
            vision_fallback_factory=_vision_fallback,
        )


@dataclass
class _RunningJob:
    bus: ProgressEventBus
    task: asyncio.Task[Any]


@dataclass
class JobRunner:
    """Tracks live pipeline runs and exposes per-job event buses for SSE."""

    data_dir: Path
    repo: JobRepository
    provider_factory: ProviderFactory
    parser_factory: Callable[[], PDFParser] = field(default=PDFParser)
    ffmpeg_runner: FFmpegRunner | None = None
    ffmpeg_binary: str | None = None
    _live: dict[str, _RunningJob] = field(default_factory=dict)

    # ----------------------------------------------------------------- run

    def start(self, *, job_id: str) -> _RunningJob:
        """Launch (or rejoin) the orchestrator task for ``job_id``.

        The orchestrator reads ``manifest.json`` from disk itself, so the
        caller doesn't need to hand a manifest in — it only needs to ensure
        the on-disk state for ``job_id`` is valid before calling.
        """

        if job_id in self._live and not self._live[job_id].task.done():
            return self._live[job_id]

        job_dir = self.data_dir / "jobs" / job_id
        if not (job_dir / "manifest.json").is_file():
            raise FileNotFoundError(
                f"cannot start job {job_id}: manifest.json missing at {job_dir}"
            )
        bus = ProgressEventBus(job_dir / "events.log")

        # Build providers up-front so any config error surfaces before the
        # background task is launched (and crash-marks the job correctly).
        try:
            language = self.provider_factory.language_factory()
            audio = self.provider_factory.audio_factory()
            visual = self.provider_factory.visual_factory()
            vision_fallback = self.provider_factory.vision_fallback_factory()
            lipsync = self.provider_factory.lipsync_factory()
        except Exception as exc:
            _logger.exception("provider build failed for job %s", job_id)
            self.repo.update_status(
                job_id, status=JobStatus.FAILED, error_message=f"provider build failed: {exc}"
            )
            raise

        deps = PipelineDependencies(
            language=language,
            audio=audio,
            visual=visual,
            repo=self.repo,
            bus=bus,
            parser=self.parser_factory(),
            vision_fallback=vision_fallback,
            ffmpeg_runner=self.ffmpeg_runner,
            ffmpeg_binary=self.ffmpeg_binary,
            lipsync=lipsync,
        )
        orchestrator = PipelineOrchestrator(deps)

        async def _runner() -> None:
            try:
                await orchestrator.run(job_dir=job_dir)
            except Exception:
                _logger.exception("orchestrator failed for job %s", job_id)
            finally:
                with suppress(Exception):
                    await language.close()
                with suppress(Exception):
                    await audio.close()
                with suppress(Exception):
                    await visual.close()
                if vision_fallback is not None:
                    with suppress(Exception):
                        await vision_fallback.close()
                if lipsync is not None:
                    with suppress(Exception):
                        await lipsync.close()
                await bus.close()

        task = asyncio.create_task(_runner(), name=f"booktoanime-job-{job_id}")
        running = _RunningJob(bus=bus, task=task)
        self._live[job_id] = running
        return running

    def get_bus(self, job_id: str) -> ProgressEventBus | None:
        running = self._live.get(job_id)
        return running.bus if running and not running.task.done() else None

    def is_running(self, job_id: str) -> bool:
        running = self._live.get(job_id)
        return bool(running and not running.task.done())

    async def shutdown(self) -> None:
        """Cancel every still-running job and wait for them to settle."""

        for running in self._live.values():
            if not running.task.done():
                running.task.cancel()
        for running in self._live.values():
            with suppress(asyncio.CancelledError, Exception):
                await running.task
            with suppress(Exception):
                await running.bus.close()
        self._live.clear()
