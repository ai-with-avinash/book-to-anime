"""End-to-end pipeline orchestrator.

Responsibilities:

* Inspect ``manifest.json`` to find the first non-completed stage.
* Run stages in order, persisting an artifact (and updating the manifest) on
  every successful completion. A failed stage leaves enough on disk for the
  next run to pick up where it left off.
* Emit progress events through a :class:`ProgressEventBus` so the SSE route
  and the on-disk events log stay in sync.
* Update the SQLite ``JobRecord`` so ``booktoanime list`` / future-resume
  logic can find resumable jobs without scanning every directory.

The orchestrator never imports concrete adapters — it takes already-built
:class:`LanguageProvider`, :class:`AudioProvider`, and :class:`VisualProvider`
instances. The CLI / API layer is responsible for building them via the
provider registry.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ParsingError
from ..parsing import PDFParser
from ..parsing.models import ParsedDocument
from ..providers import AudioProvider, LanguageProvider, VisualProvider
from ..state import JobRepository, JobStatus
from .artifacts import (
    AudioIndex,
    ImagesIndex,
    Storyboard,
    StructuredDocument,
)
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .image_renderer import ImageRendererConfig, ShotImageRenderer
from .manifest import JobManifest
from .stages import STAGE_ORDER, Stage
from .storyboard import StoryboardBuilder, StoryboardConfig
from .summarizer import SummarizationConfig, TopicSummarizer
from .topic_segmenter import TopicSegmenter
from .tts_narrator import TTSNarrator, TTSNarratorConfig
from .video_assembler import FFmpegRunner, VideoAssembler, VideoAssemblerConfig

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineDependencies:
    language: LanguageProvider
    audio: AudioProvider
    visual: VisualProvider
    repo: JobRepository
    bus: ProgressEventBus
    parser: PDFParser
    vision_fallback: LanguageProvider | None = None
    # Optional ffmpeg runner override (tests inject a stub that fakes the binary).
    ffmpeg_runner: FFmpegRunner | None = None
    ffmpeg_binary: str | None = None


class PipelineOrchestrator:
    """Runs the whole pipeline for a single job."""

    def __init__(self, deps: PipelineDependencies) -> None:
        self._deps = deps

    async def run(self, *, job_dir: Path) -> JobManifest:
        manifest_path = job_dir / "manifest.json"
        manifest = JobManifest.from_path(manifest_path)
        return await self._run_internal(manifest=manifest, job_dir=job_dir)

    async def _run_internal(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> JobManifest:
        manifest_path = job_dir / "manifest.json"
        repo = self._deps.repo

        repo.update_status(manifest.job_id, status=JobStatus.RUNNING)

        try:
            for stage in STAGE_ORDER:
                if manifest.stage_status(stage).value == "completed":
                    continue
                manifest.mark_started(stage)
                await asyncio.to_thread(manifest.save, manifest_path)
                repo.update_status(
                    manifest.job_id, status=JobStatus.RUNNING, current_stage=stage.value
                )
                await self._deps.bus.emit(
                    ProgressEvent(
                        kind=ProgressKind.STAGE_STARTED,
                        stage=stage.value,
                        message=f"stage {stage.value} started",
                    )
                )
                try:
                    await self._run_stage(stage, manifest=manifest, job_dir=job_dir)
                except BookToAnimeError as exc:
                    manifest.mark_failed(stage, str(exc))
                    await asyncio.to_thread(manifest.save, manifest_path)
                    repo.update_status(
                        manifest.job_id,
                        status=JobStatus.FAILED,
                        current_stage=stage.value,
                        error_message=exc.user_message,
                    )
                    await self._deps.bus.emit(
                        ProgressEvent(
                            kind=ProgressKind.STAGE_FAILED,
                            stage=stage.value,
                            message=str(exc),
                            user_message=exc.user_message,
                        )
                    )
                    raise

                manifest.mark_completed(stage)
                await asyncio.to_thread(manifest.save, manifest_path)
                await self._deps.bus.emit(
                    ProgressEvent(
                        kind=ProgressKind.STAGE_COMPLETED,
                        stage=stage.value,
                        message=f"stage {stage.value} completed",
                    )
                )
        except BookToAnimeError:
            raise
        else:
            repo.update_status(manifest.job_id, status=JobStatus.COMPLETED)
        return manifest

    # ----------------------------------------------------------- per-stage

    async def _run_stage(
        self,
        stage: Stage,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        if stage == Stage.PARSING:
            await self._run_parsing(job_dir=job_dir)
        elif stage == Stage.STRUCTURING:
            await self._run_structuring(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.PERSONA_SEEDING:
            await self._run_persona_seeding(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.STORYBOARD:
            await self._run_storyboard(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.IMAGES:
            await self._run_images(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.AUDIO:
            await self._run_audio(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.ASSEMBLY:
            await self._run_assembly(manifest=manifest, job_dir=job_dir)
        else:  # pragma: no cover - exhaustively handled above
            raise NotImplementedError(stage)

    async def _run_parsing(self, *, job_dir: Path) -> None:
        source = job_dir / "source.pdf"
        if not source.is_file():
            raise ParsingError(f"missing source PDF at {source}")

        parser = self._deps.parser
        parsed = await asyncio.to_thread(parser.parse, source, job_dir=job_dir)
        out_path = job_dir / "extracted" / "parsed.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(out_path.write_bytes, parsed.to_json_bytes())

    async def _run_structuring(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        parsed = ParsedDocument.from_path(job_dir / "extracted" / "parsed.json")

        spans = TopicSegmenter().segment(parsed)
        summarizer = TopicSummarizer(
            self._deps.language,
            SummarizationConfig(
                depth=manifest.config.depth,
                length_preset=manifest.config.length_preset,
                minutes_per_topic=manifest.config.minutes_per_topic,
            ),
        )
        sections = await summarizer.summarize_topics(parsed, spans)

        # NOTE: Per-image VLM explanations will be wired into the storyboard
        # prompts in module 6/7 when there's a downstream consumer. Calling
        # the explainer here without using the result was burning vision-
        # provider credits with no observable effect.

        await asyncio.to_thread(
            StructuredDocument(topics=sections).save,
            job_dir / "structured.json",
        )

    async def _run_persona_seeding(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        """Phase-1 stub for the persona-seeding stage.

        The original anime-narrator persona was removed when the project
        pivoted away from character-driven explainers. Phase 2 replaces this
        stub with ``STYLE_SEEDING`` — a one-time style-anchor render used by
        IP-Adapter on SDXL fallback shots.
        """

        del manifest, job_dir  # phase-1 stub doesn't need either
        await self._deps.bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.PERSONA_SEEDING.value,
                message=(
                    "persona seeding stubbed; replaced in phase 2 with style "
                    "seeding"
                ),
            )
        )

    async def _run_storyboard(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        structured = StructuredDocument.from_path(job_dir / "structured.json")
        builder = StoryboardBuilder(
            StoryboardConfig(panel_style=manifest.config.panel_style)
        )
        storyboard = builder.build(structured.topics)
        await asyncio.to_thread(storyboard.save, job_dir / "storyboard.json")

    async def _run_images(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        storyboard = Storyboard.from_path(job_dir / "storyboard.json")

        renderer = ShotImageRenderer(
            self._deps.visual,
            ImageRendererConfig(
                concurrency=_concurrency_for_profile(manifest.config.profile),
                width=_width_for_aspect(manifest.config.aspect_ratio),
                height=_height_for_aspect(manifest.config.aspect_ratio),
                panel_style=manifest.config.panel_style,
            ),
            bus=self._deps.bus,
        )
        await renderer.render(
            storyboard=storyboard,
            job_dir=job_dir,
        )

    async def _run_audio(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        storyboard = Storyboard.from_path(job_dir / "storyboard.json")
        narrator = TTSNarrator(
            self._deps.audio,
            TTSNarratorConfig(
                voice_id=manifest.config.narration.voice_id,
                language=manifest.config.narration.language,
                speed=manifest.config.narration.speed,
                concurrency=_concurrency_for_profile(manifest.config.profile),
            ),
            bus=self._deps.bus,
        )
        await narrator.synthesize(storyboard=storyboard, job_dir=job_dir)

    async def _run_assembly(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        storyboard = Storyboard.from_path(job_dir / "storyboard.json")
        audio_index = AudioIndex.from_path(job_dir / "audio" / "index.json")
        images_index = ImagesIndex.from_path(job_dir / "images" / "index.json")

        assembler = VideoAssembler(
            VideoAssemblerConfig(
                width=_width_for_aspect(manifest.config.aspect_ratio),
                height=_height_for_aspect(manifest.config.aspect_ratio),
            ),
            bus=self._deps.bus,
            runner=self._deps.ffmpeg_runner,
            ffmpeg_binary=self._deps.ffmpeg_binary,
        )
        await assembler.assemble(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            job_dir=job_dir,
        )


# --------------------------------------------------------------- helpers


def _concurrency_for_profile(profile: str) -> int:
    return {"default": 2, "high_quality": 1, "low_vram": 1}.get(profile, 2)


_ASPECT_RATIO_TO_DIMS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


def _width_for_aspect(aspect: str) -> int:
    return _ASPECT_RATIO_TO_DIMS.get(aspect, (1920, 1080))[0]


def _height_for_aspect(aspect: str) -> int:
    return _ASPECT_RATIO_TO_DIMS.get(aspect, (1920, 1080))[1]


__all__ = [
    "AudioIndex",
    "ImagesIndex",
    "JobManifest",
    "PipelineDependencies",
    "PipelineOrchestrator",
    "Storyboard",
    "StructuredDocument",
]
