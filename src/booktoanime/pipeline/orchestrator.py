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

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ParsingError, ProviderError
from ..parsing import PDFParser
from ..parsing.models import ParsedDocument
from ..providers import AudioProvider, LanguageProvider, VisualProvider
from ..providers.base import LipSyncProvider
from ..state import JobRepository, JobStatus
from .artifacts import (
    AudioIndex,
    ImagesIndex,
    MouthIndex,
    NarratorPersona,
    Storyboard,
    StructuredDocument,
)
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .image_renderer import ImageRendererConfig, ShotImageRenderer
from .manifest import JobManifest
from .mouth_animator import MouthAnimator, MouthAnimatorConfig
from .narrator_persona import PersonaSeederConfig, derive_persona
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
    # Lip-sync is opt-in. When the JobConfig disables it the orchestrator
    # short-circuits the MOUTH_ANIMATION stage without touching this provider,
    # so callers that don't ship lip-sync can leave it None.
    lipsync: LipSyncProvider | None = None


class PipelineOrchestrator:
    """Runs the whole pipeline (except video assembly) for a single job.

    Video assembly lives in module 7 and is invoked by the orchestrator's
    caller after this class succeeds. Keeping assembly out of here lets
    module 5 ship + be tested without an ffmpeg dependency.
    """

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
                manifest.save(manifest_path)
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
                    manifest.save(manifest_path)
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
                manifest.save(manifest_path)
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
        elif stage == Stage.STORYBOARD:
            await self._run_storyboard(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.IMAGES:
            await self._run_images(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.AUDIO:
            await self._run_audio(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.MOUTH_ANIMATION:
            await self._run_mouth_animation(manifest=manifest, job_dir=job_dir)
        elif stage == Stage.ASSEMBLY:
            await self._run_assembly(manifest=manifest, job_dir=job_dir)
        else:  # pragma: no cover - exhaustively handled above
            raise NotImplementedError(stage)

    async def _run_parsing(self, *, job_dir: Path) -> None:
        import asyncio

        source = job_dir / "source.pdf"
        if not source.is_file():
            raise ParsingError(f"missing source PDF at {source}")

        parser = self._deps.parser
        parsed = await asyncio.to_thread(parser.parse, source, job_dir=job_dir)
        out_path = job_dir / "extracted" / "parsed.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(parsed.to_json_bytes())

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

        persona = derive_persona(
            PersonaSeederConfig(
                anime_style=manifest.config.anime_style,
                narration_language=manifest.config.narration.language,
                voice_id=manifest.config.narration.voice_id,
            )
        )
        StructuredDocument(topics=sections, narrator_persona=persona).save(
            job_dir / "structured.json"
        )

    async def _run_storyboard(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        structured = StructuredDocument.from_path(job_dir / "structured.json")
        builder = StoryboardBuilder(
            StoryboardConfig(anime_style=manifest.config.anime_style)
        )
        storyboard = builder.build(structured.topics, structured.narrator_persona)
        storyboard.save(job_dir / "storyboard.json")

    async def _run_images(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        structured = StructuredDocument.from_path(job_dir / "structured.json")
        storyboard = Storyboard.from_path(job_dir / "storyboard.json")

        renderer = ShotImageRenderer(
            self._deps.visual,
            ImageRendererConfig(
                concurrency=_concurrency_for_profile(manifest.config.profile),
                width=_width_for_aspect(manifest.config.aspect_ratio),
                height=_height_for_aspect(manifest.config.aspect_ratio),
                anime_style=manifest.config.anime_style,
            ),
            bus=self._deps.bus,
        )
        _index, persona_reference = await renderer.render(
            storyboard=storyboard,
            persona=structured.narrator_persona,
            job_dir=job_dir,
        )

        # Copy the persona reference into the job directory so the artifact
        # path stays portable (relative to job_dir). The renderer's reference
        # may live under the model cache or the user data dir.
        if structured.narrator_persona.reference_image is None:
            persona_dst_dir = job_dir / "personas"
            persona_dst_dir.mkdir(parents=True, exist_ok=True)
            persona_dst = persona_dst_dir / persona_reference.name
            if persona_reference.resolve() != persona_dst.resolve():
                shutil.copyfile(persona_reference, persona_dst)
            persona_rel = f"personas/{persona_dst.name}"
            structured = structured.model_copy(
                update={
                    "narrator_persona": structured.narrator_persona.model_copy(
                        update={"reference_image": persona_rel}
                    )
                }
            )
            structured.save(job_dir / "structured.json")

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

    async def _run_mouth_animation(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        if not manifest.config.lipsync.enabled:
            await self._deps.bus.emit(
                ProgressEvent(
                    kind=ProgressKind.INFO,
                    stage=Stage.MOUTH_ANIMATION.value,
                    message="lipsync disabled; skipping mouth animation",
                )
            )
            return

        if self._deps.lipsync is None:
            raise ProviderError(
                "lipsync.enabled is true but no LipSyncProvider was wired into "
                "PipelineDependencies; pass `lipsync=...` from the launcher or "
                "set lipsync.enabled to false in the job config."
            )

        storyboard = Storyboard.from_path(job_dir / "storyboard.json")
        images_index = ImagesIndex.from_path(job_dir / "images" / "index.json")
        audio_index = AudioIndex.from_path(job_dir / "audio" / "index.json")

        animator = MouthAnimator(
            self._deps.lipsync,
            MouthAnimatorConfig(
                concurrency=_concurrency_for_profile(manifest.config.profile),
            ),
            bus=self._deps.bus,
        )
        await animator.animate(
            storyboard=storyboard,
            images_index=images_index,
            audio_index=audio_index,
            job_dir=job_dir,
        )

    async def _run_assembly(
        self,
        *,
        manifest: JobManifest,
        job_dir: Path,
    ) -> None:
        storyboard = Storyboard.from_path(job_dir / "storyboard.json")
        audio_index = AudioIndex.from_path(job_dir / "audio" / "index.json")
        images_index = ImagesIndex.from_path(job_dir / "images" / "index.json")
        mouth_index_path = job_dir / "mouth" / "index.json"
        mouth_index: MouthIndex | None = None
        if manifest.config.lipsync.enabled and mouth_index_path.is_file():
            mouth_index = MouthIndex.from_path(mouth_index_path)

        assembler = VideoAssembler(
            VideoAssemblerConfig(
                width=_width_for_aspect(manifest.config.aspect_ratio),
                height=_height_for_aspect(manifest.config.aspect_ratio),
                preserve_ken_burns=manifest.config.lipsync.preserve_ken_burns,
            ),
            bus=self._deps.bus,
            runner=self._deps.ffmpeg_runner,
            ffmpeg_binary=self._deps.ffmpeg_binary,
        )
        await assembler.assemble(
            storyboard=storyboard,
            audio_index=audio_index,
            images_index=images_index,
            mouth_index=mouth_index,
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
    "NarratorPersona",
    "PipelineDependencies",
    "PipelineOrchestrator",
    "Storyboard",
    "StructuredDocument",
]
