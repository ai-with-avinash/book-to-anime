"""End-to-end orchestrator test on the tiny PDF fixture with mocked providers.

We run the full pipeline (parsing -> structuring -> storyboard -> images ->
audio), then simulate a mid-images failure and verify ``resume`` picks up at
the failed shot.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from booktoanime.errors import ProviderError
from booktoanime.parsing import PDFParser
from booktoanime.pipeline import PipelineOrchestrator
from booktoanime.pipeline.artifacts import AudioIndex, ImagesIndex, Storyboard, StructuredDocument
from booktoanime.pipeline.events import ProgressEventBus
from booktoanime.pipeline.manifest import (
    JobConfig,
    JobManifest,
    NarrationConfig,
    ProvidersConfig,
)
from booktoanime.pipeline.orchestrator import PipelineDependencies
from booktoanime.providers.base import (
    AudioProvider,
    GeneratedAudio,
    GeneratedImage,
    ImageExplanation,
    LanguageProvider,
    VisualProvider,
)
from booktoanime.state import JobRepository, JobStatus, open_database

# --------------------------------------------------------------- fakes


class _FakeLanguage(LanguageProvider):
    name = "fake"

    def __init__(self) -> None:
        self.complete_calls = 0

    async def complete(self, request: Any) -> str:
        self.complete_calls += 1
        return json.dumps(
            {
                "summary": (
                    "This topic explains the introduction. We cover the main idea. "
                    "Then we walk through one example. Finally we summarize the takeaway."
                ),
                "key_points": ["the main idea", "one example", "the takeaway"],
                "estimated_words": 60,
            }
        )

    async def explain_image(self, image: Any, **_: Any) -> ImageExplanation:
        return ImageExplanation(summary="image summary", detail="image detail")

    async def close(self) -> None:
        return None


class _FakeAudio(AudioProvider):
    name = "fake_audio"

    def __init__(self) -> None:
        self.calls = 0

    async def list_voices(self, language: str | None = None) -> Sequence[str]:
        return ("fake_voice",)

    async def synthesize(self, request: Any, out_path: Path) -> GeneratedAudio:
        self.calls += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 24_000
        duration = max(0.2, min(2.0, len(request.text.split()) * 0.3))
        wave = np.zeros(int(sample_rate * duration), dtype=np.float32)
        sf.write(str(out_path), wave, sample_rate, "PCM_16")
        return GeneratedAudio(path=out_path, duration_seconds=duration, sample_rate=sample_rate)

    async def close(self) -> None:
        return None


class _FakeVisual(VisualProvider):
    name = "fake_visual"

    def __init__(self, *, persona_dir: Path, fail_after: int | None = None) -> None:
        self._persona_dir = persona_dir
        self._fail_after = fail_after
        self.render_count = 0

    async def prepare(self, *, panel_style: str, narrator_seed: int) -> Path:
        self._persona_dir.mkdir(parents=True, exist_ok=True)
        path = self._persona_dir / f"{panel_style}__{narrator_seed}.png"
        if not path.is_file():
            Image.new("RGB", (64, 64), (10, 30, 60)).save(path)
        return path

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        if self._fail_after is not None and self.render_count >= self._fail_after:
            raise RuntimeError("forced render failure")
        self.render_count += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), (40, 80, 160)).save(out_path)
        return GeneratedImage(
            path=out_path,
            seed=request.seed,
            width=request.width,
            height=request.height,
        )

    async def close(self) -> None:
        return None


# --------------------------------------------------------------- fixtures


def _build_manifest(*, job_id: str, source_pdf: str = "source.pdf") -> JobManifest:
    config = JobConfig(
        panel_style="clean-linework",
        narration=NarrationConfig(voice_id="fake_voice", language="en-US"),
        depth="undergraduate",
        length_preset="short",
        aspect_ratio="16:9",
        profile="default",
        providers=ProvidersConfig(language="fake", audio="fake_audio", visual="fake_visual"),
    )
    return JobManifest.for_new_job(job_id=job_id, config=config, source_pdf=source_pdf)


async def _stub_ffmpeg_runner(argv, log_path):
    out_path = Path(argv[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"\x00\x00\x00 ftypisom" + b"X" * 64)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stub ffmpeg ok", encoding="utf-8")


def _build_deps(
    *,
    job_dir: Path,
    persona_dir: Path,
    repo: JobRepository,
    visual: _FakeVisual | None = None,
    bus: ProgressEventBus | None = None,
) -> tuple[PipelineDependencies, ProgressEventBus, _FakeVisual, _FakeLanguage, _FakeAudio]:
    bus = bus or ProgressEventBus(job_dir / "events.log")
    language = _FakeLanguage()
    audio = _FakeAudio()
    visual_impl = visual or _FakeVisual(persona_dir=persona_dir)
    deps = PipelineDependencies(
        language=language,
        audio=audio,
        visual=visual_impl,
        repo=repo,
        bus=bus,
        parser=PDFParser(),
        ffmpeg_runner=_stub_ffmpeg_runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    return deps, bus, visual_impl, language, audio


# --------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_orchestrator_runs_to_completion(
    tmp_path: Path,
    tiny_pdf: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "source.pdf").write_bytes(tiny_pdf.read_bytes())

    repo = JobRepository(open_database(tmp_path / "state.db"))
    manifest = _build_manifest(job_id="01HEXAMPLE")
    repo.create_job(
        job_id=manifest.job_id,
        source_pdf=tiny_pdf,
        data_dir=tmp_path,
        config=manifest.config.model_dump(mode="json"),
    )
    manifest.save(job_dir / "manifest.json")

    deps, bus, visual, language, audio = _build_deps(
        job_dir=job_dir, persona_dir=tmp_path / "personas", repo=repo
    )

    orchestrator = PipelineOrchestrator(deps)
    final_manifest = await orchestrator.run(job_dir=job_dir)

    for stage in (
        "parsing",
        "structuring",
        "style_seeding",
        "storyboard",
        "images",
        "audio",
        "assembly",
    ):
        assert final_manifest.stages[stage].status.value == "completed", stage
    # Assembly produced the final outputs.
    assert (job_dir / "output.mp4").is_file()
    assert (job_dir / "output.srt").is_file()

    # Artifacts on disk.
    assert (job_dir / "extracted" / "parsed.json").is_file()
    structured = StructuredDocument.from_path(job_dir / "structured.json")
    assert structured.topics, "expected at least one topic"

    storyboard = Storyboard.from_path(job_dir / "storyboard.json")
    assert storyboard.shots, "expected at least one shot"

    images_index = ImagesIndex.from_path(job_dir / "images" / "index.json")
    assert len(images_index.items) == len(storyboard.shots)
    for record in images_index.items:
        assert (job_dir / record.file).is_file()

    audio_index = AudioIndex.from_path(job_dir / "audio" / "index.json")
    assert len(audio_index.items) == len(storyboard.shots)
    for record in audio_index.items:
        assert (job_dir / record.file).is_file()

    # SQLite job row reflects completion.
    db_record = repo.get(manifest.job_id)
    assert db_record is not None
    assert db_record.status == JobStatus.COMPLETED

    # Bus emitted stage-completed events for every stage.
    await bus.close()
    log_lines = (job_dir / "events.log").read_text("utf-8").splitlines()
    completed_stages = {
        json.loads(line)["stage"]
        for line in log_lines
        if json.loads(line)["kind"] == "stage_completed"
    }
    assert {"parsing", "structuring", "storyboard", "images", "audio"}.issubset(completed_stages)

    # render() runs once per shot PLUS once for the style-seeder anchor.
    assert visual.render_count == len(storyboard.shots) + 1
    assert audio.calls == len(storyboard.shots)
    assert language.complete_calls == len(structured.topics)


# --------------------------------------------------------------- resume


@pytest.mark.asyncio
async def test_orchestrator_resumes_after_image_failure(
    tmp_path: Path,
    tiny_pdf: Path,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "source.pdf").write_bytes(tiny_pdf.read_bytes())

    repo = JobRepository(open_database(tmp_path / "state.db"))
    manifest = _build_manifest(job_id="01HRESUME")
    repo.create_job(
        job_id=manifest.job_id,
        source_pdf=tiny_pdf,
        data_dir=tmp_path,
        config=manifest.config.model_dump(mode="json"),
    )
    manifest.save(job_dir / "manifest.json")

    persona_dir = tmp_path / "personas"

    # First run fails after two successful renders. The first render is the
    # style-seeder anchor (STYLE_SEEDING stage); the second is the first
    # storyboard shot. The third call (second image shot) raises.
    failing_visual = _FakeVisual(persona_dir=persona_dir, fail_after=2)
    deps_a, bus_a, _, _, _ = _build_deps(
        job_dir=job_dir, persona_dir=persona_dir, repo=repo, visual=failing_visual
    )
    orchestrator_a = PipelineOrchestrator(deps_a)
    # The fake visual provider raises RuntimeError after `fail_after` successful
    # renders; the image renderer wraps unhandled errors as ProviderError.
    with pytest.raises(ProviderError):
        await orchestrator_a.run(job_dir=job_dir)
    await bus_a.close()

    # Manifest records images stage as failed; structuring + storyboard completed.
    intermediate = JobManifest.from_path(job_dir / "manifest.json")
    assert intermediate.stages["structuring"].status.value == "completed"
    assert intermediate.stages["storyboard"].status.value == "completed"
    assert intermediate.stages["images"].status.value == "failed"

    partial_index = ImagesIndex.from_path(job_dir / "images" / "index.json")
    assert len(partial_index.items) == 1
    # render_count counts every successful provider.render() call across both
    # stages; the style anchor is one of them.
    rendered_so_far = failing_visual.render_count
    assert rendered_so_far == 2
    image_renders_so_far = len(partial_index.items)
    assert image_renders_so_far == 1

    db_record = repo.get(manifest.job_id)
    assert db_record is not None
    assert db_record.status == JobStatus.FAILED

    # Second run: a fresh visual that does not fail. It should only re-render
    # the *missing* shots, not the one already on disk.
    successful_visual = _FakeVisual(persona_dir=persona_dir)
    deps_b, bus_b, _, _, _ = _build_deps(
        job_dir=job_dir, persona_dir=persona_dir, repo=repo, visual=successful_visual
    )
    orchestrator_b = PipelineOrchestrator(deps_b)
    final = await orchestrator_b.run(job_dir=job_dir)
    await bus_b.close()

    storyboard = Storyboard.from_path(job_dir / "storyboard.json")
    expected_total = len(storyboard.shots)
    # Style anchor was already written by the failing run; the resumed visual
    # only needs to render the image-stage shots that hadn't completed.
    assert successful_visual.render_count == expected_total - image_renders_so_far

    final_index = ImagesIndex.from_path(job_dir / "images" / "index.json")
    assert len(final_index.items) == expected_total

    audio_index = AudioIndex.from_path(job_dir / "audio" / "index.json")
    assert len(audio_index.items) == expected_total

    for stage in ("parsing", "structuring", "storyboard", "images", "audio"):
        assert final.stages[stage].status.value == "completed", stage

    db_record = repo.get(manifest.job_id)
    assert db_record is not None
    assert db_record.status == JobStatus.COMPLETED
