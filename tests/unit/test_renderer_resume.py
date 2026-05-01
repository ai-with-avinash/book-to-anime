"""Verify ShotImageRenderer and TTSNarrator filesystem-truthful resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from booktoanime.errors import ProviderError
from booktoanime.pipeline.artifacts import (
    ImagesIndex,
    KenBurns,
    NarratorPersona,
    Shot,
    ShotImageRecord,
    Storyboard,
)
from booktoanime.pipeline.events import ProgressEventBus
from booktoanime.pipeline.image_renderer import ImageRendererConfig, ShotImageRenderer
from booktoanime.providers.base import GeneratedImage, VisualProvider


def _shot(idx: int) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text="t",
        duration_seconds_target=4.0,
        image_prompt="prompt",
        seed=idx,
        ken_burns=KenBurns.model_validate({"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}),
    )


def _storyboard(n: int) -> Storyboard:
    return Storyboard(shots=[_shot(i) for i in range(1, n + 1)], total_duration_seconds_target=n * 4.0)


class _RecordingVisual(VisualProvider):
    name = "rec"

    def __init__(self, *, persona_dir: Path, fail_on: set[str] | None = None) -> None:
        self._persona_dir = persona_dir
        self._fail_on = fail_on or set()
        self.rendered_ids: list[str] = []

    async def prepare(self, *, anime_style: str, narrator_seed: int) -> Path:
        self._persona_dir.mkdir(parents=True, exist_ok=True)
        path = self._persona_dir / f"{anime_style}__{narrator_seed}.png"
        if not path.is_file():
            Image.new("RGB", (32, 32), (10, 30, 60)).save(path)
        return path

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        shot_id = out_path.stem
        if shot_id in self._fail_on:
            raise ProviderError(f"forced fail on {shot_id}")
        self.rendered_ids.append(shot_id)
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


@pytest.mark.asyncio
async def test_resume_skips_shots_with_file_and_index_entry(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    # Pre-stage one finished shot on disk + in index.
    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "shot_0001.png")
    ImagesIndex(
        items=[ShotImageRecord(shot_id="shot_0001", file="images/shot_0001.png", seed=1, width=16, height=16)]
    ).save(images_dir / "index.json")

    persona = NarratorPersona(seed=42, style_descriptor="shounen-bright")
    visual = _RecordingVisual(persona_dir=tmp_path / "personas")
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=128, height=128, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    storyboard = _storyboard(3)
    index, _ = await renderer.render(storyboard=storyboard, persona=persona, job_dir=job_dir)
    await bus.close()

    # Shot 1 was already done, so only shots 2 & 3 should have been rendered.
    assert visual.rendered_ids == ["shot_0002", "shot_0003"]
    assert {r.shot_id for r in index.items} == {"shot_0001", "shot_0002", "shot_0003"}


@pytest.mark.asyncio
async def test_index_entry_with_missing_file_triggers_rerender(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    # Index lists shot_0001 but the file was deleted.
    ImagesIndex(
        items=[ShotImageRecord(shot_id="shot_0001", file="images/shot_0001.png", seed=1, width=16, height=16)]
    ).save(images_dir / "index.json")
    assert not (images_dir / "shot_0001.png").exists()

    persona = NarratorPersona(seed=42, style_descriptor="shounen-bright")
    visual = _RecordingVisual(persona_dir=tmp_path / "personas")
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    storyboard = _storyboard(1)
    index, _ = await renderer.render(storyboard=storyboard, persona=persona, job_dir=job_dir)
    await bus.close()

    assert visual.rendered_ids == ["shot_0001"]
    assert (images_dir / "shot_0001.png").is_file()
    assert len(index.items) == 1


@pytest.mark.asyncio
async def test_orphan_file_without_index_is_adopted(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    # File on disk for shot_0001, no index.json.
    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "shot_0001.png")

    persona = NarratorPersona(seed=42, style_descriptor="shounen-bright")
    visual = _RecordingVisual(persona_dir=tmp_path / "personas")
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    storyboard = _storyboard(2)
    index, _ = await renderer.render(storyboard=storyboard, persona=persona, job_dir=job_dir)
    await bus.close()

    # Orphan adopted -> renderer only had to render shot_0002.
    assert visual.rendered_ids == ["shot_0002"]
    assert {r.shot_id for r in index.items} == {"shot_0001", "shot_0002"}


@pytest.mark.asyncio
async def test_partial_failure_persists_completed_records(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    persona = NarratorPersona(seed=42, style_descriptor="shounen-bright")
    visual = _RecordingVisual(persona_dir=tmp_path / "personas", fail_on={"shot_0002"})
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    storyboard = _storyboard(3)

    with pytest.raises(ProviderError):
        await renderer.render(storyboard=storyboard, persona=persona, job_dir=job_dir)
    await bus.close()

    # Index file persists shot_0001 (pre-failure) but not shot_0002 (failed).
    index = ImagesIndex.from_path(job_dir / "images" / "index.json")
    persisted = {r.shot_id for r in index.items}
    assert "shot_0001" in persisted
    assert "shot_0002" not in persisted
