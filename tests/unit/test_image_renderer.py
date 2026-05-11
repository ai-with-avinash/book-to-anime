"""Phase-2 coverage for ``ShotImageRenderer``: visual_kind/figure_id
persistence + reconciler invalidation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from booktoanime.pipeline.artifacts import (
    ImagesIndex,
    KenBurns,
    Shot,
    ShotImageRecord,
    Storyboard,
    VisualKind,
)
from booktoanime.pipeline.events import ProgressEventBus
from booktoanime.pipeline.image_renderer import (
    ImageRendererConfig,
    ShotImageRenderer,
)
from booktoanime.providers.base import GeneratedImage, VisualProvider


def _shot(
    idx: int,
    *,
    visual_kind: VisualKind = VisualKind.ILLUSTRATION,
    figure_id: str | None = None,
) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id="topic_001",
        order=idx,
        narration_text="t",
        duration_seconds_target=4.0,
        image_prompt="prompt",
        seed=idx,
        ken_burns=KenBurns.model_validate(
            {"from": [0.0, 0.0, 1.0], "to": [0.05, 0.05, 1.1]}
        ),
        visual_kind=visual_kind,
        figure_id=figure_id,
    )


def _storyboard(shots: list[Shot]) -> Storyboard:
    return Storyboard(
        shots=shots,
        total_duration_seconds_target=float(len(shots)) * 4.0,
    )


class _RecordingVisual(VisualProvider):
    name = "rec"

    def __init__(self) -> None:
        self.rendered: list[str] = []

    async def prepare(self, *, panel_style: str, narrator_seed: int) -> Path:
        raise NotImplementedError

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        self.rendered.append(out_path.stem)
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
async def test_index_records_visual_kind_and_figure_id(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"

    storyboard = _storyboard(
        [
            _shot(1, visual_kind=VisualKind.TITLE_CARD),
            _shot(2, visual_kind=VisualKind.FIGURE, figure_id="fig_x"),
            _shot(3, visual_kind=VisualKind.ILLUSTRATION),
        ]
    )

    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        _RecordingVisual(),
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    index = await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    by_id = {r.shot_id: r for r in index.items}
    assert by_id["shot_0001"].visual_kind == VisualKind.TITLE_CARD
    assert by_id["shot_0001"].figure_id is None
    assert by_id["shot_0002"].visual_kind == VisualKind.FIGURE
    assert by_id["shot_0002"].figure_id == "fig_x"
    assert by_id["shot_0003"].visual_kind == VisualKind.ILLUSTRATION

    # Persisted on disk too.
    reloaded = ImagesIndex.from_path(job_dir / "images" / "index.json")
    reloaded_by_id = {r.shot_id: r for r in reloaded.items}
    assert reloaded_by_id["shot_0002"].figure_id == "fig_x"


@pytest.mark.asyncio
async def test_reconciler_invalidates_shot_on_visual_kind_mismatch(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    # On-disk record claims FIGURE kind; storyboard now says ILLUSTRATION.
    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "shot_0001.png")
    ImagesIndex(
        items=[
            ShotImageRecord(
                shot_id="shot_0001",
                file="images/shot_0001.png",
                seed=1,
                width=16,
                height=16,
                visual_kind=VisualKind.FIGURE,
                figure_id="fig_old",
            )
        ]
    ).save(images_dir / "index.json")

    storyboard = _storyboard([_shot(1, visual_kind=VisualKind.ILLUSTRATION)])

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    # Stale FIGURE record was dropped → renderer re-rendered the shot.
    assert visual.rendered == ["shot_0001"]
    final = ImagesIndex.from_path(images_dir / "index.json")
    assert len(final.items) == 1
    assert final.items[0].visual_kind == VisualKind.ILLUSTRATION
    assert final.items[0].figure_id is None


@pytest.mark.asyncio
async def test_reconciler_invalidates_shot_on_figure_id_mismatch(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "shot_0001.png")
    ImagesIndex(
        items=[
            ShotImageRecord(
                shot_id="shot_0001",
                file="images/shot_0001.png",
                seed=1,
                width=16,
                height=16,
                visual_kind=VisualKind.FIGURE,
                figure_id="fig_old",
            )
        ]
    ).save(images_dir / "index.json")

    storyboard = _storyboard(
        [_shot(1, visual_kind=VisualKind.FIGURE, figure_id="fig_new")]
    )

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    assert visual.rendered == ["shot_0001"]
    final = ImagesIndex.from_path(images_dir / "index.json")
    assert final.items[0].figure_id == "fig_new"


@pytest.mark.asyncio
async def test_reconciler_keeps_matching_records(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True)

    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "shot_0001.png")
    ImagesIndex(
        items=[
            ShotImageRecord(
                shot_id="shot_0001",
                file="images/shot_0001.png",
                seed=1,
                width=16,
                height=16,
                visual_kind=VisualKind.ILLUSTRATION,
                figure_id=None,
            )
        ]
    ).save(images_dir / "index.json")

    storyboard = _storyboard([_shot(1, visual_kind=VisualKind.ILLUSTRATION)])

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = ShotImageRenderer(
        visual,
        ImageRendererConfig(width=64, height=64, steps=2, guidance=4.0, concurrency=1),
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    # Record matched -> no re-render.
    assert visual.rendered == []
