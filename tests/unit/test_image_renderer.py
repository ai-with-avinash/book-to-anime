"""Phase-2 + phase-3 coverage for ``ShotImageRenderer``.

Phase 2 covered ``visual_kind`` / ``figure_id`` persistence + reconciler
invalidation. Phase 3 adds:

* dispatch on ``Shot.visual_kind`` (FIGURE + TITLE_CARD bypass SDXL),
* small-figure fall-through to SDXL,
* split semaphore concurrency (GPU vs CPU),
* end-of-stage telemetry event.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from booktoanime.errors import RenderError
from booktoanime.parsing.models import ExtractedImage
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
    topic_id: str = "topic_001",
) -> Shot:
    return Shot(
        id=f"shot_{idx:04d}",
        topic_id=topic_id,
        order=idx,
        narration_text=f"narration {idx}",
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


def _write_figure(job_dir: Path, fig_id: str, size: tuple[int, int]) -> ExtractedImage:
    """Synthesize an extracted figure PNG on disk + return its metadata."""

    rel = f"extracted/figures/{fig_id}.png"
    out = job_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 100, 50)).save(out)
    return ExtractedImage(id=fig_id, file=rel, width=size[0], height=size[1])


class _RecordingVisual(VisualProvider):
    """Stand-in :class:`VisualProvider` that records every render call."""

    name = "rec"

    def __init__(self, *, render_delay: float = 0.0) -> None:
        self.rendered: list[str] = []
        self.concurrent_now = 0
        self.max_concurrent = 0
        self._render_delay = render_delay

    async def prepare(self, *, panel_style: str, narrator_seed: int) -> Path:
        raise NotImplementedError

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        self.concurrent_now += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent_now)
        try:
            if self._render_delay:
                await asyncio.sleep(self._render_delay)
            self.rendered.append(out_path.stem)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (request.width, request.height), (40, 80, 160)).save(out_path)
            return GeneratedImage(
                path=out_path,
                seed=request.seed,
                width=request.width,
                height=request.height,
            )
        finally:
            self.concurrent_now -= 1

    async def close(self) -> None:
        return None


def _event_messages(log_path: Path) -> list[str]:
    """Return all ``message`` fields from the NDJSON event log on disk."""

    import json

    out: list[str] = []
    if not log_path.is_file():
        return out
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line).get("message", ""))
    return out


def _renderer(
    visual: VisualProvider,
    *,
    extracted_images: list[ExtractedImage] | None = None,
    topic_titles: dict[str, str] | None = None,
    style_reference: Path | None = None,
    concurrency: int = 1,
    bus: ProgressEventBus,
) -> ShotImageRenderer:
    return ShotImageRenderer(
        visual,
        ImageRendererConfig(
            width=320,
            height=180,
            steps=2,
            guidance=4.0,
            concurrency=concurrency,
            panel_style="clean-linework",
        ),
        bus=bus,
        extracted_images=extracted_images,
        topic_titles=topic_titles,
        style_reference_path=style_reference,
    )


# ---------------------------------------------------------------- phase 2 carry-over


@pytest.mark.asyncio
async def test_index_records_visual_kind_and_figure_id(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    figure = _write_figure(job_dir, "fig_x", (640, 480))

    storyboard = _storyboard(
        [
            _shot(1, visual_kind=VisualKind.TITLE_CARD),
            _shot(2, visual_kind=VisualKind.FIGURE, figure_id=figure.id),
            _shot(3, visual_kind=VisualKind.ILLUSTRATION),
        ]
    )

    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(
        _RecordingVisual(),
        extracted_images=[figure],
        topic_titles={"topic_001": "Newton's First Law"},
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
    renderer = _renderer(visual, bus=bus)
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

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
    figure_new = _write_figure(job_dir, "fig_new", (640, 480))

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
        [_shot(1, visual_kind=VisualKind.FIGURE, figure_id=figure_new.id)]
    )

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(
        visual,
        extracted_images=[figure_new],
        topic_titles={"topic_001": "Topic"},
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    # FIGURE shots bypass SDXL — provider's render() must not have been called.
    assert visual.rendered == []
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
    renderer = _renderer(visual, bus=bus)
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    assert visual.rendered == []


# ---------------------------------------------------------------- phase 3 dispatch


@pytest.mark.asyncio
async def test_figure_shot_bypasses_sdxl(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    figure = _write_figure(job_dir, "fig_a", (640, 400))

    storyboard = _storyboard(
        [_shot(1, visual_kind=VisualKind.FIGURE, figure_id=figure.id)]
    )

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(
        visual,
        extracted_images=[figure],
        topic_titles={"topic_001": "Forces"},
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    assert visual.rendered == []
    out_path = job_dir / "images" / "shot_0001.png"
    assert out_path.is_file()
    img = Image.open(out_path)
    assert img.size == (320, 180)


@pytest.mark.asyncio
async def test_title_card_shot_bypasses_sdxl(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"

    storyboard = _storyboard([_shot(1, visual_kind=VisualKind.TITLE_CARD)])

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(
        visual,
        topic_titles={"topic_001": "Conservation of Energy"},
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    assert visual.rendered == []
    out_path = job_dir / "images" / "shot_0001.png"
    assert out_path.is_file()


@pytest.mark.asyncio
async def test_illustration_shot_calls_sdxl_once(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"

    storyboard = _storyboard([_shot(1, visual_kind=VisualKind.ILLUSTRATION)])

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(visual, bus=bus)
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    assert visual.rendered == ["shot_0001"]


@pytest.mark.asyncio
async def test_figure_shot_missing_figure_id_raises_render_error(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"

    bad_shot = _shot(1, visual_kind=VisualKind.FIGURE, figure_id=None)
    storyboard = _storyboard([bad_shot])

    visual = _RecordingVisual()
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(visual, bus=bus)
    with pytest.raises(RenderError):
        await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()


@pytest.mark.asyncio
async def test_figure_shot_unknown_figure_id_raises_render_error(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"

    storyboard = _storyboard(
        [_shot(1, visual_kind=VisualKind.FIGURE, figure_id="missing")]
    )
    bus = ProgressEventBus(job_dir / "events.log")
    renderer = _renderer(_RecordingVisual(), extracted_images=[], bus=bus)
    with pytest.raises(RenderError):
        await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()


@pytest.mark.asyncio
async def test_small_figure_falls_through_to_sdxl(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    tiny = _write_figure(job_dir, "fig_tiny", (64, 64))

    storyboard = _storyboard(
        [_shot(1, visual_kind=VisualKind.FIGURE, figure_id=tiny.id)]
    )

    visual = _RecordingVisual()
    log_path = job_dir / "events.log"
    bus = ProgressEventBus(log_path)
    renderer = _renderer(
        visual,
        extracted_images=[tiny],
        topic_titles={"topic_001": "T"},
        bus=bus,
    )
    await renderer.render(storyboard=storyboard, job_dir=job_dir)
    await bus.close()

    messages = _event_messages(log_path)
    assert visual.rendered == ["shot_0001"]
    assert any("too small" in msg for msg in messages)


# ---------------------------------------------------------------- telemetry


@pytest.mark.asyncio
async def test_emits_kind_telemetry_at_end_of_stage(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    figure = _write_figure(job_dir, "fig_a", (640, 400))

    shots = [
        _shot(1, visual_kind=VisualKind.TITLE_CARD),
        _shot(2, visual_kind=VisualKind.FIGURE, figure_id=figure.id),
        _shot(3, visual_kind=VisualKind.ILLUSTRATION),
        _shot(4, visual_kind=VisualKind.ILLUSTRATION),
    ]

    log_path = job_dir / "events.log"
    bus = ProgressEventBus(log_path)
    renderer = _renderer(
        _RecordingVisual(),
        extracted_images=[figure],
        topic_titles={"topic_001": "Topic"},
        bus=bus,
    )
    await renderer.render(storyboard=_storyboard(shots), job_dir=job_dir)
    await bus.close()

    messages = _event_messages(log_path)
    summary = [m for m in messages if "figure_shots=" in m]
    assert summary, messages
    assert "figure_shots=1" in summary[-1]
    assert "illustration_shots=2" in summary[-1]
    assert "title_cards=1" in summary[-1]


# ---------------------------------------------------------------- concurrency


@pytest.mark.asyncio
async def test_split_semaphores_honoured(tmp_path: Path) -> None:
    """SDXL cap stays bound to ``concurrency``; Pillow path runs in parallel."""

    job_dir = tmp_path / "job"
    figure = _write_figure(job_dir, "fig_big", (800, 600))

    # 8 SDXL + 8 figure + 0 title shots.
    shots = []
    for i in range(1, 9):
        shots.append(_shot(i, visual_kind=VisualKind.ILLUSTRATION))
    for i in range(9, 17):
        shots.append(_shot(i, visual_kind=VisualKind.FIGURE, figure_id=figure.id))

    bus = ProgressEventBus(job_dir / "events.log")
    visual = _RecordingVisual(render_delay=0.05)
    sdxl_cap = 2
    renderer = _renderer(
        visual,
        extracted_images=[figure],
        topic_titles={"topic_001": "Topic"},
        concurrency=sdxl_cap,
        bus=bus,
    )
    await renderer.render(storyboard=_storyboard(shots), job_dir=job_dir)
    await bus.close()

    # SDXL semaphore must have capped concurrency to ``sdxl_cap``.
    assert visual.max_concurrent <= sdxl_cap
    # All 8 figure shots produced files via the Pillow path.
    figure_outputs = [
        (job_dir / "images" / f"shot_{i:04d}.png").is_file() for i in range(9, 17)
    ]
    assert all(figure_outputs)
    # SDXL called exactly 8 times (the ILLUSTRATION shots).
    assert len(visual.rendered) == 8
