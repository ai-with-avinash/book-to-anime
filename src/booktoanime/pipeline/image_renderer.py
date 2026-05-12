"""Per-shot image generation with bounded parallelism + filesystem-truthful resume.

Resume rule: a shot is "done" if and only if **both** ``images/<shot_id>.png``
exists on disk **and** ``index.json`` lists it. On entry we reconcile the index
against the filesystem so deleted files force a re-render and orphan files
(written by a crash before the index was persisted) get adopted.

Phase 3 dispatch
----------------

The renderer now branches on :class:`Shot.visual_kind`:

* :data:`VisualKind.FIGURE`     — composite the real extracted PDF figure
  via :mod:`pipeline.panel_composer` (Pillow, CPU-bound). Falls back to
  the SDXL path when the source figure is below the 256-pixel guard or
  the lookup is otherwise unrecoverable.
* :data:`VisualKind.TITLE_CARD` — render a centred title card via
  :mod:`pipeline.panel_composer`.
* :data:`VisualKind.ILLUSTRATION` — existing SDXL path, gated on the
  GPU-bound semaphore. Style anchor (when available) is forwarded to
  the provider as the IP-Adapter reference image.

Two semaphores enforce the contention split:

* ``_sdxl_semaphore`` — GPU-bound, profile-driven concurrency cap.
* ``_compose_semaphore`` — CPU-bound, ``min(8, cpu_count())`` cap.

On partial failure (one shot raises) we:

* let the in-flight tasks finish or be cancelled (and ``await`` them so we
  don't leak),
* persist every successfully-completed record we collected,
* re-raise the original :class:`BookToAnimeError` so the orchestrator can
  mark the stage as failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ProviderError, RenderError
from ..parsing.models import ExtractedImage
from ..providers import ImageGenRequest, VisualProvider
from .artifacts import (
    ImagesIndex,
    Shot,
    ShotImageRecord,
    Storyboard,
    VisualKind,
)
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .panel_composer import compose_figure_panel, compose_title_card
from .stages import Stage

# Avoid importing heavier siblings at module top; the loaders below pull them
# in lazily so the renderer stays cheap to import in unit tests that inject
# explicit state.

_logger = logging.getLogger(__name__)

# Minimum source-figure edge (in pixels) for the direct Pillow composite
# path. Below this we fall through to SDXL so we don't blow up a 64x64
# thumbnail into a blurry 1920x1080 panel.
_MIN_FIGURE_EDGE = 256


@dataclass(frozen=True)
class ImageRendererConfig:
    width: int = 1024
    height: int = 576
    steps: int = 12
    guidance: float = 5.5
    concurrency: int = 2
    # Panel-style key shared with the visual provider's prompt-fragment map
    # and the panel composer's palette / typography choices.
    panel_style: str = "clean-linework"


class ShotImageRenderer:
    """Run the per-shot image stage with dispatch on :class:`VisualKind`."""

    def __init__(
        self,
        provider: VisualProvider,
        config: ImageRendererConfig,
        *,
        bus: ProgressEventBus,
        extracted_images: list[ExtractedImage] | None = None,
        topic_titles: dict[str, str] | None = None,
        style_reference_path: Path | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._bus = bus
        self._extracted_images: dict[str, ExtractedImage] = {
            img.id: img for img in (extracted_images or [])
        }
        self._topic_titles: dict[str, str] = dict(topic_titles or {})
        self._style_reference_path = style_reference_path

    async def render(
        self,
        *,
        storyboard: Storyboard,
        job_dir: Path,
    ) -> ImagesIndex:
        """Render all shots and return the resulting :class:`ImagesIndex`."""

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        index_path = images_dir / "index.json"

        # Lazily hydrate per-job dispatch state from on-disk artifacts when the
        # caller didn't supply them. Keeps the orchestrator's call site
        # backwards-compatible while still letting unit tests inject explicit
        # state via the constructor.
        if not self._extracted_images:
            self._extracted_images = _load_extracted_images(job_dir)
        if not self._topic_titles:
            self._topic_titles = _load_topic_titles(job_dir)
        if self._style_reference_path is None:
            self._style_reference_path = _load_style_reference_path(job_dir)

        completed = _reconcile_existing_index(index_path, storyboard, images_dir)
        existing_ids = {record.shot_id for record in completed}

        sdxl_semaphore = asyncio.Semaphore(max(1, self._config.concurrency))
        compose_semaphore = asyncio.Semaphore(max(1, min(8, os.cpu_count() or 1)))
        total = len(storyboard.shots)
        completed_count = len(existing_ids)
        new_records: dict[str, ShotImageRecord] = {}
        first_error: BookToAnimeError | None = None

        async def render_one(shot: Shot) -> None:
            nonlocal first_error
            if shot.id in existing_ids:
                return
            try:
                record = await self._dispatch(
                    shot,
                    images_dir=images_dir,
                    job_dir=job_dir,
                    sdxl_semaphore=sdxl_semaphore,
                    compose_semaphore=compose_semaphore,
                    abort=lambda: first_error is not None,
                )
            except BookToAnimeError as exc:
                if first_error is None:
                    first_error = exc
                await self._bus.emit(
                    ProgressEvent(
                        kind=ProgressKind.SHOT_FAILED,
                        stage=Stage.IMAGES.value,
                        message=str(exc),
                        shot_id=shot.id,
                        user_message=exc.user_message,
                    )
                )
                raise
            new_records[shot.id] = record
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.SHOT_COMPLETED,
                    stage=Stage.IMAGES.value,
                    message=f"rendered {shot.id}",
                    shot_id=shot.id,
                )
            )

        tasks = [asyncio.create_task(render_one(shot)) for shot in storyboard.shots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for shot, result in zip(storyboard.shots, results, strict=True):
            if isinstance(result, BookToAnimeError):
                if first_error is None:
                    first_error = result
                continue
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                # Unexpected non-pipeline exception — surface as ProviderError
                # so the stage fails cleanly.
                if first_error is None:
                    first_error = ProviderError(
                        f"unexpected error rendering shot {shot.id}: {result}"
                    )
                continue

        for shot in storyboard.shots:
            if shot.id in new_records:
                completed.append(new_records[shot.id])
                completed_count += 1

        ImagesIndex(items=completed).save(index_path)
        if completed_count and total:
            await self._emit_progress_ratio(completed_count, total)

        if first_error is None:
            await self._emit_kind_telemetry(storyboard, completed)

        if first_error is not None:
            raise first_error

        return ImagesIndex(items=completed)

    # -------------------------------------------------------------- dispatch

    async def _dispatch(
        self,
        shot: Shot,
        *,
        images_dir: Path,
        job_dir: Path,
        sdxl_semaphore: asyncio.Semaphore,
        compose_semaphore: asyncio.Semaphore,
        abort: Callable[[], bool],
    ) -> ShotImageRecord:
        if shot.visual_kind == VisualKind.FIGURE:
            return await self._render_figure(
                shot,
                images_dir=images_dir,
                job_dir=job_dir,
                sdxl_semaphore=sdxl_semaphore,
                compose_semaphore=compose_semaphore,
                abort=abort,
            )
        if shot.visual_kind == VisualKind.TITLE_CARD:
            async with compose_semaphore:
                if abort():
                    raise asyncio.CancelledError()
                return await asyncio.to_thread(
                    self._render_title_card_sync, shot, images_dir
                )

        # ILLUSTRATION (default).
        async with sdxl_semaphore:
            if abort():
                raise asyncio.CancelledError()
            return await self._render_illustration(shot, images_dir)

    # ------------------------------------------------------------- variants

    async def _render_figure(
        self,
        shot: Shot,
        *,
        images_dir: Path,
        job_dir: Path,
        sdxl_semaphore: asyncio.Semaphore,
        compose_semaphore: asyncio.Semaphore,
        abort: Callable[[], bool],
    ) -> ShotImageRecord:
        if shot.figure_id is None:
            raise RenderError(
                f"shot {shot.id} marked FIGURE but has no figure_id"
            )
        figure = self._extracted_images.get(shot.figure_id)
        if figure is None:
            raise RenderError(
                f"shot {shot.id} references unknown figure_id {shot.figure_id!r}"
            )

        figure_path = job_dir / figure.file
        if not figure_path.is_file():
            raise RenderError(
                f"shot {shot.id} figure file missing on disk: {figure_path}"
            )

        # Tiny source figures don't survive a 1920x1080 blow-up; fall through
        # to SDXL with an INFO event so the operator can spot it in logs.
        if figure.width < _MIN_FIGURE_EDGE or figure.height < _MIN_FIGURE_EDGE:
            await self._bus.emit(
                ProgressEvent(
                    kind=ProgressKind.INFO,
                    stage=Stage.IMAGES.value,
                    message=(
                        f"figure {shot.figure_id} too small "
                        f"({figure.width}x{figure.height}); "
                        f"falling back to SDXL for {shot.id}"
                    ),
                )
            )
            async with sdxl_semaphore:
                if abort():
                    raise asyncio.CancelledError()
                return await self._render_illustration(shot, images_dir)

        async with compose_semaphore:
            if abort():
                raise asyncio.CancelledError()
            return await asyncio.to_thread(
                self._render_figure_sync,
                shot,
                figure_path,
                images_dir,
            )

    def _render_figure_sync(
        self,
        shot: Shot,
        figure_path: Path,
        images_dir: Path,
    ) -> ShotImageRecord:
        out_path = images_dir / f"{shot.id}.png"
        caption = (shot.narration_text or shot.image_prompt or "").strip()
        title = self._topic_titles.get(shot.topic_id, shot.topic_id)
        try:
            panel = compose_figure_panel(
                figure_path=figure_path,
                caption=caption,
                title=title,
                panel_style=self._config.panel_style,
                target_size=(self._config.width, self._config.height),
            )
            panel.save(out_path)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(
                f"failed to compose figure panel for shot {shot.id}: {exc}"
            ) from exc
        return ShotImageRecord(
            shot_id=shot.id,
            file=f"images/{out_path.name}",
            seed=shot.seed,
            width=self._config.width,
            height=self._config.height,
            visual_kind=shot.visual_kind,
            figure_id=shot.figure_id,
        )

    def _render_title_card_sync(
        self,
        shot: Shot,
        images_dir: Path,
    ) -> ShotImageRecord:
        out_path = images_dir / f"{shot.id}.png"
        title = self._topic_titles.get(shot.topic_id, shot.topic_id)
        subtitle = (shot.narration_text or "").strip()
        try:
            card = compose_title_card(
                title=title,
                subtitle=subtitle,
                panel_style=self._config.panel_style,
                target_size=(self._config.width, self._config.height),
            )
            card.save(out_path)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(
                f"failed to compose title card for shot {shot.id}: {exc}"
            ) from exc
        return ShotImageRecord(
            shot_id=shot.id,
            file=f"images/{out_path.name}",
            seed=shot.seed,
            width=self._config.width,
            height=self._config.height,
            visual_kind=shot.visual_kind,
            figure_id=shot.figure_id,
        )

    async def _render_illustration(
        self,
        shot: Shot,
        images_dir: Path,
    ) -> ShotImageRecord:
        out_path = images_dir / f"{shot.id}.png"
        request = ImageGenRequest(
            prompt=shot.image_prompt,
            negative_prompt=shot.negative_prompt,
            width=self._config.width,
            height=self._config.height,
            seed=shot.seed,
            steps=self._config.steps,
            guidance=self._config.guidance,
            reference_image=self._style_reference_path,
            reference_strength=shot.ip_adapter_strength,
        )
        try:
            generated = await self._provider.render(request, out_path)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"unhandled visual error on shot {shot.id}: {exc}"
            ) from exc

        return ShotImageRecord(
            shot_id=shot.id,
            file=f"images/{out_path.name}",
            seed=generated.seed,
            width=generated.width,
            height=generated.height,
            visual_kind=shot.visual_kind,
            figure_id=shot.figure_id,
        )

    # ----------------------------------------------------------- telemetry

    async def _emit_progress_ratio(self, done: int, total: int) -> None:
        ratio = (done / total) if total else 1.0
        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.IMAGES.value,
                message=f"images: {done}/{total}",
                progress=ratio,
            )
        )

    async def _emit_kind_telemetry(
        self,
        storyboard: Storyboard,
        completed: list[ShotImageRecord],
    ) -> None:
        """Emit per-kind shot counts at the end of the stage.

        Counts come from ``completed`` (what we actually wrote, including any
        shots that fell through from FIGURE to ILLUSTRATION via the small-edge
        guard) so operators can spot a job where every figure got bumped to
        SDXL fallback.
        """

        by_kind = {kind: 0 for kind in VisualKind}
        for record in completed:
            by_kind[record.visual_kind] = by_kind.get(record.visual_kind, 0) + 1

        await self._bus.emit(
            ProgressEvent(
                kind=ProgressKind.INFO,
                stage=Stage.IMAGES.value,
                message=(
                    f"figure_shots={by_kind[VisualKind.FIGURE]} "
                    f"illustration_shots={by_kind[VisualKind.ILLUSTRATION]} "
                    f"title_cards={by_kind[VisualKind.TITLE_CARD]}"
                ),
            )
        )
        _logger.info(
            "image stage complete: figure=%d illustration=%d title_card=%d (total shots=%d)",
            by_kind[VisualKind.FIGURE],
            by_kind[VisualKind.ILLUSTRATION],
            by_kind[VisualKind.TITLE_CARD],
            len(storyboard.shots),
        )


def _reconcile_existing_index(
    index_path: Path,
    storyboard: Storyboard,
    images_dir: Path,
) -> list[ShotImageRecord]:
    """Load the index, drop entries whose file is missing, adopt orphans on disk.

    Orphan adoption is best-effort: a file that's missing from the index gets
    a synthetic record built from the storyboard shot (so its seed/dimensions
    line up). If the storyboard doesn't list the file, we leave it alone.
    """

    if not index_path.is_file():
        loaded: list[ShotImageRecord] = []
    else:
        try:
            loaded = list(ImagesIndex.from_path(index_path).items)
        except Exception as exc:
            _logger.warning(
                "ignoring unreadable images index (will rebuild from filesystem): %s", exc
            )
            loaded = []

    by_id = {record.shot_id: record for record in loaded}
    shots_by_id = {shot.id: shot for shot in storyboard.shots}

    # Drop records whose file no longer exists OR whose visual_kind / figure_id
    # no longer match the storyboard (phase 3 dispatch reads visual_kind, so a
    # stale value would render the wrong panel kind on resume).
    survived: list[ShotImageRecord] = []
    for record in loaded:
        shot = shots_by_id.get(record.shot_id)
        if shot is None:
            continue
        full_path = images_dir.parent / record.file
        if not full_path.is_file():
            continue
        if record.visual_kind != shot.visual_kind:
            _logger.info(
                "invalidating shot %s: visual_kind changed %s -> %s",
                record.shot_id,
                record.visual_kind,
                shot.visual_kind,
            )
            continue
        if record.figure_id != shot.figure_id:
            _logger.info(
                "invalidating shot %s: figure_id changed %r -> %r",
                record.shot_id,
                record.figure_id,
                shot.figure_id,
            )
            continue
        survived.append(record)

    # Adopt files-on-disk that the storyboard knows about but the index doesn't.
    survived_ids = {record.shot_id for record in survived}
    for shot in storyboard.shots:
        if shot.id in survived_ids or shot.id in by_id:
            continue
        on_disk = images_dir / f"{shot.id}.png"
        if on_disk.is_file():
            survived.append(
                ShotImageRecord(
                    shot_id=shot.id,
                    file=f"images/{on_disk.name}",
                    seed=shot.seed,
                    width=0,
                    height=0,
                    visual_kind=shot.visual_kind,
                    figure_id=shot.figure_id,
                )
            )
    return survived


def _load_extracted_images(job_dir: Path) -> dict[str, ExtractedImage]:
    """Load all :class:`ExtractedImage` records from ``<job_dir>/extracted/parsed.json``.

    Empty on missing / unreadable file — callers handle FIGURE shots that
    can't resolve a figure_id by raising :class:`RenderError`.
    """

    parsed_path = job_dir / "extracted" / "parsed.json"
    if not parsed_path.is_file():
        return {}
    try:
        from ..parsing.models import ParsedDocument  # local import to keep top light

        parsed = ParsedDocument.from_path(parsed_path)
    except Exception as exc:
        _logger.warning("could not load parsed.json for figure dispatch: %s", exc)
        return {}
    out: dict[str, ExtractedImage] = {}
    for page in parsed.pages:
        for img in page.images:
            out[img.id] = img
    return out


def _load_topic_titles(job_dir: Path) -> dict[str, str]:
    """Load topic-id → topic-title map from ``<job_dir>/structured.json``."""

    structured_path = job_dir / "structured.json"
    if not structured_path.is_file():
        return {}
    try:
        from .artifacts import StructuredDocument

        doc = StructuredDocument.from_path(structured_path)
    except Exception as exc:
        _logger.warning("could not load structured.json for title cards: %s", exc)
        return {}
    return {topic.id: topic.title for topic in doc.topics}


def _load_style_reference_path(job_dir: Path) -> Path | None:
    """Resolve the IP-Adapter style anchor produced by ``STYLE_SEEDING``.

    Returns ``None`` if the manifest is missing, the artifact pointer is
    absent, or the referenced file no longer exists on disk.
    """

    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        from .manifest import JobManifest

        manifest = JobManifest.from_path(manifest_path)
    except Exception as exc:
        _logger.warning("could not load manifest for style reference: %s", exc)
        return None
    style_ref = manifest.artifacts.style_reference
    if style_ref is None:
        return None
    candidate = job_dir / style_ref.file
    return candidate if candidate.is_file() else None
