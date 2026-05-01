"""Per-shot image generation with bounded parallelism + filesystem-truthful resume.

Resume rule: a shot is "done" if and only if **both** ``images/<shot_id>.png``
exists on disk **and** ``index.json`` lists it. On entry we reconcile the index
against the filesystem so deleted files force a re-render and orphan files
(written by a crash before the index was persisted) get adopted.

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
from dataclasses import dataclass
from pathlib import Path

from ..errors import BookToAnimeError, ProviderError
from ..providers import ImageGenRequest, VisualProvider
from .artifacts import ImagesIndex, NarratorPersona, Shot, ShotImageRecord, Storyboard
from .events import ProgressEvent, ProgressEventBus, ProgressKind
from .stages import Stage

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRendererConfig:
    width: int = 1920
    height: int = 1080
    steps: int = 28
    guidance: float = 5.5
    concurrency: int = 2
    # Style key passed verbatim to ``VisualProvider.prepare`` so the persona
    # cache key matches what the visual provider's _STYLE_FRAGMENTS lookup
    # expects. Falling back to parsing the persona descriptor produced
    # cache keys like "shounen-bright_narrator_persona__<seed>.png".
    anime_style: str = "shounen-bright"


class ShotImageRenderer:
    def __init__(
        self,
        provider: VisualProvider,
        config: ImageRendererConfig,
        *,
        bus: ProgressEventBus,
    ) -> None:
        self._provider = provider
        self._config = config
        self._bus = bus

    async def render(
        self,
        *,
        storyboard: Storyboard,
        persona: NarratorPersona,
        job_dir: Path,
    ) -> tuple[ImagesIndex, Path]:
        """Render all shots and return ``(index, persona_reference_path)``.

        The persona reference path is returned so the orchestrator can copy
        it into the job directory and store a relative path in
        ``structured.json`` (avoids leaking the absolute model-cache path).
        """

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        index_path = images_dir / "index.json"

        completed = _reconcile_existing_index(index_path, storyboard, images_dir)
        existing_ids = {record.shot_id for record in completed}

        persona_reference = await self._ensure_persona_reference(persona, job_dir)

        semaphore = asyncio.Semaphore(max(1, self._config.concurrency))
        total = len(storyboard.shots)
        completed_count = len(existing_ids)
        new_records: dict[str, ShotImageRecord] = {}
        first_error: BookToAnimeError | None = None

        async def render_one(shot: Shot) -> None:
            nonlocal first_error
            if shot.id in existing_ids:
                return
            async with semaphore:
                if first_error is not None:
                    raise asyncio.CancelledError()
                try:
                    record = await self._render_one(shot, persona_reference, images_dir)
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

        if first_error is not None:
            raise first_error

        return ImagesIndex(items=completed), persona_reference

    async def _ensure_persona_reference(
        self,
        persona: NarratorPersona,
        job_dir: Path,
    ) -> Path:
        if persona.reference_image:
            absolute = (job_dir / persona.reference_image).resolve()
            if absolute.is_file():
                return absolute

        return await self._provider.prepare(
            anime_style=self._config.anime_style, narrator_seed=persona.seed
        )

    async def _render_one(
        self,
        shot: Shot,
        persona_reference: Path,
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
            reference_image=persona_reference if shot.use_persona_reference else None,
            reference_strength=shot.ip_adapter_strength,
        )
        try:
            generated = await self._provider.render(request, out_path)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"unhandled visual error on shot {shot.id}: {exc}") from exc

        rel_file = f"images/{out_path.name}"
        return ShotImageRecord(
            shot_id=shot.id,
            file=rel_file,
            seed=generated.seed,
            width=generated.width,
            height=generated.height,
        )

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
    storyboard_ids = {shot.id for shot in storyboard.shots}

    # Drop records whose file no longer exists.
    survived: list[ShotImageRecord] = []
    for record in loaded:
        if record.shot_id not in storyboard_ids:
            continue
        full_path = images_dir.parent / record.file
        if full_path.is_file():
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
                )
            )
    return survived
