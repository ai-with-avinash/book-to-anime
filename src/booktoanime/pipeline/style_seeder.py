"""Per-job style-anchor seeder.

The seeder renders a single abstract style reference image once per
``(panel_style, seed)`` pair and writes it under ``<job_dir>/style/``. Phase
3's renderer feeds the resulting image to IP-Adapter for every
``VisualKind.ILLUSTRATION`` shot so SDXL fallback renders stay visually
consistent across the whole job.

Design notes:

* The seeder owns *no* anime character / narrator concept. The base prompt
  comes from :mod:`pipeline.styles` and deliberately says "no subject".
* :meth:`StyleSeeder.seed` is idempotent. If the on-disk file exists with the
  configured ``(panel_style, seed)`` cache key, we short-circuit and skip the
  provider call. The renderer is also responsible for skipping when the
  manifest already records a matching reference.
* :class:`StyleReference` is a Pydantic model with ``extra="forbid"`` and
  ``file`` constrained to :data:`artifacts.JobRelPath` so a tampered
  ``manifest.json`` cannot point downstream consumers at arbitrary paths.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..providers.base import ImageGenRequest, VisualProvider
from .artifacts import StyleReference
from .styles import STYLE_ANCHOR_BASE, STYLE_FRAGMENTS

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class StyleSeederConfig(BaseModel):
    """Config for :class:`StyleSeeder`.

    ``frozen=True`` keeps the config hashable + safe to share across coroutines.
    The default seed mirrors the legacy narrator-seeder constant so existing
    jobs running on stable seeds get the same cache key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_style: str
    seed: int = 42


class StyleSeeder:
    """Render (or load) the per-job style-anchor reference image."""

    def __init__(
        self,
        config: StyleSeederConfig,
        visual_provider: VisualProvider,
    ) -> None:
        self._config = config
        self._provider = visual_provider
        self._render_lock = asyncio.Lock()

    async def seed(self, job_dir: Path) -> StyleReference:
        """Produce ``<job_dir>/style/style__<panel_style>__<seed>.png``.

        If the file already exists at the cache-key path, no provider call is
        issued. Returns a :class:`StyleReference` with a job-relative path so
        it round-trips into the manifest cleanly.
        """

        style_dir = job_dir / "style"
        filename = _style_filename(self._config.panel_style, self._config.seed)
        rel_file = f"style/{filename}"
        out_path = style_dir / filename

        if out_path.is_file():
            return StyleReference(
                file=rel_file,
                panel_style=self._config.panel_style,
                seed=self._config.seed,
            )

        # Serialize concurrent calls inside one process so a parallel resume
        # doesn't render the same anchor twice.
        async with self._render_lock:
            if out_path.is_file():
                return StyleReference(
                    file=rel_file,
                    panel_style=self._config.panel_style,
                    seed=self._config.seed,
                )

            await asyncio.to_thread(style_dir.mkdir, parents=True, exist_ok=True)

            fragment = STYLE_FRAGMENTS.get(self._config.panel_style, self._config.panel_style)
            prompt = f"{STYLE_ANCHOR_BASE}, {fragment}"

            request = ImageGenRequest(
                prompt=prompt,
                negative_prompt=None,
                width=1024,
                height=1024,
                seed=self._config.seed,
                steps=28,
                guidance=5.5,
                reference_image=None,
                reference_strength=0.0,
            )
            await self._provider.render(request, out_path)

        return StyleReference(
            file=rel_file,
            panel_style=self._config.panel_style,
            seed=self._config.seed,
        )


def _style_filename(panel_style: str, seed: int) -> str:
    safe = _FILENAME_SAFE.sub("_", panel_style.strip()) or "style"
    return f"style__{safe}__{seed}.png"


__all__ = ["StyleReference", "StyleSeeder", "StyleSeederConfig"]
