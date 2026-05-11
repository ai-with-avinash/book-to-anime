"""Unit tests for the per-job style-anchor seeder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from booktoanime.pipeline.artifacts import StyleReference
from booktoanime.pipeline.style_seeder import (
    StyleSeeder,
    StyleSeederConfig,
)
from booktoanime.providers.base import GeneratedImage, VisualProvider


class _RecordingVisual(VisualProvider):
    name = "recording-visual"

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.render_calls = 0

    async def prepare(self, *, panel_style: str, narrator_seed: int) -> Path:
        raise NotImplementedError

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        self.requests.append(request)
        self.render_calls += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), (1, 2, 3)).save(out_path)
        return GeneratedImage(
            path=out_path,
            seed=request.seed,
            width=request.width,
            height=request.height,
        )

    async def close(self) -> None:
        return None


# --------------------------------------------------------------- prompt build


@pytest.mark.asyncio
async def test_seed_builds_prompt_with_anchor_base_and_style_fragment(
    tmp_path: Path,
) -> None:
    visual = _RecordingVisual()
    seeder = StyleSeeder(
        StyleSeederConfig(panel_style="chalkboard-sketch", seed=7),
        visual_provider=visual,
    )

    reference = await seeder.seed(tmp_path)

    assert visual.render_calls == 1
    sent = visual.requests[0]
    assert "abstract style reference" in sent.prompt
    # Style fragment for the chosen panel style is appended.
    assert "chalkboard" in sent.prompt.lower()
    assert sent.seed == 7
    # The seeder reference points at the on-disk file under <job>/style/.
    assert reference.file.startswith("style/")
    assert (tmp_path / reference.file).is_file()


@pytest.mark.asyncio
async def test_seed_falls_back_to_literal_when_style_unknown(
    tmp_path: Path,
) -> None:
    visual = _RecordingVisual()
    seeder = StyleSeeder(
        StyleSeederConfig(panel_style="not-a-real-style", seed=42),
        visual_provider=visual,
    )

    await seeder.seed(tmp_path)
    assert "not-a-real-style" in visual.requests[0].prompt


# --------------------------------------------------------------- idempotence


@pytest.mark.asyncio
async def test_seed_is_idempotent_when_file_already_exists(tmp_path: Path) -> None:
    visual = _RecordingVisual()
    seeder = StyleSeeder(
        StyleSeederConfig(panel_style="clean-linework", seed=42),
        visual_provider=visual,
    )

    first = await seeder.seed(tmp_path)
    assert visual.render_calls == 1

    # Second invocation must short-circuit because the file is on disk.
    second = await seeder.seed(tmp_path)
    assert visual.render_calls == 1, "expected zero additional renders"
    assert second == first


# --------------------------------------------------------------- path stability


@pytest.mark.asyncio
async def test_seed_file_path_stable_across_runs_for_same_cache_key(
    tmp_path: Path,
) -> None:
    visual_a = _RecordingVisual()
    visual_b = _RecordingVisual()

    seeder_a = StyleSeeder(
        StyleSeederConfig(panel_style="watercolor-technical", seed=42),
        visual_provider=visual_a,
    )
    ref_a = await seeder_a.seed(tmp_path)

    # Wipe the on-disk file then re-run with a fresh provider. Path stays put.
    (tmp_path / ref_a.file).unlink()

    seeder_b = StyleSeeder(
        StyleSeederConfig(panel_style="watercolor-technical", seed=42),
        visual_provider=visual_b,
    )
    ref_b = await seeder_b.seed(tmp_path)

    assert ref_a.file == ref_b.file


@pytest.mark.asyncio
async def test_seed_filename_differs_per_seed_or_style(tmp_path: Path) -> None:
    visual = _RecordingVisual()
    seeder_one = StyleSeeder(
        StyleSeederConfig(panel_style="clean-linework", seed=1),
        visual_provider=visual,
    )
    seeder_two = StyleSeeder(
        StyleSeederConfig(panel_style="clean-linework", seed=2),
        visual_provider=visual,
    )
    seeder_three = StyleSeeder(
        StyleSeederConfig(panel_style="flat-vector-infographic", seed=1),
        visual_provider=visual,
    )

    ref1 = await seeder_one.seed(tmp_path)
    ref2 = await seeder_two.seed(tmp_path)
    ref3 = await seeder_three.seed(tmp_path)

    assert len({ref1.file, ref2.file, ref3.file}) == 3


# --------------------------------------------------------------- json round-trip


def test_style_reference_round_trips_through_json() -> None:
    ref = StyleReference(
        file="style/style__clean-linework__42.png",
        panel_style="clean-linework",
        seed=42,
    )
    payload = ref.model_dump_json()
    revived = StyleReference.model_validate(json.loads(payload))
    assert revived == ref
