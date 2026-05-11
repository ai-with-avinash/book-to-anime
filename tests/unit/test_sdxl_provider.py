"""Tests for the SDXL+IP-Adapter visual provider using a stub pipeline.

The real diffusers/torch stack is gated behind ``[visual]`` install extras and
is not exercised here; the provider's logic — caching personas, normalizing
dimensions, hopping calls to a thread, applying IP-Adapter scale, etc. — is
covered by injecting a deterministic stub pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from booktoanime.errors import ProviderError
from booktoanime.providers import registry
from booktoanime.providers.base import ImageGenRequest
from booktoanime.providers.visual.sdxl_diffusers import (
    SDXLDiffusersProvider,
    _round_to_multiple,
    _style_reference_filename,
)


@dataclass
class _StubResult:
    images: list[Image.Image]


@dataclass
class _StubPipeline:
    """Records every call and returns a solid-color Pillow image."""

    fill: tuple[int, int, int] = (180, 100, 60)
    calls: list[dict[str, Any]] = field(default_factory=list)
    scale_calls: list[float] = field(default_factory=list)
    ip_adapter_loads: list[tuple[str, str, str]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> _StubResult:
        self.calls.append(kwargs)
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        return _StubResult(images=[Image.new("RGB", (width, height), self.fill)])

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.scale_calls.append(float(scale))

    def load_ip_adapter(
        self,
        repo: str,
        *,
        subfolder: str = "",
        weight_name: str = "",
    ) -> None:
        self.ip_adapter_loads.append((repo, subfolder, weight_name))


def _provider(
    *,
    persona_dir: Path,
    pipeline: _StubPipeline | None = None,
) -> SDXLDiffusersProvider:
    return SDXLDiffusersProvider(
        default_width=128,
        default_height=128,
        default_steps=4,
        default_guidance=4.0,
        persona_dir=persona_dir,
        pipeline=pipeline or _StubPipeline(),
    )


# --------------------------------------------------------------- prepare()


@pytest.mark.asyncio
async def test_prepare_writes_persona_and_caches_path(tmp_path: Path) -> None:
    pipeline = _StubPipeline()
    provider = _provider(persona_dir=tmp_path, pipeline=pipeline)

    path1 = await provider.prepare(panel_style="shounen-bright", narrator_seed=42)
    assert path1.is_file()
    assert path1.parent == tmp_path
    assert path1.name == "shounen-bright__42.png"
    assert len(pipeline.calls) == 1

    # Calling again with the same key reuses the cached file.
    path2 = await provider.prepare(panel_style="shounen-bright", narrator_seed=42)
    assert path2 == path1
    assert len(pipeline.calls) == 1, "second prepare() should not re-render"


@pytest.mark.asyncio
async def test_prepare_unknown_style_uses_literal_value(tmp_path: Path) -> None:
    pipeline = _StubPipeline()
    provider = _provider(persona_dir=tmp_path, pipeline=pipeline)

    await provider.prepare(panel_style="vaporwave", narrator_seed=7)

    prompt = pipeline.calls[0]["prompt"]
    assert "vaporwave" in prompt


@pytest.mark.asyncio
async def test_prepare_style_reference_filename_safe_for_special_chars(tmp_path: Path) -> None:
    provider = _provider(persona_dir=tmp_path)
    path = await provider.prepare(panel_style="some/weird style!", narrator_seed=1)
    assert path.name == "some_weird_style___1.png"


# --------------------------------------------------------------- render()


@pytest.mark.asyncio
async def test_render_writes_png_with_normalized_dimensions(tmp_path: Path) -> None:
    pipeline = _StubPipeline()
    provider = _provider(persona_dir=tmp_path / "personas", pipeline=pipeline)

    out_path = tmp_path / "shot_0001.png"
    request = ImageGenRequest(
        prompt="anime cat sitting on a roof",
        width=129,  # not a multiple of 8 -> should round down to 128
        height=257,  # rounds to 256
        seed=11,
        steps=3,
        guidance=4.5,
    )
    result = await provider.render(request, out_path)

    assert result.path == out_path
    assert out_path.is_file()
    assert (result.width, result.height) == (128, 256)
    assert result.seed == 11

    call = pipeline.calls[0]
    assert call["width"] == 128
    assert call["height"] == 256
    assert call["num_inference_steps"] == 3
    assert call["guidance_scale"] == 4.5
    # No reference image → ip_adapter_image kwarg dropped entirely so the
    # UNet does not require image_embeds when IP-Adapter is unloaded.
    assert "ip_adapter_image" not in call


@pytest.mark.asyncio
async def test_render_uses_persona_reference_and_sets_scale(tmp_path: Path) -> None:
    pipeline = _StubPipeline()
    persona_dir = tmp_path / "personas"
    provider = _provider(persona_dir=persona_dir, pipeline=pipeline)

    persona_path = await provider.prepare(panel_style="shounen-bright", narrator_seed=42)

    request = ImageGenRequest(
        prompt="hero in a school hallway",
        width=128,
        height=128,
        seed=99,
        steps=3,
        guidance=4.5,
        reference_image=persona_path,
        reference_strength=0.7,
    )
    await provider.render(request, tmp_path / "shot_0002.png")

    # Last call corresponds to the render() invocation.
    last_call = pipeline.calls[-1]
    assert last_call["ip_adapter_image"] is not None
    assert isinstance(last_call["ip_adapter_image"], Image.Image)
    # First call (the persona render) didn't have a reference, so only the
    # render() call should have driven set_ip_adapter_scale.
    assert pipeline.scale_calls == [pytest.approx(0.7)]


@pytest.mark.asyncio
async def test_render_missing_reference_raises(tmp_path: Path) -> None:
    provider = _provider(persona_dir=tmp_path)
    request = ImageGenRequest(
        prompt="ok",
        width=128,
        height=128,
        seed=1,
        steps=2,
        guidance=4.0,
        reference_image=tmp_path / "does_not_exist.png",
    )
    with pytest.raises(ProviderError, match="reference image not found"):
        await provider.render(request, tmp_path / "out.png")


@pytest.mark.asyncio
async def test_render_empty_prompt_raises(tmp_path: Path) -> None:
    provider = _provider(persona_dir=tmp_path)
    request = ImageGenRequest(
        prompt="   ",
        width=128,
        height=128,
        seed=1,
        steps=2,
        guidance=4.0,
    )
    with pytest.raises(ProviderError, match="empty prompt"):
        await provider.render(request, tmp_path / "out.png")


@pytest.mark.asyncio
async def test_render_pipeline_returning_no_images_raises(tmp_path: Path) -> None:
    class _EmptyPipeline:
        def __call__(self, **_: Any) -> _StubResult:
            return _StubResult(images=[])

        def set_ip_adapter_scale(self, scale: float) -> None:
            return None

    provider = _provider(persona_dir=tmp_path, pipeline=_EmptyPipeline())
    request = ImageGenRequest(
        prompt="hi",
        width=128,
        height=128,
        seed=1,
        steps=2,
        guidance=4.0,
    )
    with pytest.raises(ProviderError, match="no images"):
        await provider.render(request, tmp_path / "out.png")


# --------------------------------------------------------------- lazy load


@pytest.mark.asyncio
async def test_pipeline_factory_invoked_only_on_first_use(tmp_path: Path) -> None:
    stub = _StubPipeline()
    factory_calls = 0

    def factory() -> _StubPipeline:
        nonlocal factory_calls
        factory_calls += 1
        return stub

    provider = SDXLDiffusersProvider(
        default_width=128,
        default_height=128,
        default_steps=2,
        default_guidance=4.0,
        persona_dir=tmp_path,
        pipeline_factory=factory,
    )

    assert factory_calls == 0
    await provider.prepare(panel_style="shounen-bright", narrator_seed=1)
    assert factory_calls == 1

    await provider.render(
        ImageGenRequest(
            prompt="hi",
            width=128,
            height=128,
            seed=2,
            steps=2,
            guidance=4.0,
        ),
        tmp_path / "out.png",
    )
    assert factory_calls == 1


@pytest.mark.asyncio
async def test_pipeline_import_error_surfaces_install_hint(tmp_path: Path) -> None:
    def factory() -> _StubPipeline:
        raise ImportError("no diffusers")

    provider = SDXLDiffusersProvider(
        default_width=128,
        default_height=128,
        default_steps=2,
        default_guidance=4.0,
        persona_dir=tmp_path,
        pipeline_factory=factory,
    )
    with pytest.raises(ImportError, match=r"booktoanime\[visual\]"):
        await provider.prepare(panel_style="shounen-bright", narrator_seed=1)


# --------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("value", "multiple", "expected"),
    [
        (1024, 8, 1024),
        (129, 8, 128),
        (1, 8, 8),
        (0, 8, 8),
        (-5, 8, 8),
    ],
)
def test_round_to_multiple(value: int, multiple: int, expected: int) -> None:
    assert _round_to_multiple(value, multiple) == expected


def test_style_reference_filename_strips_unsafe_chars() -> None:
    assert _style_reference_filename("shounen-bright", 42) == "shounen-bright__42.png"
    assert _style_reference_filename("a/b c", 0) == "a_b_c__0.png"
    assert _style_reference_filename("", 1) == "style__1.png"


# --------------------------------------------------------------- registry


def test_self_registers_via_visual_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKTOANIME_DATA_DIR", str(Path("/tmp/booktoanime_test")))
    config = {
        "active": "sdxl_diffusers",
        "sdxl_diffusers": {
            "checkpoint": "stub/checkpoint",
            "width": 128,
            "height": 128,
            "steps": 2,
            "guidance": 4.0,
            "ip_adapter_repo": None,
        },
    }
    provider = registry.build_visual_provider(config)
    assert provider.name == "sdxl_diffusers"
