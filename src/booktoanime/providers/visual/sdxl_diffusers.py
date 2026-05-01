"""Default visual provider: SDXL via Hugging Face ``diffusers`` + IP-Adapter.

Design choices:

* The torch/diffusers stack is **lazy-loaded** on first :meth:`prepare` /
  :meth:`render` call. Building the provider is cheap; the user pays the
  multi-GB model download only when a job actually starts.
* The diffusers ``StableDiffusionXLPipeline`` is sync. Calls hop to a
  thread pool via :func:`asyncio.to_thread` so they never block the event
  loop.
* Character consistency: a single persona reference image is rendered once
  (deterministic seed) and reused as the IP-Adapter reference for every shot.
  Persona images are cached on disk under the ``persona_dir`` so resuming or
  re-running a job doesn't repaint the narrator.
* The actual diffusers pipeline is hidden behind a small
  :class:`SDXLPipelineLike` Protocol so tests can inject a stub that returns
  Pillow images without importing torch.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ...errors import ProviderError
from ..base import GeneratedImage, ImageGenRequest, VisualProvider
from ..registry import register_visual_provider

_logger = logging.getLogger(__name__)

# SDXL pipelines require dimensions divisible by 8.
_DIM_MULTIPLE = 8

# Anime-style prompt fragments. The user picks a style name in config; the
# storyboard prompt is concatenated with the matching fragment so every shot
# carries the same baseline aesthetic.
_STYLE_FRAGMENTS: Mapping[str, str] = {
    "shounen-bright": (
        "shounen anime, bright saturated colors, clean line art, dynamic composition, "
        "studio anime production, high detail"
    ),
    "shoujo-soft": (
        "shoujo anime, soft pastel palette, sparkly highlights, gentle expressions, "
        "watercolor-like backgrounds"
    ),
    "seinen-muted": (
        "seinen anime, muted earthy palette, cinematic lighting, mature realistic "
        "proportions, painterly textures"
    ),
    "chibi": (
        "chibi anime, oversized heads, small bodies, cute exaggerated expressions, "
        "playful flat shading"
    ),
}

_NEGATIVE_PROMPT_BASELINE = (
    "low quality, blurry, distorted, watermark, text, signature, deformed hands, "
    "bad anatomy, jpeg artifacts"
)

_PERSONA_BASE_PROMPT = (
    "anime narrator character portrait, single subject centered, "
    "looking at camera, neutral background, full upper body framing"
)


@dataclass(frozen=True)
class _PipelineCallArgs:
    """Just the parameters we feed to the underlying pipeline call."""

    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    seed: int
    steps: int
    guidance: float
    ip_adapter_image: Image.Image | None
    ip_adapter_scale: float


@runtime_checkable
class SDXLPipelineLike(Protocol):
    """Minimal duck-typed view of the diffusers SDXL pipeline.

    Real instances are ``StableDiffusionXLPipeline``. Tests pass a stub that
    accepts the same kwargs and returns an object whose ``.images`` is a list
    of Pillow images.
    """

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
        ip_adapter_image: Image.Image | None = ...,
    ) -> Any:
        ...

    def set_ip_adapter_scale(self, scale: float) -> None:
        ...


class SDXLDiffusersProvider(VisualProvider):
    name = "sdxl_diffusers"

    def __init__(
        self,
        *,
        checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0",
        ip_adapter_repo: str | None = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        ip_adapter_weights: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
        default_width: int = 1024,
        default_height: int = 1024,
        default_steps: int = 28,
        default_guidance: float = 5.5,
        device: str | None = None,
        persona_dir: Path | None = None,
        pipeline: SDXLPipelineLike | None = None,
        pipeline_factory: Callable[[], SDXLPipelineLike] | None = None,
    ) -> None:
        """Construct an SDXL+IP-Adapter visual provider.

        Args:
            checkpoint: HF hub id for the SDXL base/anime checkpoint.
            ip_adapter_repo: HF repo for IP-Adapter weights. Set to ``None``
                to disable IP-Adapter entirely (provider will fall back to
                prompt-only consistency).
            ip_adapter_subfolder: Subfolder inside the IP-Adapter repo.
            ip_adapter_weights: Filename of the IP-Adapter weights to load.
            default_width / default_height: Used by :meth:`prepare` and as
                fallback when a request omits dimensions.
            default_steps / default_guidance: Used as fallback shot defaults.
            device: ``"cuda"`` / ``"mps"`` / ``"cpu"`` / ``None`` (auto).
            persona_dir: Where persona reference images are cached. Defaults
                to ``$BOOKTOANIME_DATA_DIR/personas`` (or a tmpdir for tests).
            pipeline: An already-loaded pipeline. Skips the heavy factory call.
            pipeline_factory: Zero-arg callable that returns a pipeline. The
                default factory loads SDXL + IP-Adapter via diffusers.
        """

        self._checkpoint = checkpoint
        self._ip_adapter_repo = ip_adapter_repo
        self._ip_adapter_subfolder = ip_adapter_subfolder
        self._ip_adapter_weights = ip_adapter_weights
        self._default_width = _round_to_multiple(default_width, _DIM_MULTIPLE)
        self._default_height = _round_to_multiple(default_height, _DIM_MULTIPLE)
        self._default_steps = max(1, default_steps)
        self._default_guidance = float(default_guidance)
        self._device = device
        self._persona_dir = persona_dir or _default_persona_dir()
        self._pipeline: SDXLPipelineLike | None = pipeline
        self._pipeline_factory = pipeline_factory
        self._pipeline_lock = asyncio.Lock()

    # ------------------------------------------------------ VisualProvider API

    async def prepare(self, *, anime_style: str, narrator_seed: int) -> Path:
        """Generate (or load cached) narrator-persona reference image.

        Cache key: ``{anime_style}__{narrator_seed}.png``. Reusing the same
        seed across the entire video gives the IP-Adapter a stable reference.
        """

        self._persona_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._persona_dir / _persona_filename(anime_style, narrator_seed)
        if cache_path.is_file():
            return cache_path

        style_fragment = _STYLE_FRAGMENTS.get(anime_style)
        if style_fragment is None:
            _logger.warning(
                "unknown anime_style %r; using literal value as prompt fragment", anime_style
            )
            style_fragment = anime_style

        prompt = f"{_PERSONA_BASE_PROMPT}, {style_fragment}"
        request = ImageGenRequest(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT_BASELINE,
            width=self._default_width,
            height=self._default_height,
            seed=narrator_seed,
            steps=self._default_steps,
            guidance=self._default_guidance,
            reference_image=None,
            reference_strength=0.0,
        )
        await self._render_to(request, cache_path, persona_image=None)
        return cache_path

    async def render(self, request: ImageGenRequest, out_path: Path) -> GeneratedImage:
        persona = self._load_reference(request.reference_image)
        return await self._render_to(request, out_path, persona_image=persona)

    async def close(self) -> None:
        # Diffusers pipelines do not expose a portable async close; rely on
        # GC. We deliberately keep the loaded pipeline around for the lifetime
        # of the provider so subsequent jobs can reuse it.
        return None

    # ------------------------------------------------------ internals

    async def _render_to(
        self,
        request: ImageGenRequest,
        out_path: Path,
        *,
        persona_image: Image.Image | None,
    ) -> GeneratedImage:
        if not request.prompt.strip():
            raise ProviderError("image generation request has empty prompt")

        pipeline = await self._get_pipeline()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        width = _round_to_multiple(request.width or self._default_width, _DIM_MULTIPLE)
        height = _round_to_multiple(request.height or self._default_height, _DIM_MULTIPLE)
        steps = max(1, request.steps or self._default_steps)
        guidance = float(request.guidance or self._default_guidance)

        call_args = _PipelineCallArgs(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or _NEGATIVE_PROMPT_BASELINE,
            width=width,
            height=height,
            seed=request.seed,
            steps=steps,
            guidance=guidance,
            ip_adapter_image=persona_image,
            ip_adapter_scale=float(request.reference_strength or 0.0),
        )

        try:
            image = await asyncio.to_thread(_invoke_pipeline, pipeline, call_args)
        except Exception as exc:
            raise ProviderError(f"SDXL render failed: {exc}") from exc

        # Pillow's save call is small but I/O; keep off the loop.
        await asyncio.to_thread(image.save, str(out_path), "PNG")

        return GeneratedImage(
            path=out_path,
            seed=request.seed,
            width=image.width,
            height=image.height,
        )

    @staticmethod
    def _load_reference(path: Path | None) -> Image.Image | None:
        if path is None:
            return None
        if not path.is_file():
            raise ProviderError(f"reference image not found: {path}")
        return Image.open(path).convert("RGB")

    async def _get_pipeline(self) -> SDXLPipelineLike:
        if self._pipeline is not None:
            return self._pipeline

        async with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline

            factory = self._pipeline_factory or self._default_pipeline_factory
            try:
                pipeline = await asyncio.to_thread(factory)
            except ImportError as exc:
                raise ImportError(
                    "the 'diffusers' / 'torch' stack is required for the SDXL provider. "
                    "Install with `pip install booktoanime[visual]`."
                ) from exc
            except Exception as exc:
                raise ProviderError(f"SDXL pipeline init failed: {exc}") from exc

            self._pipeline = pipeline
            return pipeline

    def _default_pipeline_factory(self) -> SDXLPipelineLike:
        """Build the real diffusers pipeline. Imports happen here so missing
        ``torch``/``diffusers`` raise :class:`ImportError` lazily.
        """

        import torch
        from diffusers import StableDiffusionXLPipeline

        device = self._device or _auto_device(torch)
        dtype = torch.float16 if device != "cpu" else torch.float32

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._checkpoint, torch_dtype=dtype
        ).to(device)

        if self._ip_adapter_repo is not None:
            pipeline.load_ip_adapter(
                self._ip_adapter_repo,
                subfolder=self._ip_adapter_subfolder,
                weight_name=self._ip_adapter_weights,
            )

        return pipeline  # type: ignore[no-any-return]


def _invoke_pipeline(pipeline: SDXLPipelineLike, args: _PipelineCallArgs) -> Image.Image:
    """Single sync call into the pipeline. Runs in a worker thread."""

    pipeline_device = _detect_pipeline_device(pipeline)
    generator = _make_generator(args.seed, device=pipeline_device)

    if args.ip_adapter_image is not None:
        # Apply IP-Adapter scale just-in-time per shot so different shots can
        # vary how strongly they hew to the persona reference. ``AttributeError``
        # is the typical "no IP-Adapter loaded" failure; ``RuntimeError`` covers
        # diffusers shape/state errors. We deliberately do NOT swallow OOM,
        # CUDA, or memory errors so they propagate.
        with contextlib.suppress(AttributeError, RuntimeError):
            pipeline.set_ip_adapter_scale(args.ip_adapter_scale)

    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        ip_adapter_image=args.ip_adapter_image,
    )

    images = getattr(output, "images", None)
    if not images:
        raise ProviderError("pipeline returned no images")
    image = images[0]
    if not isinstance(image, Image.Image):
        raise ProviderError(f"pipeline returned unexpected image type: {type(image).__name__}")
    return image


def _make_generator(seed: int, *, device: str | None = None) -> Any:
    """Build a torch ``Generator`` matching ``device`` for reproducible randoms.

    Diffusers requires the generator and the pipeline to share a device for
    deterministic output. If torch isn't importable (test stubs), fall back
    to the raw int seed — stub pipelines don't care.
    """

    try:
        import torch
    except ImportError:
        return seed
    if device:
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            # Older torch builds reject some device strings; fall back to CPU.
            generator = torch.Generator()
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _detect_pipeline_device(pipeline: SDXLPipelineLike) -> str | None:
    """Best-effort sniff of a diffusers pipeline's primary device."""

    device = getattr(pipeline, "device", None)
    if device is None:
        return None
    return str(device)


def _auto_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _round_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    rounded = (value // multiple) * multiple
    return rounded if rounded > 0 else multiple


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _persona_filename(anime_style: str, narrator_seed: int) -> str:
    safe = _FILENAME_SAFE.sub("_", anime_style.strip()) or "style"
    return f"{safe}__{narrator_seed}.png"


def _default_persona_dir() -> Path:
    env_dir = os.environ.get("BOOKTOANIME_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser() / "personas"
    # Fall back to platformdirs only when the env override isn't set.
    from platformdirs import user_data_dir

    return Path(user_data_dir("booktoanime", appauthor=False)) / "personas"


@register_visual_provider("sdxl_diffusers")
def _factory(sub_config: Mapping[str, Any]) -> SDXLDiffusersProvider:
    width = int(sub_config.get("width", 1024))
    height = int(sub_config.get("height", 1024))
    steps = int(sub_config.get("steps", 28))
    guidance = float(sub_config.get("guidance", 5.5))
    checkpoint = str(sub_config.get("checkpoint", "stabilityai/stable-diffusion-xl-base-1.0"))
    ip_adapter_repo = sub_config.get("ip_adapter_repo", "h94/IP-Adapter")
    ip_adapter = sub_config.get("ip_adapter")
    ip_adapter_subfolder = str(sub_config.get("ip_adapter_subfolder", "sdxl_models"))
    ip_adapter_weights = str(
        sub_config.get(
            "ip_adapter_weights",
            f"{ip_adapter}.safetensors" if ip_adapter else "ip-adapter-plus_sdxl_vit-h.safetensors",
        )
    )
    persona_dir_raw = sub_config.get("persona_dir")
    persona_dir = Path(persona_dir_raw).expanduser() if persona_dir_raw else None
    device = sub_config.get("device")
    return SDXLDiffusersProvider(
        checkpoint=checkpoint,
        ip_adapter_repo=str(ip_adapter_repo) if ip_adapter_repo else None,
        ip_adapter_subfolder=ip_adapter_subfolder,
        ip_adapter_weights=ip_adapter_weights,
        default_width=width,
        default_height=height,
        default_steps=steps,
        default_guidance=guidance,
        device=str(device) if device else None,
        persona_dir=persona_dir,
    )


__all__ = [
    "SDXLDiffusersProvider",
    "SDXLPipelineLike",
]
