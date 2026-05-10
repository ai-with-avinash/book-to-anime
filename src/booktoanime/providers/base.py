"""Abstract provider interfaces and value objects.

These types are the only contract the pipeline depends on; concrete adapters
under ``providers.language``, ``providers.audio``, and ``providers.visual`` plug
in via :mod:`booktoanime.providers.registry`.

All I/O is async. Adapters that wrap synchronous SDKs MUST hop those calls to
``asyncio.to_thread`` rather than blocking the event loop. Adapters MUST also
honor cancellation: long-running calls should propagate ``asyncio.CancelledError``
back to the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn in an LLM conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """Inputs to a single LLM completion call.

    Attributes:
        messages: Ordered conversation; first message may be a system prompt.
        max_tokens: Hard cap on output tokens.
        temperature: Sampling temperature in ``[0.0, 2.0]``; ``0.0`` == greedy.
        json_mode: If True, request structured JSON output where supported.
            Adapters that lack native JSON mode MUST fall back to a strict
            "respond with valid JSON only" system instruction.
        stop: Optional stop sequences.
        seed: Optional integer seed for deterministic output where supported.
    """

    messages: Sequence[ChatMessage]
    max_tokens: int
    temperature: float = 0.2
    json_mode: bool = False
    stop: Sequence[str] | None = None
    seed: int | None = None


@dataclass(frozen=True)
class VisionInput:
    """An image input plus optional grounding text passed to a VLM.

    Attributes:
        image_path: Absolute path to a local image file (PNG/JPEG/WEBP).
        surrounding_text: Page or paragraph context to ground the explanation.
        caption_hint: Optional caption pulled from the PDF (e.g. ``"Figure 2.3"``).
    """

    image_path: Path
    surrounding_text: str
    caption_hint: str | None = None


@dataclass(frozen=True)
class TTSRequest:
    """Inputs to a single TTS synthesis call.

    Attributes:
        text: Plain text to speak. SSML is NOT assumed.
        voice_id: Provider-specific voice identifier.
        language: BCP-47 tag (``"en-US"``, ``"ja-JP"``). Must match the voice.
        speed: ``0.5``-``2.0`` multiplier on baseline speaking rate.
    """

    text: str
    voice_id: str
    language: str
    speed: float = 1.0


@dataclass(frozen=True)
class ImageGenRequest:
    """Inputs to a single image generation call."""

    prompt: str
    width: int
    height: int
    seed: int
    steps: int
    guidance: float
    negative_prompt: str | None = None
    reference_image: Path | None = None
    reference_strength: float = 0.65


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    seed: int
    width: int
    height: int


@dataclass(frozen=True)
class GeneratedAudio:
    path: Path
    duration_seconds: float
    sample_rate: int


@dataclass(frozen=True)
class AnimatedShot:
    """One per-shot lip-synced video clip produced by a :class:`LipSyncProvider`.

    ``duration_seconds`` is the *measured* output length (re-probed from the
    file). It can drift from the input WAV by a frame or two because some
    lip-sync models snap to fixed FPS — assemblers that splice these clips
    together must use this measured value, not the source audio length.
    """

    path: Path
    duration_seconds: float
    fps: float


@dataclass(frozen=True)
class ImageExplanation:
    """A grounded explanation of an embedded PDF image.

    Attributes:
        summary: 1-3 sentences, narration-ready.
        detail: Longer explanation used during storyboard reasoning.
    """

    summary: str
    detail: str


# --------------------------------------------------------------------- contracts


class LanguageProvider(ABC):
    """Pluggable text-LLM provider.

    Implementations must be safe to instantiate from a config dict and reusable
    across many calls within a single job.
    """

    name: str

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> str:
        """Run one completion. Return the assistant text only.

        Raises:
            booktoanime.errors.ProviderAuthError: Credentials missing/rejected.
            booktoanime.errors.ProviderRateLimitError: Quota exceeded.
            booktoanime.errors.ProviderTransientError: Network or 5xx error.
            booktoanime.errors.ProviderError: Any other provider failure.
        """

    @abstractmethod
    async def explain_image(
        self,
        image: VisionInput,
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> ImageExplanation:
        """Generate a grounded explanation of an embedded image.

        Implementations that lack vision capability MUST raise
        :class:`booktoanime.errors.CapabilityNotSupportedError` so the
        orchestrator can route to ``language.vision_fallback``.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release HTTP clients, sessions, and any pooled resources."""


class AudioProvider(ABC):
    """Pluggable TTS provider."""

    name: str

    @abstractmethod
    async def list_voices(self, language: str | None = None) -> Sequence[str]:
        """Return supported voice ids, optionally filtered by language."""

    @abstractmethod
    async def synthesize(self, request: TTSRequest, out_path: Path) -> GeneratedAudio:
        """Write narration audio to ``out_path`` and return metadata."""

    @abstractmethod
    async def close(self) -> None: ...


class VisualProvider(ABC):
    """Pluggable image-generation provider with character/style consistency."""

    name: str

    @abstractmethod
    async def prepare(self, *, anime_style: str, narrator_seed: int) -> Path:
        """Generate (or load cached) narrator-persona reference image.

        Returns the path used as the IP-Adapter reference for later shots.
        Implementations without IP-Adapter support must still return a usable
        reference image and rely on prompt-only consistency.
        """

    @abstractmethod
    async def render(self, request: ImageGenRequest, out_path: Path) -> GeneratedImage:
        """Render one shot image to ``out_path``."""

    @abstractmethod
    async def close(self) -> None: ...


class LipSyncProvider(ABC):
    """Pluggable lip-sync provider.

    Given a still anime portrait and a per-shot narration WAV, produce a
    short mp4 with mouth motion roughly synced to the audio. Adapters that
    require model downloads SHOULD cache them under ``data_dir/models/``;
    adapters that hit a hosted API MUST honor cancellation and surface
    actionable errors via :class:`booktoanime.errors.ProviderError`.
    """

    name: str

    @abstractmethod
    async def animate(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        out_path: Path,
    ) -> AnimatedShot:
        """Write a lip-synced clip to ``out_path`` and return its metadata."""

    @abstractmethod
    async def close(self) -> None: ...
