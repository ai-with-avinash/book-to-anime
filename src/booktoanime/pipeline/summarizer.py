"""Depth-aware per-topic summarization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..errors import ProviderError
from ..parsing.models import ParsedDocument
from ..providers import ChatMessage, CompletionRequest, LanguageProvider
from .artifacts import TopicSection
from .topic_segmenter import TopicSpan

Depth = Literal["eli5", "undergraduate", "expert"]
LengthPreset = Literal["short", "standard", "in_depth"]


_DEPTH_INSTRUCTIONS: dict[Depth, str] = {
    "eli5": (
        "Write at an Explain-Like-I'm-5 level. Use short sentences, everyday "
        "analogies, and avoid jargon. Briefly define any unfamiliar word."
    ),
    "undergraduate": (
        "Write at an undergraduate level. Be clear and precise; introduce "
        "key terminology with brief definitions; aim for educated-laymen depth."
    ),
    "expert": (
        "Write at an expert level. Be technical and precise; assume domain "
        "vocabulary; preserve nuance over breadth."
    ),
}


# Approximate per-topic narration budget (seconds) per length preset.
# These are *targets* — the summarizer aims for them and the storyboard
# stage may compress further if total exceeds the preset envelope.
_PRESET_TOTAL_SECONDS: dict[LengthPreset, tuple[float, float]] = {
    "short": (5 * 60.0, 10 * 60.0),
    "standard": (15 * 60.0, 25 * 60.0),
    "in_depth": (40 * 60.0, 60 * 60.0),
}


@dataclass(frozen=True)
class SummarizationConfig:
    depth: Depth = "undergraduate"
    length_preset: LengthPreset = "standard"
    minutes_per_topic: float | None = None
    max_tokens_per_topic: int = 700


class TopicSummarizer:
    """Calls a :class:`LanguageProvider` once per topic and collects the result.

    The output isn't free-form prose — we ask for JSON with ``summary``,
    ``key_points`` (list of bullets), and ``estimated_words``. We compute
    seconds at 165 wpm if the model gives us word counts; otherwise we
    estimate from character length.
    """

    def __init__(self, provider: LanguageProvider, config: SummarizationConfig) -> None:
        self._provider = provider
        self._config = config

    async def summarize_topics(
        self,
        document: ParsedDocument,
        spans: Sequence[TopicSpan],
    ) -> list[TopicSection]:
        target_seconds = self._target_seconds_per_topic(len(spans))
        sections: list[TopicSection] = []

        for span in spans:
            text = self._collect_topic_text(document, span)
            payload = await self._summarize_one(span.title, text, target_seconds)
            summary_value = str(payload.get("summary", "")).strip()
            key_points_raw = payload.get("key_points", [])
            key_points: list[str] = (
                [str(item) for item in key_points_raw]
                if isinstance(key_points_raw, list)
                else []
            )
            estimated = self._estimate_seconds(payload, fallback_text=summary_value)
            sections.append(
                TopicSection(
                    id=span.id,
                    title=span.title,
                    page_range=(span.page_start, span.page_end_inclusive),
                    summary=summary_value,
                    key_points=key_points,
                    image_refs=self._collect_image_refs(document, span),
                    table_refs=self._collect_table_refs(document, span),
                    estimated_narration_seconds=estimated,
                )
            )
        return sections

    # -------------------------------------------------------------- helpers

    def _target_seconds_per_topic(self, topic_count: int) -> float:
        if self._config.minutes_per_topic is not None:
            return float(self._config.minutes_per_topic) * 60.0
        lower, upper = _PRESET_TOTAL_SECONDS[self._config.length_preset]
        midpoint = (lower + upper) / 2.0
        if topic_count <= 0:
            return midpoint
        return max(20.0, midpoint / topic_count)

    @staticmethod
    def _collect_topic_text(document: ParsedDocument, span: TopicSpan) -> str:
        lines: list[str] = []
        for idx in range(span.page_start, span.page_end_inclusive + 1):
            if 0 <= idx < len(document.pages):
                page_text = document.pages[idx].text.strip()
                if page_text:
                    lines.append(page_text)
        return "\n\n".join(lines)

    @staticmethod
    def _collect_image_refs(document: ParsedDocument, span: TopicSpan) -> list[str]:
        refs: list[str] = []
        for idx in range(span.page_start, span.page_end_inclusive + 1):
            if 0 <= idx < len(document.pages):
                refs.extend(image.id for image in document.pages[idx].images)
        return refs

    @staticmethod
    def _collect_table_refs(document: ParsedDocument, span: TopicSpan) -> list[str]:
        refs: list[str] = []
        for idx in range(span.page_start, span.page_end_inclusive + 1):
            if 0 <= idx < len(document.pages):
                refs.extend(table.id for table in document.pages[idx].tables)
        return refs

    async def _summarize_one(
        self,
        title: str,
        text: str,
        target_seconds: float,
    ) -> dict[str, object]:
        depth_instruction = _DEPTH_INSTRUCTIONS[self._config.depth]
        word_target = max(40, int(target_seconds * 165 / 60))
        system_prompt = (
            "You are condensing one chapter of a document for a narrated "
            "explainer video. Use concrete examples and clear definitions. "
            "Match the source's domain — don't force STEM framing on non-"
            "technical content. " + depth_instruction
        )
        user_prompt = (
            f"Topic title: {title}\n"
            f"Target narration length: {target_seconds:.0f} seconds (~{word_target} words).\n"
            "Reply with valid JSON only:\n"
            '{"summary": "<narration-ready prose>",'
            ' "key_points": ["point 1", "point 2", ...],'
            ' "estimated_words": <int>}\n\n'
            f"Source text (truncated if long):\n{text[:8000]}"
        )

        response = await self._provider.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
                max_tokens=self._config.max_tokens_per_topic,
                temperature=0.3,
                json_mode=True,
            )
        )

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"summarizer response was not JSON: {response[:200]}"
            ) from exc

        if not isinstance(payload, dict) or not payload.get("summary"):
            raise ProviderError(f"summarizer response missing 'summary': {response[:200]}")
        return payload

    @staticmethod
    def _estimate_seconds(payload: dict[str, object], *, fallback_text: object) -> float:
        words = payload.get("estimated_words")
        # ``bool`` is an ``int`` subclass in Python; rule it out explicitly so
        # ``"estimated_words": true`` doesn't yield ~0.36 s.
        if isinstance(words, int | float) and not isinstance(words, bool) and words > 0:
            return float(words) * 60.0 / 165.0
        text = str(fallback_text or "")
        if not text:
            return 30.0
        return max(10.0, len(text.split()) * 60.0 / 165.0)
