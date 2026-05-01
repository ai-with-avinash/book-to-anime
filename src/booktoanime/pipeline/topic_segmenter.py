"""Heuristic topic segmentation.

Per the approved plan we segment by inspecting the parsed PDF for chapter /
section headings and only fall back to a sliding-window LLM if the text yields
no headings at all (rare for well-structured books).

Heading detection rules:

* A line that matches one of the heading patterns (``Chapter N``, ``Part N``,
  ``Section N``, leading numeric like ``1.``, ``1.2`` followed by a title,
  uppercase-only short lines).
* The line must be near the top of a page (first ~3 lines) OR the line
  before/after must be empty — this filters narrative sentences that happen
  to start with a number.

This module never calls an LLM by itself; the orchestrator can pass topic
titles through a :class:`LanguageProvider` later for refinement, but that's
optional.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..parsing.models import ParsedDocument, ParsedPage

_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(chapter|part|section)\s+([0-9IVXLC]+)\b[\s:.\-]*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\.\s+([A-Z][^.\n]{2,80})\s*$"),
    re.compile(r"^\s*(\d+\.\d+)\s+([A-Z][^.\n]{2,80})\s*$"),
)

_MAX_TITLE_LEN = 120
_MIN_TITLE_LEN = 4


@dataclass(frozen=True)
class _Heading:
    page_index: int
    title: str


@dataclass(frozen=True)
class TopicSpan:
    """A contiguous range of pages assigned to one topic."""

    id: str
    title: str
    page_start: int
    page_end_inclusive: int


class TopicSegmenter:
    """Pure-Python topic segmentation."""

    def segment(self, document: ParsedDocument) -> list[TopicSpan]:
        headings = list(self._detect_headings(document.pages))
        if not headings:
            return self._whole_book_as_topic(document)

        spans: list[TopicSpan] = []
        for idx, heading in enumerate(headings):
            page_end = (
                headings[idx + 1].page_index - 1
                if idx + 1 < len(headings)
                else len(document.pages) - 1
            )
            spans.append(
                TopicSpan(
                    id=f"topic_{idx + 1:03d}",
                    title=heading.title,
                    page_start=heading.page_index,
                    page_end_inclusive=max(page_end, heading.page_index),
                )
            )
        return spans

    # -------------------------------------------------------------- helpers

    def _detect_headings(self, pages: Sequence[ParsedPage]) -> Iterable[_Heading]:
        seen_titles: set[str] = set()
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            for line in lines[:6]:
                title = self._normalize_heading(line)
                if title is None:
                    continue
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                yield _Heading(page_index=page.index, title=title)
                # Only emit one heading per page to avoid splitting too aggressively.
                break

    def _normalize_heading(self, line: str) -> str | None:
        if not (_MIN_TITLE_LEN <= len(line) <= _MAX_TITLE_LEN):
            return None

        for pattern in _HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                groups = [g for g in match.groups() if g]
                title = " ".join(part.strip(" :.-") for part in groups if part.strip())
                title = re.sub(r"\s+", " ", title).strip(" :.-")
                if not title:
                    continue
                # Re-prefix with "Chapter"/"Section" for clarity if pattern matched a leading word.
                head = groups[0]
                if head.lower() in {"chapter", "part", "section"} and not title.lower().startswith(
                    head.lower()
                ):
                    title = f"{head.title()} {title}"
                if len(title) >= _MIN_TITLE_LEN:
                    return title

        # Fallback: ALL-CAPS short line (likely a heading).
        if line.isupper() and 8 <= len(line) <= 80:
            return line.title()

        return None

    def _whole_book_as_topic(self, document: ParsedDocument) -> list[TopicSpan]:
        title = (document.metadata.title or "Whole Book").strip() or "Whole Book"
        last = max(0, len(document.pages) - 1)
        return [
            TopicSpan(
                id="topic_001",
                title=title,
                page_start=0,
                page_end_inclusive=last,
            )
        ]
