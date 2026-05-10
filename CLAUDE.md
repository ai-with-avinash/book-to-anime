# BookToAnime — repo conventions for Claude

Project-specific guidance. Global rules in `~/.claude/CLAUDE.md` still apply
(Python, async I/O, uv, pytest, ruff, no hardcoded secrets, etc.).

## Layout

```
src/booktoanime/
  api/         FastAPI routers + schemas (request/response only)
  pipeline/    stage logic: parsing, topic_segmenter, summarizer,
               storyboard, image_renderer, tts_narrator, mouth_animator,
               video_assembler, orchestrator
  providers/   pluggable language/audio/visual/lipsync adapters
  state/       job repo + manifest persistence
  parsing/     PDF parsing models
  web/         Jinja templates + static assets
tests/         pytest, fully mocked, no network
```

Pipeline order: `parsing → structuring → storyboard → images → audio → assembly`.
Each stage writes a versioned artifact under `<data_dir>/jobs/<job_id>/`.

## Stage-specific notes

### Topic segmentation (`pipeline/topic_segmenter.py`)

* Heuristic only — never calls an LLM.
* Detects headings in first ~6 lines per page via regex + ALL-CAPS fallback.
* `_MIN_SPAN_WORDS = 150` filters spurious headings: any span with less
  body text is merged forward into the previous span. First-span sparse
  case held as seed; final span extended to end-of-doc.
* If you tighten/loosen detection, also re-tune `_MIN_SPAN_WORDS`.

### Summarization (`pipeline/summarizer.py`)

* Per-topic word budget: `max(20s, midpoint / N) * 165 wpm / 60`.
* Hard ceiling from `max_tokens_per_topic=700` (~525 words ≈ 3.2 min).
* JSON-mode required; non-JSON or missing `summary` → `ProviderError`.

### Storyboard (`pipeline/storyboard.py`)

* Targets ~8 s per shot, hard min 4 s, hard max 14 s, at 165 wpm.
* Trailing chunks below `_MIN_SHOT_SECONDS` get merged into previous.

### Resume rules

* Each stage resumes at first non-completed manifest entry.
* Image + audio stages reconcile `index.json` vs. on-disk files: missing
  files force re-render, orphan files get adopted.

## Conventions

* Async-only for I/O. Wrap blocking SDKs in `asyncio.to_thread`.
* Routers do request/response only. Business logic in pipeline/services.
* Providers register via `@register_*_provider("name")` decorators.
* Errors: typed `ProviderError`, `ParsingError`, etc. — surface actionable
  messages in UI; full stack traces only in `<job_dir>/logs/job.log`.
* Never introduce AGPL or other copyleft runtime deps. PyMuPDF is excluded.
* Tests fully mocked, no network. Run `pytest && ruff check src tests`.

## Tooling

* `uv` for env + lockfile. Recreate via `uv pip install -e ".[dev]"` inside
  `.venv`.
* CI mirrors the same Python + lockfile.
