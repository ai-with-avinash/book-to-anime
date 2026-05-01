# BookToAnime — Audit Report (2026-05-01 UTC)

Scope: 51 source files (`src/booktoanime/`), 115 tests, plan
`/Users/avinash/.claude/plans/i-want-to-build-quiet-catmull.md`,
git HEAD `1cdf8cd` ahead of `origin/main` by 1 commit.

Static gates run via allowed commands only. No source files were modified.

---

## Top 5 must-fix

| # | Severity | File:line | One-line summary |
|---|---|---|---|
| 1 | HIGH | `pipeline/video_assembler.py:248-278` | n=1 single-shot filtergraph likely fails; xfade chain offset math accumulates incorrectly past shot 3; `offsets` list (243-246) is dead code |
| 2 | HIGH | `pipeline/image_renderer.py:158` | `_ensure_persona_reference` derives `anime_style` from `persona.style_descriptor.split(",")[0].strip()` which yields `"shounen-bright narrator persona"` not `"shounen-bright"` — wrong cache key + wrong style fragment lookup |
| 3 | HIGH | `pyproject.toml:88-90` | `[tool.hatch.build.targets.wheel.force-include]` re-includes resources already shipped via `packages = ["src/booktoanime"]` → wheel build emits 4 `Duplicate name` warnings + every web asset stored twice (verified by `unzip -l`) |
| 4 | HIGH | `api/routes_jobs.py:164-165` | `_load_manifest` returns `JobManifest \| None` but `JobRunner.start` is typed `manifest: JobManifest`; if manifest is corrupt the resume endpoint passes None — call doesn't crash today only because `start` ignores the parameter (manifest is always re-read from disk by orchestrator), but the dead parameter + missing null-check is a contract violation |
| 5 | HIGH | `api/routes_sse.py:33-49` | SSE `event_stream` checks `request.is_disconnected()` *before* the blocking `queue.get()`; on a long idle stream the coroutine hangs forever holding a subscriber slot when the client disconnects between events |

## Plan compliance matrix

| Plan § | Title | Verdict | Gap |
|---|---|---|---|
| §1 | Project Summary (one paragraph) | PASS | README §1 matches plan summary verbatim. |
| §2 | Directory Layout | PARTIAL | Actual tree matches plan + 3 extras: `pipeline/srt_sidecar.py` (added by §16 ambiguity decision), `providers/_retry.py`, `providers/language/_sdk_helpers.py` (private helpers, plan §5 doesn't list — acceptable). Missing: plan promised `routes_models.py`, `disk.py`, `paths.py`, `logging.py`, `config.py`, `models_cache/*` — none implemented. v1 deferred but plan claimed them in §5. |
| §3 | Provider Interfaces | PASS | `providers/base.py` matches plan §3 signatures. `ChatMessage`, `CompletionRequest`, `VisionInput`, `TTSRequest`, `ImageGenRequest`, `GeneratedImage`, `GeneratedAudio`, `ImageExplanation` all frozen dataclasses; `LanguageProvider` / `AudioProvider` / `VisualProvider` ABCs with documented methods. |
| §4 | JSON Schemas | PASS | `manifest.json`, `extracted/parsed.json`, `structured.json`, `storyboard.json`, `images/index.json`, `audio/index.json` all implemented as pydantic v2 models with `extra="forbid"` and `JobRelPath` validation. `events.log` NDJSON shape matches plan §4.6. |
| §5 | Module list (one-liners) | PARTIAL | `cli`, `errors`, `parsing.*`, `providers.*`, `pipeline.*`, `state.*`, `api.*`, `web/*` all present. Missing modules: `paths.py`, `logging.py`, `config.py`, `disk.py`, `models_cache.*`, `pipeline.video_assembler` (present, despite plan listing it last), `routes_models.py`. The "missing" set are `MODELS DOWNLOAD` infra promised in plan but never built (defer to v0.2 acceptable). |
| §6 | Dependency list (with licenses) | PASS | Every default runtime dep in plan §6 present in `pyproject.toml`. PyMuPDF correctly absent. License labels match. Optional extras `[anthropic]/[gemini]/[groq]/[together]/[fireworks]/[mistral]/[openai-compat]/[kokoro]/[xtts]/[visual]/[all-providers]` all wired. |
| §7 | Example `config.yaml` | PASS | `config.example.yaml` parses cleanly via `yaml.safe_load`; every active provider exists in registry; vision_fallback documented; local-LLM examples (Ollama/vLLM/LM Studio/llama.cpp) commented; `xtts` + `flux_dev` opt-in blocks present. |
| §8 | README Outline | PASS | All 15 numbered sections from plan §8 present in README. |
| §9 | Ambiguities & Tradeoffs (17) | PASS | Decisions taken (one per ambiguity): see "Ambiguity decisions" below. |
| §10 | Build Order | PASS | Module order in commit history matches plan §10: parsing → providers (lang/audio/visual) → pipeline orchestration → API+frontend → assembly. |
| §11 | Verification | PARTIAL | `pytest tests/unit/test_pdf_parser.py tests/unit/test_image_extractor.py` PASS; `pytest tests` 115/115 PASS; `booktoanime --help` PASS; `--providers-mock` flag promised in plan but never implemented (use injectable `PipelineDependencies` instead — orthogonal mechanism). End-to-end against real LLM/SDXL/Kokoro/ffmpeg NOT verified. |

### Ambiguity decisions (plan §9)

1. Topic segmentation: heuristic (TOC + heading detection), no LLM refinement (just headings). Whole-book fallback when zero headings. ✓ Recommendation followed minus the LLM-refinement step.
2. Persona consistency: IP-Adapter only, single seed. ✓
3. Storyboard granularity: ~7-9s shots driven by sentence chunking at 165 wpm, capped at 14s. ✓
4. Length-preset enforcement: preset is a *target*; `minutes_per_topic` overrides. ✓
5. VLM availability: silent degradation via `_synthesize_from_caption` in `image_explainer.py` (BUT not currently invoked — see §B M-7).
6. TTS language coverage: en-US + en-GB allow-lists in Kokoro; non-English raises ProviderError on voice validation. ✓
7. Concurrency: profile→cap map (default=2, high_quality=1, low_vram=1). ✓
8. Resume granularity: per-stage everywhere, per-shot for images + audio (filesystem-truthful reconciliation). ✓
9. DB schema scope: minimal — jobs table only; rich data in events.log + index.json. ✓
10. htmx + Alpine.js (no build step). ✓
11. First-run gate: stop-with-message in CLI, exits non-zero. ✓
12. Cost-estimate freshness: `docs/costs.md` with explicit "Last updated: 2026-04". ✓
13. Anime-style preset list: 4 presets (`shounen-bright`, `shoujo-soft`, `seinen-muted`, `chibi`). ✓
14. Tables in narration: prose-only (no table rendering). ✓
15. Background music / SFX: none (in roadmap). ✓
16. Captions / subtitles: `.srt` sidecar shipped. ✓
17. Telemetry: explicit none, README documents. ✓

---

## Findings

### `pipeline/video_assembler.py`

**video_assembler.py:248-249 — HIGH — n=1 single-shot filtergraph likely invalid**
For n=1: video filter is `[v0]copy[vout]` and audio filter `[1:a]concat=n=1:v=0:a=1[aout]`. ffmpeg's `concat` filter requires at least 2 inputs in the typical case; a single-shot run will likely fail with "concat needs at least 2 segments". Test `test_assemble_invokes_runner_and_writes_subtitles` exercises n=2; no test covers n=1.
*Fix*: special-case n=1 to use `[1:a]anull[aout]` (or skip the audio filter chain entirely and pass `-map 1:a` directly).

**video_assembler.py:243-246 — MEDIUM — `offsets` list computed but never used**
The local `offsets = [0.0]` + loop populating it is dead code; the actual xfade timing uses `cumulative` later. Either delete `offsets` or use it.
*Fix*: remove the dead `offsets` block.

**video_assembler.py:251-271 — HIGH — xfade offset math drifts past shot 3**
`cumulative` starts at `durations[shot[0]]`, then per-iteration `cumulative += durations[shot[idx]] - fade_seconds`. For shot 2 the offset `cumulative - fade_seconds` correctly equals `dur(0) - fade(0,1)`. For shot 3 it equals `dur(0) + dur(1) - 2*fade` — but the previous xfade output's effective duration is `dur(0) + dur(1) - fade(0,1)`, so the next xfade's offset should be `dur(0) + dur(1) - fade(0,1) - fade(1,2)` (relative to the start of that already-faded output). The code's accumulator under-counts by exactly the correct amount per shot only when all fades are equal — non-uniform crossfades shift shots out of sync.
*Fix*: track `previous_output_duration` separately from raw cumulative input duration; offset for xfade k = `previous_output_duration - fade_k`.

**video_assembler.py:295 — MEDIUM — `process.communicate()` buffers all stdout/stderr in RAM**
ffmpeg verbose stderr can be hundreds of MB on a long encode. `communicate()` reads it all before `log_path.write_bytes`. Long real renders may OOM.
*Fix*: stream stderr to disk via `asyncio.create_subprocess_exec(..., stderr=open(log_path, "wb"))` or per-line tail.

**video_assembler.py:204 — LOW — `-shortest` with image-loop inputs may truncate video before audio finishes**
With xfade-chained video and concat'd audio, the audio total = sum(durations); video total ≈ same minus fade overlap. `-shortest` cuts to the shorter — fine if math is right, but bug 3 above means video may be longer than audio for non-uniform fades.

### `pipeline/image_renderer.py`

**image_renderer.py:158 — HIGH — wrong anime_style derived from persona descriptor**
`persona.style_descriptor` (built by `narrator_persona.derive_persona`) has shape `f"{anime_style} narrator persona, voice {voice_id} in {language}"`. Splitting on `,` returns `"shounen-bright narrator persona"`, not `"shounen-bright"`. Effects:
- Visual provider's `_STYLE_FRAGMENTS.get(anime_style)` returns `None` → falls back to literal value as prompt (wrong fragment).
- Cache key in `SDXLDiffusersProvider.prepare` becomes `"shounen-bright_narrator_persona__<seed>.png"` — incompatible with any later code that recomputes the cache key from raw `anime_style`. Resume idempotency partially broken.
*Fix*: pass `anime_style` directly into `NarratorPersona` (new field) or split on `" "` and take first token, or reorder the descriptor so `split(",")[0]` returns just the style.

### `api/routes_jobs.py`

**api/routes_jobs.py:164-165 — HIGH — passes `manifest=None` to typed `start(manifest: JobManifest)`**
`_load_manifest` declared `JobManifest | None`. mypy doesn't catch the call because `request.app.state.runner` is `Any`. Today this is harmless because `JobRunner.start` never reads the `manifest` parameter (orchestrator re-reads from disk), but the parameter is dead code AND a future maintainer adding `manifest.config.something` inside `start` will get `AttributeError: 'NoneType'`.
*Fix*: either drop the `manifest` parameter from `start` or make `_load_manifest` raise on missing/corrupt manifest at the route boundary.

**api/routes_jobs.py:108 — LOW — `_generate_job_id` uses `secrets.choice` on `[A-Z0-9]` not real ULID**
`ulid-py` is a runtime dependency but only the docstring mentions ULID — the actual ID is a 26-char alphanumeric. Either wire ulid-py in or drop it.

### `api/routes_sse.py`

**api/routes_sse.py:33-49 — HIGH — disconnect detection blocks behind `queue.get()`**
After the first event, the client disconnects. `await request.is_disconnected()` is checked at the *top* of the loop, then the next iteration is `await queue.get()` which never resolves (no more events arrive). The coroutine hangs holding the bus subscriber slot until the bus is closed.
*Fix*: race the queue read with disconnect using `asyncio.wait({queue.get(), request.is_disconnected()}, return_when=FIRST_COMPLETED)` or use sse-starlette's `EventSourceResponse(send_timeout=...)`.

### `pipeline/events.py`

**pipeline/events.py:99-104 — MEDIUM — drop policy mismatches docstring**
Docstring (line 80-82) says "oldest events dropped". Implementation drops the *new* event when `put_nowait` raises `QueueFull`, keeping the oldest 1024.
*Fix*: either update the docstring or implement true drop-oldest (`queue.get_nowait()` + retry).

**pipeline/events.py:92-94 — LOW — `emit` raises RuntimeError if bus closed during shutdown**
Orchestrator emits unconditionally. A race between `bus.close()` and `bus.emit()` produces an unrelated `RuntimeError("event bus is closed")` that propagates as the orchestrator's failure cause. Should silently no-op on closed.

### `pipeline/orchestrator.py`

**pipeline/orchestrator.py:68-72 — LOW — stale docstring**
Says "Runs the whole pipeline (except video assembly)" + "Module 7 owns assembly; orchestrator stops at audio". Module 7 wired the assembly stage in (line 165-166); docstring outdated.

**pipeline/orchestrator.py:92 — LOW — premature `update_status(RUNNING)`**
Called before checking whether all stages are already completed (resume on completed job). Manifest flips RUNNING → COMPLETED for no-op runs. Cosmetic.

### `pipeline/srt_sidecar.py`

**srt_sidecar.py:38 — LOW — joins blocks with `\n` but each block already ends with `\n`**
Output uses double newlines between cues — correct SRT spacing, intentional. PASS but worth a brief comment.

**srt_sidecar.py:35 — LOW — narration_text not sanitized for control chars**
Embedded `\r` or other ASCII control bytes from a model response would land in the SRT verbatim. Most players tolerate; some don't.

### `pyproject.toml`

**pyproject.toml:88-90 — HIGH — wheel duplicate-name warnings**
```
[tool.hatch.build.targets.wheel.force-include]
"src/booktoanime/web" = "booktoanime/web"
"src/booktoanime/state/schema.sql" = "booktoanime/state/schema.sql"
```
`packages = ["src/booktoanime"]` already includes every file under `src/booktoanime/` including non-Python. `force-include` re-adds them, producing 4 `UserWarning: Duplicate name` lines on every build and inflating the wheel. Verified by `unzip -l /tmp/btoa-build/booktoanime-0.0.1-py3-none-any.whl` showing each web file twice.
*Fix*: delete the entire `[tool.hatch.build.targets.wheel.force-include]` section.

**pyproject.toml:32 — MEDIUM — `ulid-py` listed as runtime dep but unused**
Only reference in source: `api/routes_jobs.py:219` docstring "ULID-like ... without the ulid-py dep cost". Actual ID generation uses `secrets.choice`.
*Fix*: drop from `dependencies` OR replace `_generate_job_id` with `ulid.new().str`.

**pyproject.toml:50 — SPECULATIVE — `together`, `fireworks-ai`, `mistralai` extras unused**
The respective vendor wrappers reuse `OpenAICompatibleProvider` (raw httpx). The native SDKs are never imported anywhere. Either remove the extras or wire native adapters.

### `parsing/pdf_parser.py`

**parsing/pdf_parser.py:151 — LOW — "empty password" decrypt return value**
`reader.decrypt("") == 0` treats 0 as failure. pypdf 5.x returns a `PasswordType` enum, not an int — comparison still works because Enum compares by value, but it's brittle across versions.

**parsing/pdf_parser.py:197 — MEDIUM — `_caption_hint` returns ONE caption per page applied to ALL images**
Three figures on a page all get the same `caption_hint` ("Figure 1.1"). Already noted in earlier review.
*Fix*: bbox proximity match, deferred per project plan §9.

### Cross-cutting

**Stage enum coverage — PASS**
All Python references to stage names use `Stage.<NAME>.value`. Only string-literal usage is in `web/templates/job.html` (cosmetic UI loop).

**JobRelPath coverage — PASS**
Applied on `ParsedDocument.source_pdf`, `ExtractedImage.file`, `NarratorPersona.reference_image`, `ShotImageRecord.file`, `ShotAudioRecord.file`. No other "job-relative path" fields exist.

**Pydantic `extra="forbid"` — PASS**
26 BaseModel classes, 26 `extra="forbid"` declarations.

**Async close awaits — PASS**
`api/deps.py:149-157, 182` covers language/audio/visual/vision_fallback/bus on every job teardown.

**NOTICE alignment — PASS**
Every default runtime dep in pyproject.toml appears in NOTICE. Every NOTICE entry traces to a real dep (or to a bundled extra documented as opt-in).

**Version alignment — PASS**
`pyproject.toml version = "0.0.1"` matches `__version__ = "0.0.1"` matches `booktoanime version` output.

**`booktoanime --help` — PASS**
Returns Typer help with `run / resume / version` commands.

---

## Static-analysis summary

| Tool | Command | Exit | Findings count | Notes |
|---|---|---|---|---|
| ruff | `.venv/bin/ruff check src tests` | 0 | 0 | All checks passed (selectors `E,F,I,B,W,UP,N,SIM,RUF`). |
| mypy | `.venv/bin/python -m mypy --strict src` | 0 | 0 | 51 source files, no issues. NOTE: misses dynamic-attribute issues (e.g. `request.app.state.runner` is Any → bug 4 above slipped through). |
| pytest | `.venv/bin/python -m pytest tests` | 0 | 115 passed | All stubs; no real LLM/SDXL/ffmpeg. |
| build (wheel) | `.venv/bin/python -m build --wheel --outdir /tmp/btoa-build` | 0 | 4 warnings | `Duplicate name: 'booktoanime/web/...'` × 4 — see pyproject.toml finding above. |
| import smoke | `python -c "import booktoanime; ..."` | 0 | 51 modules | All modules import cold. |

---

## Packaging check

- **wheel build**: PASS — `booktoanime-0.0.1-py3-none-any.whl` produced. WARNINGS: 4 duplicate-name (see HIGH finding above).
- **wheel contents**: PASS for inclusion (verified each via `unzip -l`):
  - `booktoanime/web/static/app.css` (DUPLICATED — appears twice)
  - `booktoanime/web/templates/_layout.html` (DUPLICATED)
  - `booktoanime/web/templates/index.html` (DUPLICATED)
  - `booktoanime/web/templates/job.html` (DUPLICATED)
  - `booktoanime/state/schema.sql` (single — doesn't duplicate because hatch's default packaging maybe picks .sql up via package, force-include re-adds, ZIP only stored once due to identical hash. Verified single entry.)
- **entrypoint**: PASS — `booktoanime --help` returns non-empty; `booktoanime version` returns `booktoanime 0.0.1`.

---

## Live-run gap

The following components have NEVER been exercised end-to-end with their real
upstream:

- **Real LLM (any provider)**: 8 adapters all unit-tested with mocked HTTP /
  stub SDK clients. No production-like book has been summarized through Groq /
  Anthropic / Gemini / OpenAI-compatible. Token-budget assumptions in
  `summarizer.py` (`max_tokens_per_topic=700`, `target_seconds * 165 / 60`
  word target) are unverified.
- **Real Kokoro TTS**: model weights never downloaded by the test suite.
  Engine is stubbed via `KokoroEngine` Protocol. Sample-rate handling, soft
  clipping behavior, and Kokoro's actual chunk-tuple shape are
  literature-only.
- **Real SDXL + IP-Adapter**: pipeline factory never invoked on real torch.
  Persona consistency claims rely on `IP-Adapter` working as documented.
  Device selection (`mps` on Apple Silicon) untested. CUDA path untested.
- **Real ffmpeg**: filter_complex argv has been built and inspected by
  `test_video_assembler` (string assertions) but never fed to the actual
  binary. Bugs noted under `video_assembler.py` (n=1 case, xfade offset
  math) would surface only at first real run.
- **Browser SSE under sustained load**: `routes_sse.py` works in
  TestClient (which auto-disconnects). Long-lived browser tabs + the
  disconnect-blocking issue (HIGH finding 5) are unverified.
- **Docker image**: `Dockerfile` has not been built; no `docker build .`
  attempt within audit scope.
- **Encrypted/corrupted/image-only PDFs**: synthetic fixtures cover the
  typed-error paths but real-world malformed PDFs may surface other pypdf /
  pdfplumber exceptions not mapped through the typed hierarchy.

---

## Closing verdict

**PARTIAL** for shipping a v0.0.1.

Blockers before tagging:
1. Fix `video_assembler.py` n=1 case + offset math (HIGH 1, 3) — first real
   ffmpeg run with one shot OR three+ shots will produce a broken / truncated
   MP4.
2. Fix `image_renderer.py:158` style derivation (HIGH 2) — persona cache
   key drifts from explicit `anime_style` argument; resume across
   re-orchestration may regenerate persona unnecessarily.
3. Drop `[tool.hatch.build.targets.wheel.force-include]` (HIGH 3) — wheel
   ships duplicates today.
4. Fix `routes_sse.py` disconnect race (HIGH 5) — first real long browser
   session leaves a hung subscriber per disconnect.
5. Either wire ulid-py or drop it (MEDIUM).

Non-blocker (defer): real-world ffmpeg encode against a 10-shot tiny PDF, run
on a CUDA box for SDXL, and a 5-min browser SSE soak.

After fixes 1-5: **PASS for v0.0.1 alpha**.
