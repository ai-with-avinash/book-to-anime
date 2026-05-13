# Changelog

All notable changes to this project are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/) and the
project adheres to [Semantic Versioning](https://semver.org/) starting at
v0.1.0.

## [0.1.0] — Unreleased (StudyPanels rename + phase-1 strip)

This release pivots the project from **"BookToAnime — PDF to anime explainer"**
to **"StudyPanels — PDF to STEM motion-comic study explainer"**. The brand,
default visual stack, and runtime path all change. The public CLI binary
remains `booktoanime` for this release to avoid breaking entry-point
consumers; the binary rename to `studypanels` is queued for v0.2.

### BREAKING

- **`anime_style` configuration key renamed to `panel_style`.** All
  references in `config.yaml`, the upload form, the JSON API, the job
  manifest, and the storyboard config use the new name. Old manifests are
  refused at load time — see *Manifest schema bump* below.
- **Lip-sync stack removed in its entirety.**
  - The `MOUTH_ANIMATION` pipeline stage is gone.
  - The `LipSyncProvider` ABC, the `AnimatedShot` dataclass, and the
    `passthrough`, `sadtalker_local`, and `replicate_hosted` adapters under
    `providers/lipsync/` are deleted.
  - The `MouthIndex` / `MouthShotRecord` artifacts and the `mouth/`
    job-directory subtree are gone.
  - The `lipsync:` block has been removed from `config.example.yaml` and
    the matching `LipSyncConfig` / `JobConfig.lipsync` fields are gone from
    the manifest.
  - `lipsync_enabled` is no longer accepted by the job-create endpoint.
- **`narrator_persona.py` module deleted.** The character-narrator persona
  concept no longer fits the STEM motion-comic framing. The
  `NarratorPersona` Pydantic model has been removed from
  `pipeline/artifacts.py`. Phase 2 reintroduces a *style anchor* (no
  character) under `STYLE_SEEDING`.
- **`Stage.PERSONA_SEEDING`** is now a no-op stub that emits an `INFO`
  progress event and advances. Phase 2 renames it to `STYLE_SEEDING` and
  wires a real `style_seeder` implementation.
- **Manifest schema v1 → v2 (incompatible).** `JobManifest.from_path`
  refuses to read any manifest whose `manifest_schema_version` differs from
  the build-time constant. There is no in-place migration in this release.
  **You must back up or delete `<data_dir>/jobs/` before running v0.1.0
  against a directory previously used by a v0.0.x build.** The recommended
  command is:
  ```bash
  mv "$(python -c 'from booktoanime.cli import _default_data_dir; print(_default_data_dir())')/jobs" \
     "$(python -c 'from booktoanime.cli import _default_data_dir; print(_default_data_dir())')/jobs.pre-v0.1.bak"
  ```
- **Unused vendor extras removed from `pyproject.toml`:** `ulid-py`,
  `xtts`, `together`, `fireworks`, `mistral`. The corresponding adapters
  under `providers/language/` already used the raw HTTP `httpx` path, so
  they continue to work without their native SDKs. The `all-providers`
  extra now bundles only `openai`, `anthropic`, `google-genai`, and `groq`.
- **`_generate_job_id` no longer pretends to use ULID.** It now returns
  `secrets.token_urlsafe(13)` — fewer characters, fully URL-safe, sufficient
  entropy for one user's jobs.

### Added

- New CLI subcommand **`booktoanime check`** — probes Ollama
  (`/api/tags` + configured model present), Kokoro weight cache,
  `ffmpeg`, and `tesseract`. Exits non-zero on any failure. Auto-runs
  inside `booktoanime run` unless `--skip-preflight` is passed.
- New error types in `errors.py`:
  - `RenderError` — panel composer / figure compositing failures (consumed
    in phase 3).
  - `ManifestSchemaMismatch` — raised by the manifest loader on schema
    version drift.
- `JobManifest.manifest_schema_version: int` field; the loader refuses
  to read manifests whose value differs from the build constant.
- Hatchling auto-includes non-Python files under `packages = `, so the
  web `static/` and `templates/` trees ride along in the wheel without an
  explicit `force-include` block. The wheel build now produces zero
  "Duplicate name" warnings. Phase 3 will add `web/static/fonts/` under
  the same tree.

### Changed

- Project metadata: `version = "0.1.0"`, description, keywords, and author
  field updated for the StudyPanels brand.
- README rewritten to lead with the local Ollama free-stack path; hosted
  options demoted to "optional fallback". The panel-style overview is
  intentionally left as a placeholder; phase 2 fills it.
- `NOTICE` no longer attributes Replicate, SadTalker, or XTTS.
- Web UI rebranded: header text, page titles, and palette swapped to a
  cream/slate/teal academic palette. Lip-sync checkbox dropped from the
  upload form.
- `VisualProvider.prepare` keyword argument renamed `anime_style` →
  `panel_style`. The SDXL provider's filename helper and base prompt
  constant follow.
- `StoryboardConfig.anime_style` → `StoryboardConfig.panel_style`.
  `ImageRendererConfig.anime_style` → `ImageRendererConfig.panel_style`.
  `JobConfig.anime_style` → `JobConfig.panel_style`.

### Removed

- `src/booktoanime/providers/lipsync/` directory and the
  `register_lipsync_provider` / `build_lipsync_provider` registry hooks.
- `src/booktoanime/pipeline/mouth_animator.py`.
- `src/booktoanime/pipeline/narrator_persona.py`.
- `Stage.MOUTH_ANIMATION` and its slot in `STAGE_ORDER`.
- `pipeline/video_assembler.py`: the `mouth_index` parameter, the
  per-shot lip-sync mp4 input branch, the `preserve_ken_burns` config
  knob, and the `_subset_mouth` helper. The assembler now always uses
  the static `images/shot_*.png` inputs. **The xfade math is unchanged
  in this release; phase 4 fixes the AUDIT.md HIGH bugs.**
- Test `tests/unit/test_mouth_animator.py`.

### Phase 2 — Visual rewrite (Unreleased)

- **`Stage.PERSONA_SEEDING` renamed to `Stage.STYLE_SEEDING`.** The stage
  is no longer a stub: a new `pipeline/style_seeder.py` module produces a
  single per-job IP-Adapter style anchor (no character) keyed on
  `(panel_style, seed)` and writes it to `<job_dir>/style/`.
- **`pipeline/styles.py` (NEW)** exports `STYLE_FRAGMENTS` — the canonical
  panel-style → prompt-fragment mapping consumed by the storyboard
  builder, the SDXL provider, and the style seeder. Four real styles
  ship: `clean-linework`, `chalkboard-sketch`, `watercolor-technical`,
  `flat-vector-infographic`.
- **`VisualKind` enum (NEW)** in `pipeline/artifacts.py` — `FIGURE`,
  `ILLUSTRATION`, `TITLE_CARD`. `Shot` gains `visual_kind` and
  `figure_id` (stable `ExtractedImage.image_id` reference, not a list
  index). Phase 2 plumbs the labels end-to-end; phase 3 wires the
  renderer dispatch.
- **`pipeline/storyboard.py`** rewritten prompt template:
  `"educational illustration of {topic_title}: {focus}, {style_fragment}"`
  (no `anime explainer scene` literal). New `_sentence_clean` helper.
  Storyboard now assigns `visual_kind`: first shot of a topic with three
  or more shots becomes `TITLE_CARD`; unconsumed `image_refs` become
  `FIGURE` (one per shot); remainder fall through to `ILLUSTRATION`.
- **`pipeline/summarizer.py`** system prompt neutralised. Drops the
  book-only framing and tells the model to match the source's domain
  instead of forcing STEM phrasing on every document.
- **`pipeline/image_renderer.py`** records `visual_kind` and `figure_id`
  in `images/index.json`. Reconciler invalidates an on-disk shot whose
  recorded kind or figure_id no longer matches the storyboard so resume
  re-renders stale panels rather than reusing them.
- **`JobArtifacts.style_reference`** (NEW pointer field) — the
  manifest now persists the seeder's `StyleReference` so resume can
  short-circuit re-rendering the anchor.
- **UI dropdown** in `web/templates/index.html` lists all four panel
  styles. `README.md` panel-style placeholder replaced with a real
  section. `config.example.yaml` comment lists the four choices.
- **Note:** v0.1.0 is still WIP. Jobs created on a phase-1 build cannot
  be resumed by a phase-2 build because their manifests don't carry the
  new `artifacts.style_reference` field and their storyboards lack the
  `visual_kind` / `figure_id` columns. Back up or delete
  `<data_dir>/jobs/` after pulling phase 2.

### Phase 3 — Figure-first rendering (Unreleased)

- **New `pipeline/panel_composer.py`** — Pillow-based panel composition.
  Two helpers:
  - `compose_figure_panel(figure_path, caption, title, panel_style,
    target_size)` — aspect-aware layout. Source aspect ≥ 1.2 gets a
    bottom-strip layout (figure on top, caption beneath); portrait /
    square sources get a side-panel layout (square figure block on the
    left, caption column on the right). Letterboxes figures inside their
    panel, applies the panel-style background colour + contrast-checked
    text colour, wraps captions with overflow ellipsis.
  - `compose_title_card(title, subtitle, panel_style, target_size)` —
    centred 56 pt bold title with 28 pt subtitle on the panel-style
    background.
  Both raise `RenderError` on unknown `panel_style` or font-load failure.
- **Bundled OFL-licensed font** — Inter Regular + Bold (407 KB / 415 KB)
  ship under `src/booktoanime/web/static/fonts/` and are loaded via
  `importlib.resources` so the path works from an installed wheel as well
  as from a source checkout. Attributed in `NOTICE` under OFL-1.1.
- **`pipeline/image_renderer.py` dispatch on `Shot.visual_kind`:**
  - `VisualKind.FIGURE` — resolves the source figure via
    `ExtractedImage.id` lookup, then composes a panel through
    `panel_composer.compose_figure_panel`. **Bypasses SDXL entirely.**
    Missing `figure_id` on a FIGURE shot now raises `RenderError`
    (previously surfaced as a `KeyError`).
  - `VisualKind.TITLE_CARD` — composes a card via
    `panel_composer.compose_title_card`. **Bypasses SDXL.**
  - `VisualKind.ILLUSTRATION` — existing SDXL path, with the
    `STYLE_SEEDING` reference image forwarded as the IP-Adapter anchor.
- **Split semaphores:** GPU-bound `_sdxl_semaphore` keeps its
  profile-driven cap; new CPU-bound `_compose_semaphore` is sized to
  `min(8, os.cpu_count() or 1)` so the Pillow path stops serialising
  behind SDXL.
- **Small-figure guard:** figures whose shortest edge is below 256 px
  fall through to SDXL with an `INFO` event so operators can spot a job
  whose figures were all bumped to fallback.
- **End-of-stage telemetry:** the image stage emits a single
  `ProgressEvent` with `figure_shots=… illustration_shots=… title_cards=…`
  so operators can monitor the figure-first vs. SDXL split without
  parsing per-shot events.
- **Tests:** new `tests/unit/test_panel_composer.py` covers layout
  dispatch, palette correctness (BG-pixel spot check, not pixel-hash),
  sentence-cleaning, unknown-style + missing-file + font-load-failure
  error paths. `tests/unit/test_image_renderer.py` extended with FIGURE /
  TITLE_CARD bypass assertions, missing/unknown `figure_id` error paths,
  small-figure fall-through, split-semaphore concurrency cap, and the
  end-of-stage telemetry event.
- **Wheel packaging:** hatchling's auto-include picks up the new
  `web/static/fonts/` subtree under `packages = ["src/booktoanime"]`;
  no force-include block needed. The wheel build is verified to bundle
  `Inter-{Regular,Bold}.ttf` via `unzip -l`.

### Phase 4 — Polish + AUDIT bug fixes (Unreleased)

- **Fix `video_assembler.py` n=1 single-shot filtergraph** (AUDIT HIGH 1).
  The previous `concat=n=1:v=0:a=1` audio filter was invalid for a single
  input; ffmpeg's `concat` filter requires at least two segments. The
  assembler now special-cases `n == 1` with `[0:v]copy[vout]` for video
  and `[1:a]anull[aout]` for audio, so the first ever real run of a
  single-topic job no longer fails to encode.
- **Fix `video_assembler.py` xfade offset math** (AUDIT HIGH 3). The
  previous accumulator added raw input durations and subtracted per-pair
  fades, which under-counted by the cumulative fade overlap for any
  non-uniform fade configuration past shot 3. The assembler now tracks
  `rendered_so_far` — the running length of the already-faded output —
  separately from raw cumulative input duration. The xfade offset for
  shot k is computed as `rendered_so_far - fade_k`, then
  `rendered_so_far = offset + durations[k]`. A unit test asserts the
  closed-form values for 3 shots with non-uniform fades.
- **Stream ffmpeg stderr directly to disk** (AUDIT MEDIUM). The default
  runner now binds ffmpeg's `stderr` fd to the opened log file via
  `asyncio.create_subprocess_exec(..., stderr=log_handle)` instead of
  buffering through `process.communicate()`. Long encodes can produce
  hundreds of MB of verbose ffmpeg chatter; the previous path could OOM
  the orchestrator on a real book-length run.
- **Dropped dead `offsets` block** in the filtergraph builder (AUDIT
  MEDIUM). The local `offsets = [0.0]` accumulator was computed but
  never used — the actual xfade timing always ran off the `cumulative`
  / `rendered_so_far` variable.
- **Fix `routes_sse.py` disconnect race** (AUDIT HIGH 5). The SSE
  event-stream loop now races `queue.get()` against
  `request.is_disconnected()` via `asyncio.wait(FIRST_COMPLETED)`
  wrapping both in explicit `create_task` calls. Any orphan task is
  cancelled on every iteration, and a `CancelledError` propagated into
  the wait will cancel both children before re-raising so no
  ``Task was destroyed but it is pending!`` warnings escape on
  generator teardown. The pre-existing `async with bus.subscribe()`
  finally block continues to free the subscriber slot on exit.
- **New `tests/unit/test_routes_sse.py`** — three coroutine-level
  assertions covering: disconnect-mid-stream-with-idle-queue terminates
  cleanly, no leaked asyncio tasks across stream entry/exit, and the
  bus-close sentinel emits a final `done` event before the generator
  returns.
- **New `tests/integration/test_real_ffmpeg.py`** — exercises the real
  ffmpeg binary against synthesized black-frame PNGs and silent WAVs for
  `n = 1`, `n = 3`, and `n = 10` shots. ffprobe extracts the output
  duration and asserts it matches the closed-form
  `sum(durations) - (n-1) * fade_seconds` value within 200 ms. The
  module fails (not skips) when ffmpeg is absent so CI cannot silently
  pass on a misconfigured runner.
- **New `real_ffmpeg` pytest marker** registered in
  `[tool.pytest.ini_options].markers` so the integration tests can be
  invoked with `pytest -m real_ffmpeg` and skipped from the default
  unit-test sweep with `-m "not real_ffmpeg"`.
- **v0.1.0 release candidate.** All AUDIT.md HIGH-severity bugs in the
  pivot scope are now closed; tag pending the phase 4 human checkpoint.
