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
