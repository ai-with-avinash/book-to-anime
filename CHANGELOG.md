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
