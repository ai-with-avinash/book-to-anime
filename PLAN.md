# Pivot Plan: BookToAnime → StudyPanels (PDF → STEM Motion-Comic Explainer)

## Context

Current product positions itself as "PDF → anime explainer". Honest assessment:

- Output is **SDXL stills + Ken-Burns + xfade + Kokoro narration**, not anime motion. Anime audience will reject quality ceiling.
- "PDF → anime" is solution looking for problem. NotebookLM beats it on convenience for free.
- Hosted-LLM costs ($0.05–$8/book) + Replicate lipsync are paid; user constraint is **strictly free**.
- For STEM textbooks, **SDXL-generated "physics visualizations" hallucinate equations, diagrams, wiring**. Pipeline already extracts real figures from the PDF but never uses them as visuals — wastes highest-fidelity asset it has.

**Pivot:** rebrand + refocus as **"PDF → STEM motion-comic study explainer"**. Real extracted figures as primary visual (correct, not hallucinated). Comic-panel framing for honesty (stills + motion, not animation). Ollama + Kokoro + local SDXL for fully-free runtime.

**Intended outcome:**
- Single free local pipeline. No paid APIs in default path.
- STEM-correct visuals (real figures > hallucinated illustrations).
- Honest framing: "motion comic" not "anime".
- Reuse ~75% of existing architecture (parsing, manifest, resume, SSE, providers, assembler all survive).

---

## Execution model

- **Repo-level `PLAN.md`** created at `/Users/avinash/conductor/workspaces/book-to-anime/melbourne/PLAN.md` mirroring this plan (phases + gates).
- Each phase = one `general-purpose` agent task. Tight scope, file allowlist, exit-gate command list.
- **Branch-per-phase strategy.** Each phase runs on its own branch `pivot/phase-N-<name>`. Merge sequentially after gate green + (where applicable) human checkpoint. Easy rollback.
- After agent completes, **gate runs in parent session** (not agent). Green required before next phase. Parent also runs `git diff --name-only` and asserts changed-file set ⊆ allowlist.
- **Hard halt on red gate.** No auto-fix loop. Spawn second agent with diff + failure report; if still red, halt to user.
- **Two human-eyeball checkpoints** (phase 3, phase 4) ~10–15 min each. Non-negotiable.

---

## Cross-cutting decisions (apply to all phases)

### CC-1. Manifest schema versioning
Add `manifest_schema_version: int` field to `JobManifest`. Phase 1 sets to `2` (current implicit = `1`). Loader refuses mismatch with clear msg + `booktoanime resume` does not auto-upgrade — user instructed to delete `data_dir/jobs/` or run `booktoanime migrate` (deferred). Phase 1 step 0 nukes any existing `data_dir/jobs/` after confirmation prompt.

### CC-2. Hatch wheel asset packaging
Hatchling default packaging does **not** include non-Python files under `packages=`. Phase 1 re-adds **narrow** `[tool.hatch.build.targets.wheel.force-include]` for `src/booktoanime/web/static/` and `src/booktoanime/web/templates/` (not the whole package — avoids the duplicate-name warning AUDIT.md flagged). Phase 3 adds `src/booktoanime/web/static/fonts/` to the same include block. Gate verifies wheel contains no duplicate names.

### CC-3. Allowlist enforcement
Each phase prompt lists allowed paths + forbidden paths. Parent session runs `git diff --name-only origin/main...HEAD` after agent finishes and aborts merge if any changed file outside allowlist.

### CC-4. CHANGELOG.md
Created phase 1. Each phase appends a section. No last-minute changelog drift.

### CC-5. New `errors.py` types
- `RenderError` — panel composer / figure compositing failures
- `ManifestSchemaMismatch` — loader rejects old manifest version

Added in phase 1.

### CC-6. Free-stack preflight
New CLI command `booktoanime check` (phase 1) — probes Ollama `/api/tags`, asserts configured model present, asserts Kokoro weight file in cache, asserts ffmpeg + tesseract binaries on PATH. Auto-run inside `booktoanime run` unless `--skip-preflight`.

---

## Scope summary

| Domain | Action |
|---|---|
| Parsing, segmentation, manifest core, event bus, SSE, registry, ABCs, ffmpeg assembler, CLI, dotenv | **KEEP** |
| `PERSONA_SEEDING` stage | **REPURPOSE** → `STYLE_SEEDING` (style anchor, no character) |
| Visual stage prompts + style presets | **REWRITE** (anime → comic/technical) |
| Storyboard prompt template + `Shot.visual_kind` | **REWRITE / EXTEND** |
| Image renderer | **EXTEND** — dispatch on `visual_kind` |
| Web UI form + labels + palette + brand | **REWRITE** (phase 1) |
| Config + schema field names | **RENAME** (`anime_style` → `panel_style`) |
| README + NOTICE + CHANGELOG | **REWRITE** |
| Replicate / SadTalker / passthrough lipsync | **DELETE** |
| `MOUTH_ANIMATION` stage | **DELETE** |
| `narrator_persona.py` | **DELETE** (phase 1) |

---

## Phase 1 — Strip + rename + rebrand (Week 1)

**Goal:** remove all lipsync/mouth-animation + character-narrator code, rename `anime_style` → `panel_style`, rebrand UI, add schema versioning + preflight, set foundations.

### Agent scope
- **Allowed:** `src/booktoanime/`, `tests/`, `config.example.yaml`, `pyproject.toml`, `README.md`, `NOTICE`, `Dockerfile`, `CHANGELOG.md` (NEW)
- **Forbidden:** any `_STYLE_FRAGMENTS` value edit (phase 2), any storyboard prompt rewrite (phase 2), any new `panel_composer.py` (phase 3), any `video_assembler.py` xfade math change (phase 4)
- Phase 1 **may** touch `video_assembler.py` only to strip `MouthIndex` and `lipsync.enabled` branches (per fix C3).

### Tasks
1. **Nuke existing job data.** Step 0: print path of `data_dir/jobs/` (resolved via platformdirs), delete or move to `data_dir/jobs.pre-v0.1.bak/`. Document in CHANGELOG.
2. Delete `src/booktoanime/providers/lipsync/` (entire dir).
3. Delete `src/booktoanime/pipeline/mouth_animator.py`.
4. Delete `src/booktoanime/pipeline/narrator_persona.py` (per M6). Orchestrator `PERSONA_SEEDING` stage becomes a stub that logs `INFO` + advances — replaced fully in phase 2 with `STYLE_SEEDING`.
5. Delete `Stage.MOUTH_ANIMATION` from `pipeline/stages.py`; update `STAGE_ORDER` (drop entry, keep `PERSONA_SEEDING` for now).
6. Delete `MouthIndex` from `pipeline/artifacts.py`.
7. Delete `LipSyncProvider` ABC + `AnimatedShot` dataclass from `providers/base.py`.
8. Drop `JobConfig.lipsync` field from `pipeline/manifest.py`. Drop `PipelineDependencies.lipsync` from `orchestrator.py`. Drop `_run_mouth_animation` method and call site.
9. Strip `pipeline/video_assembler.py`: remove `MouthIndex` import + the `if manifest.config.lipsync.enabled and mouth_index_path.is_file()` branch. Assembler always consumes static `images/shot_*.png`. **No xfade math change here** — phase 4.
10. **Add `manifest_schema_version: int = 2`** to `JobManifest`. Loader (`JobManifest.from_path`) raises `ManifestSchemaMismatch` on `version != 2`.
11. **Add `errors.py` types:** `RenderError`, `ManifestSchemaMismatch`.
12. Rename `anime_style` → `panel_style` across:
    - `pipeline/manifest.py` (default `"clean-linework"` — placeholder name, real fragments in phase 2)
    - `api/schemas.py`
    - `api/routes_jobs.py`
    - `pipeline/image_renderer.py` config field
    - `pipeline/storyboard.py` config field (prompt rewrite is phase 2)
    - `web/templates/index.html` form field name + label (single placeholder option only)
    - `config.example.yaml`
13. Drop `lipsync:` block from `config.example.yaml`.
14. **pyproject cleanup:**
    - Drop `ulid-py` dependency (per C4).
    - Drop `xtts` extra.
    - Drop `together`, `fireworks`, `mistral` extras (AUDIT SPECULATIVE finding — vendor adapters reuse OpenAI-compat HTTP).
    - Drop matching libs from `all-providers`.
    - **Add narrow `[tool.hatch.build.targets.wheel.force-include]`** (per CC-2):
      ```toml
      [tool.hatch.build.targets.wheel.force-include]
      "src/booktoanime/web/static" = "booktoanime/web/static"
      "src/booktoanime/web/templates" = "booktoanime/web/templates"
      ```
    - Bump `version = "0.1.0"` in `pyproject.toml` and `src/booktoanime/__init__.py`.
15. Replace `_generate_job_id` body with `secrets.token_urlsafe(13)`; drop ULID docstring (per C4).
16. **New CLI command `booktoanime check`** (per CC-6). Subcommand of Typer app. Probes:
    - Ollama `/api/tags` reachable + configured model present (via httpx)
    - Kokoro weight file in `<data_dir>/models/kokoro/` (or document where Kokoro caches)
    - `ffmpeg -version` exit 0
    - `tesseract --version` exit 0
    Auto-invoked at start of `booktoanime run` unless `--skip-preflight`.
17. **UI rebrand + palette (per M8):**
    - `web/templates/_layout.html` brand title → "StudyPanels"
    - `web/static/app.css` palette → academic slate/teal/cream (drop saturated anime palette)
18. **CHANGELOG.md** (NEW, per M7 / CC-4):
    - v0.1.0 entry — list breaking changes (rename, drop lipsync, drop anime, manifest schema bump, data dir reset).
19. **README rewrite (scope: strip + rebrand only, per M5):**
    - Title + headline → "StudyPanels: PDF → STEM motion-comic study explainer"
    - Three-paths table → "Local Ollama" promoted as recommended; hosted moved below as "optional fallback"
    - Drop lipsync section
    - Drop Replicate from acknowledgments
    - **Defer** panel-style section to phase 2 (placeholder note: "Style presets — see v0.1 panel styles section below (filled in phase 2)")
20. `NOTICE` — remove Replicate, SadTalker, XTTS attributions.
21. **Delete tests:** `tests/unit/test_mouth_animator*.py`, `tests/unit/test_*lipsync*.py`, `tests/unit/test_narrator_persona.py` (if exists).
22. **Update tests:** `test_orchestrator.py` (drop MOUTH_ANIMATION), `test_storyboard.py` + `test_image_renderer.py` (rename field), `test_manifest.py` (assert schema version + raise on mismatch).
23. **New test:** `tests/unit/test_preflight.py` — mock Ollama + binaries; assert preflight passes/fails correctly.
24. **Branch:** all work on `pivot/phase-1-strip-rename`.

### Exit gate (parent session, must pass)
```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check src tests
.venv/bin/python -m mypy --strict src
.venv/bin/python -c "from booktoanime.pipeline.stages import Stage; assert not hasattr(Stage, 'MOUTH_ANIMATION')"
.venv/bin/python -c "from booktoanime.pipeline.manifest import JobManifest; assert hasattr(JobManifest, 'model_fields') and 'manifest_schema_version' in JobManifest.model_fields"
.venv/bin/python -c "from booktoanime.errors import RenderError, ManifestSchemaMismatch"
# scrub check (excludes docs)
! grep -r -E "anime_style|MouthIndex|LipSyncProvider|narrator_persona|ulid" src tests --include="*.py" | grep -v "test_legacy"
# wheel build, zero duplicate-name warnings, font dir absent (phase 3 adds)
.venv/bin/python -m build --wheel --outdir /tmp/btoa-build 2>&1 | tee /tmp/btoa-build.log
! grep -i "duplicate name" /tmp/btoa-build.log
# allowlist enforcement
git diff --name-only origin/main...HEAD | grep -v -E '^(src/booktoanime/|tests/|config\.example\.yaml|pyproject\.toml|README\.md|NOTICE|Dockerfile|CHANGELOG\.md)$' && exit 1 || true
```

### Deliverable
- All static gates green, scrub clean.
- LOC delta roughly −1000 to −1400.
- Branch ready to merge into `main`.

---

## Phase 2 — Visual stage rewrite + STYLE_SEEDING (Week 2)

**Goal:** anime → STEM/comic prompt fragments. New `STYLE_SEEDING` stage produces one style anchor per job for IP-Adapter consistency on SDXL fallback shots. Storyboard sets `Shot.visual_kind`. Renderer still treats all kinds as SDXL render — figure-first dispatch in phase 3.

### Agent scope
- **Allowed:** `src/booktoanime/providers/visual/sdxl_diffusers.py`, `src/booktoanime/pipeline/storyboard.py`, `src/booktoanime/pipeline/image_renderer.py`, `src/booktoanime/pipeline/summarizer.py`, `src/booktoanime/pipeline/artifacts.py`, `src/booktoanime/pipeline/stages.py`, `src/booktoanime/pipeline/orchestrator.py`, `src/booktoanime/pipeline/style_seeder.py` (NEW — replaces deleted `narrator_persona.py`), `src/booktoanime/api/schemas.py`, `src/booktoanime/api/routes_jobs.py`, `src/booktoanime/web/templates/index.html`, `config.example.yaml`, `README.md`, `CHANGELOG.md`, `tests/unit/test_storyboard.py`, `tests/unit/test_image_renderer.py`, `tests/unit/test_visual_*.py`, `tests/unit/test_style_seeder.py` (NEW)
- **Forbidden:** creating `panel_composer.py`; image_renderer dispatch on `visual_kind`; `video_assembler.py` edits.

### Tasks
1. **Replace `_STYLE_FRAGMENTS`** in `sdxl_diffusers.py:46-63`:
   - `clean-linework`: `"clean line art illustration, minimal color palette, technical diagram aesthetic, neutral background, sharp edges, no shading"`
   - `chalkboard-sketch`: `"chalkboard illustration, white chalk on dark green background, hand-drawn diagram style, classroom aesthetic"`
   - `watercolor-technical`: `"soft watercolor technical illustration, muted earth palette, hand-painted scientific drawing, light pencil outline"`
   - `flat-vector-infographic`: `"flat vector infographic, bold geometric shapes, limited four-color palette, modern educational design"`
2. Replace `_PERSONA_BASE_PROMPT` with `_STYLE_ANCHOR_BASE = "abstract style reference, no subject, neutral composition, single textural sample"` — used only by `STYLE_SEEDING`.
3. **New module `pipeline/style_seeder.py`** (per H1):
   - `class StyleSeeder` with config `(panel_style, seed)`
   - Async `seed(job_dir, visual_provider) → StyleReference(file: JobRelPath, panel_style: str, seed: int)`
   - Calls `visual_provider.prepare(VisualSeedRequest(prompt=f"{_STYLE_ANCHOR_BASE}, {style_fragment}", seed=seed))`
   - Saves reference under `job_dir/style/style__<panel_style>__<seed>.png`
   - Cache key: `(panel_style, seed)` — re-use across job runs and across jobs with same key.
4. **Wire `STYLE_SEEDING` into orchestrator** replacing the phase-1 stub `PERSONA_SEEDING`:
   - Rename `Stage.PERSONA_SEEDING` → `Stage.STYLE_SEEDING` in `stages.py`.
   - Orchestrator method `_run_style_seeding(manifest, job_dir)` calls `StyleSeeder`. Result stored in `manifest.artifacts.style_reference` (new field).
   - On resume, idempotent — if file exists with matching cache key, skip.
5. **Update `SDXLDiffusersProvider`:**
   - `prepare()` method now accepts a style-only request (no character base prompt). Returns image keyed on `(panel_style, seed)`.
   - `generate()` accepts optional `ip_adapter_image` parameter; orchestrator passes the style reference for `VisualKind.ILLUSTRATION` shots only.
   - For `VisualKind.FIGURE` and `VisualKind.TITLE_CARD`, image_renderer will short-circuit before calling provider (phase 3) — phase 2 keeps current SDXL-all-shots behavior.
6. **Add `VisualKind` enum** to `pipeline/artifacts.py`: `FIGURE`, `ILLUSTRATION`, `TITLE_CARD`.
7. **Extend `Shot`** in `pipeline/artifacts.py`:
   - Add `visual_kind: VisualKind = VisualKind.ILLUSTRATION`
   - Add `figure_id: str | None = None` (per H2 — stable ID, references `ExtractedImage.image_id`, not list index)
8. **Rewrite `storyboard.py:100-107` `_image_prompt()`:**
   - Template: `f"educational illustration of {topic.title.lower().strip()}: {focus}, {panel_style_fragment}"`
   - Topic title sentence-clean + truncate to 60 chars
   - `Shot.visual_kind` assignment:
     - First shot of topic AND topic has >2 shots → `TITLE_CARD`
     - Topic has `image_refs` not yet consumed → `FIGURE`, set `figure_id = image_ref.image_id` (per H2)
     - Else → `ILLUSTRATION`
     - Topics with ≤2 shots: no title card (per L4)
9. **Update `image_renderer.py`:**
   - Add `visual_kind` and `figure_id` to `index.json` shot record (per H4 — reconciler will use in phase 3).
   - Phase 2 renderer still calls SDXL for all kinds; phase 3 adds dispatch.
   - Reconciler invalidates shot when on-disk `index.json.visual_kind` mismatches current storyboard's `visual_kind` (per H4).
10. **Update `summarizer.py:145-148` system prompt** (per H8):
    - `"You are condensing one chapter of a document for a narrated explainer video. Use concrete examples and clear definitions. Match the source's domain — don't force STEM framing on non-technical content."`
11. **Update `web/templates/index.html`** dropdown — 4 real panel-style options.
12. **Update `config.example.yaml`** — list new style names in comment.
13. **README** — add panel-styles section (the placeholder from phase 1).
14. **CHANGELOG** — append phase-2 entry.
15. **Tests:**
    - `test_storyboard.py`: assert new prompt format; `visual_kind` assignment for various topic configs (with/without image_refs, ≤2 shots edge case)
    - `test_image_renderer.py`: assert `index.json` records `visual_kind` + `figure_id`; reconciler invalidates on kind mismatch
    - `test_style_seeder.py` (NEW): cache key correctness, idempotent resume, file path stability
    - Scrub assert: no `shounen`, `shoujo`, `seinen`, `chibi`, `narrator persona`, `anime explainer scene` in source
16. **Branch:** `pivot/phase-2-visual`.

### Exit gate
```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check src tests
.venv/bin/python -m mypy --strict src
.venv/bin/python -c "from booktoanime.pipeline.artifacts import VisualKind; assert {'FIGURE','ILLUSTRATION','TITLE_CARD'} <= {k.name for k in VisualKind}"
.venv/bin/python -c "from booktoanime.pipeline.style_seeder import StyleSeeder"
.venv/bin/python -c "from booktoanime.pipeline.stages import Stage; assert hasattr(Stage,'STYLE_SEEDING') and not hasattr(Stage,'PERSONA_SEEDING')"
! grep -r -i -E "shounen|shoujo|seinen|chibi|anime explainer scene|narrator persona" src --include="*.py"
git diff --name-only origin/main...HEAD | grep -v -E '^<allowlist>$' && exit 1 || true   # parent fills <allowlist>
```

### Deliverable
- Anime literals scrubbed from runtime code.
- `STYLE_SEEDING` stage producing per-job style anchor.
- `VisualKind` + `figure_id` plumbed through storyboard → renderer (dispatch logic comes phase 3).

---

## Phase 3 — Figure-first rendering (Week 3) — **HUMAN CHECKPOINT 1**

**Goal:** real extracted figures rendered as comic panels (Pillow). SDXL only as fallback. Title cards via Pillow. Aspect-aware layout.

### Agent scope
- **Allowed:** `src/booktoanime/pipeline/panel_composer.py` (NEW), `src/booktoanime/pipeline/image_renderer.py`, `src/booktoanime/web/static/fonts/` (NEW dir), `pyproject.toml` (font include), `tests/unit/test_panel_composer.py` (NEW), `tests/unit/test_image_renderer.py`, `NOTICE`, `CHANGELOG.md`
- **Forbidden:** AUDIT.md ffmpeg / SSE bug fixes (phase 4), UI palette tweaks (already in phase 1).

### Tasks
1. **Bundle font:** `src/booktoanime/web/static/fonts/Inter-Regular.ttf` + `Inter-Bold.ttf` (OFL-1.1, ~300KB each).
2. **Update `pyproject.toml`** force-include to add `fonts/` subdir (extend block from phase 1):
   ```toml
   "src/booktoanime/web/static" = "booktoanime/web/static"
   ```
   (existing rule already covers `fonts/` sub-tree)
3. **Update `NOTICE`** — attribute Inter font (OFL-1.1).
4. **Implement `pipeline/panel_composer.py`:**
   - `compose_figure_panel(figure_path, caption, title, panel_style, target_size=(1920,1080)) → PIL.Image.Image`
     - **Aspect-aware layout (per M2):**
       - Source aspect ≥ 1.2 (landscape) → **bottom strip** layout. Figure area 1920×800, caption strip 1920×280.
       - Source aspect < 1.2 (portrait or square) → **side-panel** layout. Figure area 1080×1080 left, caption panel 840×1080 right.
     - **Fonts (per M3):** title 36pt bold, caption 22pt regular. Inter via Pillow `ImageFont.truetype`.
     - Title truncated to 60 chars + sentence-clean (strip trailing periods, normalize whitespace — per L5).
     - Caption wrapped, max 4 lines, ellipsis on overflow.
     - BG palette per `panel_style`: white (clean-linework), dark green #1a3a2e (chalkboard), cream #f5ebd6 (watercolor), light gray #e8e8e8 (flat-vector).
     - Letterbox figure inside its panel preserving aspect.
     - Raises `RenderError` on font load failure (added in phase 1).
   - `compose_title_card(title, subtitle, panel_style, target_size) → PIL.Image.Image`
     - Centered title 56pt bold + subtitle 28pt on style-themed BG.
   - Helper `_sentence_clean(text) → str` — strip trailing periods, normalize whitespace, truncate.
5. **Update `image_renderer.py`:**
   - **Dispatch on `shot.visual_kind`:**
     - `FIGURE`: load `ExtractedImage` by `shot.figure_id` (per H2 — stable lookup); call `compose_figure_panel`; save; **no SDXL call**.
     - `TITLE_CARD`: call `compose_title_card`; save; **no SDXL call**.
     - `ILLUSTRATION`: existing SDXL render path, passing `style_reference` as IP-Adapter image (per H1).
   - **Split semaphores (per H3):**
     - `_sdxl_semaphore` = current concurrency cap (GPU-bound, profile-driven).
     - `_compose_semaphore` = `min(8, cpu_count())` (CPU-bound, Pillow).
     - Dispatch picks correct sem before awaiting.
   - **Telemetry (per L3):** at end of IMAGES stage, emit `ProgressEvent(kind=INFO, message=f"figure_shots={n_fig} illustration_shots={n_ill} title_cards={n_tc}")`.
6. **Caption-strip + SRT relationship (per H7):**
   - Caption strip text = `Shot.title` (≤60 chars, sentence-clean). Short callout, not full narration.
   - `output.srt` continues to carry full narration text (unchanged from existing `srt_sidecar.py`).
7. **Tests (per M1 — no pixel hashes):**
   - `test_panel_composer.py`:
     - Aspect-fit landscape vs portrait dispatch (assert output dimensions, region sizes via crop bbox)
     - Caption truncation (assert text content via `ImageDraw.textbbox` or render-then-OCR — defer OCR, just assert text passed through font draw API was truncated upstream)
     - Font load failure raises `RenderError` (mock `truetype` to raise)
     - Palette correctness — assert alpha-channel histogram bucket counts per style (not pixel hashes)
   - `test_image_renderer.py`:
     - Dispatch on `visual_kind` (assert SDXL mock NOT called for FIGURE/TITLE_CARD)
     - `figure_id` → `ExtractedImage` lookup; missing ID raises `RenderError` (not `KeyError`)
     - Reconciler invalidates shot when `index.json.visual_kind` mismatches storyboard (per H4)
     - Concurrency: spawn 20 mock shots, assert SDXL sem cap honored, Pillow sem cap honored separately
   - Snapshot test removed (was pixel-hash) — replaced with dim + histogram asserts.
8. **CHANGELOG** — append phase-3 entry.
9. **Branch:** `pivot/phase-3-figure-first`.

### Exit gate (autonomous)
```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check src tests
.venv/bin/python -m mypy --strict src
test -f src/booktoanime/web/static/fonts/Inter-Regular.ttf
test -f src/booktoanime/web/static/fonts/Inter-Bold.ttf
.venv/bin/python -c "from booktoanime.pipeline.panel_composer import compose_figure_panel, compose_title_card"
# real render smoke (per L8)
.venv/bin/python -c "from booktoanime.pipeline.panel_composer import compose_title_card; compose_title_card('Smoke test', 'subtitle', 'clean-linework', (320,180)).save('/tmp/smoke_title.png')"
test -f /tmp/smoke_title.png
.venv/bin/python -m build --wheel --outdir /tmp/btoa-build 2>&1 | tee /tmp/btoa-build.log
unzip -l /tmp/btoa-build/*.whl | grep Inter-Regular.ttf
git diff --name-only origin/main...HEAD | grep -v -E '^<allowlist>$' && exit 1 || true
```

### Exit gate (HUMAN CHECKPOINT 1 — ~10 min)
1. Agent-prepared smoke script: bundled fixture PDF (8-page open-source PDF with figures) → run pipeline with Kokoro + SDXL + Ollama.
2. **You watch `output.mp4`:**
   - Real figures show as figure panels (not SDXL-generated)
   - Caption strip readable at 1080p and on phone preview
   - Portrait figures use side-panel layout; landscape use bottom strip
   - Title cards bookend topics (only for topics >2 shots)
   - SDXL fallback only on shots without figures, style-consistent (IP-Adapter using style anchor)
3. **Decide:** PASS → merge `pivot/phase-3-figure-first` → main; spawn phase 4. FAIL → spec fixes, agent re-runs.

### Deliverable
- Hybrid figure + SDXL pipeline functional.
- One reviewed MP4 with real figures.
- Wheel contains font assets.

---

## Phase 4 — Polish + AUDIT blockers (Week 4) — **HUMAN CHECKPOINT 2**

**Goal:** fix AUDIT.md HIGH ffmpeg + SSE bugs. Real end-to-end on 3 PDFs. Tag v0.1.0.

### Agent scope
- **Allowed:** `src/booktoanime/pipeline/video_assembler.py`, `src/booktoanime/api/routes_sse.py`, `tests/unit/test_video_assembler.py`, `tests/unit/test_routes_sse.py` (NEW), `tests/integration/__init__.py` (NEW), `tests/integration/test_real_ffmpeg.py` (NEW), `pyproject.toml` (pytest markers), `CHANGELOG.md`
- **Forbidden:** any pipeline-stage logic change; storyboard / prompt changes; new modules outside the above.

### Tasks
1. **Fix `video_assembler.py:248-249` n=1** (AUDIT HIGH 1):
   - Special-case single shot: skip concat filter entirely. Audio map = `-map 1:a` direct.
2. **Fix `video_assembler.py:251-271` xfade offset math** (AUDIT HIGH 3):
   - Track `previous_output_duration` separately from raw cumulative input duration.
   - Per xfade k: `offset_k = previous_output_duration - fade_k`. Then `previous_output_duration += durations[shot_k] - fade_k`.
   - Unit test with non-uniform fades [0.5, 1.0, 0.3] across 5 shots — assert computed offsets match closed-form math.
3. **Stream stderr safely** (AUDIT MEDIUM, per M4):
   ```python
   log_handle = await asyncio.to_thread(open, log_path, "wb")
   try:
       process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.DEVNULL, stderr=log_handle)
       returncode = await process.wait()
   finally:
       await asyncio.to_thread(log_handle.close)
   ```
4. **Delete dead `offsets` block** (`video_assembler.py:243-246`).
5. **Fix `routes_sse.py:33-49` disconnect race** (AUDIT HIGH 5, per C6):
   ```python
   queue_task = asyncio.create_task(queue.get())
   disc_task = asyncio.create_task(request.is_disconnected())
   done, pending = await asyncio.wait({queue_task, disc_task}, return_when=asyncio.FIRST_COMPLETED)
   for t in pending:
       t.cancel()
   if disc_task in done and disc_task.result():
       break
   event = queue_task.result()
   ```
   Plus loop-level cancellation cleanup of any orphaned queue task on stream exit.
6. **New `tests/unit/test_routes_sse.py`** — simulate disconnect mid-stream, assert subscriber removed, no task left pending.
7. **New `tests/integration/test_real_ffmpeg.py`** — gated on `pytest.mark.real_ffmpeg`:
   - Synthesize black-frame PNGs (1s, 2s, 3s durations) + silent WAVs
   - Run assembler for n=1, n=3, n=10
   - Assert output MP4 duration matches expected ±200ms via `ffprobe`
   - **Gate behavior (per M10):** at test setup, `shutil.which("ffmpeg")` — if missing, `pytest.fail(...)`, NOT skip. Forces CI to install ffmpeg.
8. **Update `pyproject.toml`** — add `real_ffmpeg` marker.
9. **CHANGELOG** — append phase-4 entry, mark v0.1.0 release candidate.
10. **Branch:** `pivot/phase-4-bugs-polish`.

### Exit gate (autonomous)
```bash
.venv/bin/python -m pytest tests/unit tests/integration -m "not real_ffmpeg"
.venv/bin/ruff check src tests
.venv/bin/python -m mypy --strict src
which ffmpeg && .venv/bin/python -m pytest tests/integration -m real_ffmpeg
git diff --name-only origin/main...HEAD | grep -v -E '^<allowlist>$' && exit 1 || true
```

### Exit gate (HUMAN CHECKPOINT 2 — ~15 min)
1. Run pipeline on 3 PDFs:
   - 8-page STEM textbook chapter (figure-rich)
   - 30-page lecture notes (text-heavy, few figures)
   - 15-page paper (mixed text + figures + tables)
2. **You watch each `output.mp4`:**
   - No frozen segments (xfade math fix held)
   - n=1 edge case (single short topic) plays correctly
   - Audio + video stay in sync
   - SSE bar reaches 100% in browser; tail orchestrator log for absence of "subscriber leaked"
3. **Network capture:** `tcpdump` or `nettop` confirms zero paid endpoints contacted (no `replicate.com`, `api.anthropic.com`, `api.openai.com`, etc.) during run.
4. **Decide:** PASS → merge `pivot/phase-4-bugs-polish` → main; tag `v0.1.0`. FAIL → bug list back to phase 4 agent.

### Deliverable
- All AUDIT.md HIGH bugs closed.
- 3 reviewed MP4s.
- v0.1.0 tagged.

---

## Critical files summary

| Phase | Path | Action |
|---|---|---|
| 1 | `providers/lipsync/` (dir) | DELETE |
| 1 | `pipeline/mouth_animator.py` | DELETE |
| 1 | `pipeline/narrator_persona.py` | DELETE |
| 1 | `pipeline/manifest.py` | rename, drop lipsync, +schema_version |
| 1 | `api/schemas.py`, `api/routes_jobs.py` | rename |
| 1 | `web/templates/index.html`, `_layout.html`, `web/static/app.css` | rename field, rebrand, palette |
| 1 | `config.example.yaml` | rename, drop lipsync |
| 1 | `pyproject.toml` | dep cleanup, version, hatch include |
| 1 | `errors.py` | add `RenderError`, `ManifestSchemaMismatch` |
| 1 | `cli.py` | add `booktoanime check` + auto-preflight |
| 1 | `pipeline/video_assembler.py` | strip MouthIndex only (no xfade change) |
| 1 | `README.md`, `NOTICE`, `CHANGELOG.md` | strip + rebrand + new |
| 2 | `providers/visual/sdxl_diffusers.py` | new style fragments, style anchor |
| 2 | `pipeline/style_seeder.py` | NEW |
| 2 | `pipeline/stages.py`, `orchestrator.py` | STYLE_SEEDING stage |
| 2 | `pipeline/storyboard.py` | new prompt, VisualKind/figure_id |
| 2 | `pipeline/artifacts.py` | VisualKind enum, Shot fields |
| 2 | `pipeline/summarizer.py` | neutral prompt |
| 2 | `pipeline/image_renderer.py` | reconciler + visual_kind record |
| 2 | `README.md`, `CHANGELOG.md` | panel-style section, append |
| 3 | `pipeline/panel_composer.py` | NEW |
| 3 | `pipeline/image_renderer.py` | dispatch + split semaphores + telemetry |
| 3 | `web/static/fonts/Inter-{Regular,Bold}.ttf` | NEW assets |
| 3 | `pyproject.toml`, `NOTICE`, `CHANGELOG.md` | font attribution + include |
| 4 | `pipeline/video_assembler.py` | bugfix n=1 + xfade math + stderr stream |
| 4 | `api/routes_sse.py` | bugfix disconnect race |
| 4 | `tests/integration/test_real_ffmpeg.py` | NEW |
| 4 | `pyproject.toml` | pytest marker |
| 4 | `CHANGELOG.md` | v0.1.0 RC |

---

## Reusable functions (no rewrites)

- `parsing/pdf_parser.PDFParser` — figure + caption extraction
- `pipeline/topic_segmenter.TopicSegmenter` — chapter detection
- `pipeline/manifest.JobManifest` — atomic save/load (+ phase 1 schema version field)
- `pipeline/events.ProgressEventBus` — SSE pubsub
- `pipeline/video_assembler.VideoAssembler` — Ken Burns + xfade + audio (post phase-4 fixes)
- `providers/audio/kokoro.KokoroAudioProvider` — free TTS
- `providers/language/openai_compatible.OpenAICompatibleProvider` — Ollama path
- `providers/visual/sdxl_diffusers.SDXLDiffusersProvider._render_to` — keep core path
- `state/job_repo.JobRepository` — SQLite tracking
- `providers/_retry` — generic retry
- `pipeline/srt_sidecar` — keep, full-narration SRT continues independent of caption strip

---

## Agent orchestration spec

```
Per phase:
  1. Parent creates branch pivot/phase-N-<name> from main.
  2. Parent spawns general-purpose agent. Prompt:
     - Context paragraph + path to this plan + path to repo PLAN.md
     - Phase N scope: file allowlist + forbidden list (verbatim from plan section)
     - Tasks (numbered, verbatim)
     - Exit gate command list
     - Rules:
       * Do NOT proceed if gate fails; report blockers + stop
       * Do NOT touch files outside allowlist
       * Commit at end of phase only after agent's local gate run green
       * No destructive git operations
     - Run: foreground
  3. Post-agent (parent):
     - Re-run gate commands independently
     - Run allowlist enforcement: git diff --name-only origin/main...HEAD ⊆ allowlist
     - If gate red OR allowlist violated:
       * Spawn second agent with diff + failure report, re-run
       * If still red: halt, surface to user
     - If green AND phase has human checkpoint: pause, await user verdict
     - If green AND no human checkpoint OR user PASS: merge branch into main, proceed to next phase
```

---

## Risk register

| Risk | Mitigation |
|---|---|
| Old manifest.json on disk breaks loader | Phase 1 step 0 nukes/moves `jobs/`; schema_version raises clear error otherwise |
| Hatchling wheel drops font assets | Phase 1 re-adds narrow force-include; phase 3 gate verifies `unzip -l` |
| Extracted figures low-res | Aspect-aware composer (M2); fallback to SDXL when source < 256px (renderer guard) |
| Real textbooks copyrighted | README disclaimer prominent: user owns content responsibility |
| Kokoro English-only | Document; multilingual = v0.2 |
| SDXL fallback drift across shots | STYLE_SEEDING + IP-Adapter style ref (H1) |
| Resume picks stale visual_kind | Reconciler invalidates on kind mismatch (H4) |
| Agent edits outside allowlist | Parent enforces via `git diff --name-only`; halts on violation |
| Agent claims gate green when it isn't | Parent re-runs gates independently |
| Phase 3 visual quality regression | Human checkpoint 1 |
| Phase 4 xfade math regression | Real-ffmpeg integration test + human checkpoint 2 |
| Ollama not running locally | Preflight check (CC-6) fails fast with install hint |
| ffmpeg / tesseract missing | Preflight check + phase-4 integration test FAILS (not skips) on missing ffmpeg |

---

## Verification (final, post-phase-4)

1. **Static gates:** `pytest && ruff check src tests && mypy --strict src` green
2. **Wheel:** zero duplicate-name warnings, fonts present
3. **CLI:** `booktoanime --help` + `booktoanime version` returns `0.1.0`; `booktoanime check` passes on a configured free-stack box
4. **Free-stack run:** Ollama + Kokoro + SDXL local — 20-page STEM PDF → MP4 produced end-to-end
5. **No paid network calls:** tcpdump / nettop confirms zero hits to Replicate / Anthropic / OpenAI / Groq during run
6. **Bug regressions:** real-ffmpeg integration test green for n=1, n=3, n=10
7. **SSE soak:** 10-min browser session with mid-disconnect — no leaked subscriber tasks
8. **Tag:** `git tag v0.1.0 && git push --tags` (manual)

---

## Out of scope (defer to v0.2+)

- Full motion (Wan 2.1 / AnimateDiff)
- Branded B2B explainers
- Per-image bbox caption matching
- Multilingual narration
- KaTeX/MathJax equation rendering
- Auto-domain detection for summarizer framing
- SDXL transparent/neutral-BG output for clean panel-composer wrapping (chalkboard inconsistency, H5 polish)
- Manifest schema migration tool (`booktoanime migrate`)
