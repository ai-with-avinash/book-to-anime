# BookToAnime

Convert a PDF book into an anime-style narrated MP4 explainer video, locally.
Open-source under Apache-2.0. No proprietary runtime required, no telemetry,
all artifacts live in your own data directory.

```
$ booktoanime           # opens http://127.0.0.1:8765 in your browser
```

Pluggable across language, audio, and visual providers — pick a hosted LLM
(Anthropic, OpenAI, Gemini, Groq, Together, Fireworks, DeepSeek, Mistral) or
run fully local against any OpenAI-compatible endpoint (Ollama, vLLM, LM
Studio, llama.cpp). Default install is small; heavy ML deps (torch,
diffusers) live behind opt-in install extras.

> **Status:** v0.0.1 — pipeline end-to-end runs in tests; large-PDF + GPU
> tuning still in progress. See [Roadmap](#roadmap).

---

## Three recommended starter paths

| Path | Provider | Rough per-book LLM cost\* |
|---|---|---|
| Cheap-and-fast hosted | Groq Llama-3.3 70B *or* Gemini 2.5 Flash | $0.05 – $0.50 |
| Best-quality hosted   | Claude Sonnet 4.6 *or* GPT-4o-class       | $2 – $8 |
| Fully local           | Ollama / vLLM via OpenAI-compatible       | $0 |

\* TTS and image generation costs are independent. The default Kokoro TTS
and SDXL image stack are free to run locally on CPU or modest GPU. Hosted
LLM costs vary with PDF length, depth setting, and shot count.

See [`docs/costs.md`](docs/costs.md) for a fuller breakdown.

---

## Quickstart

```bash
# 1. Install (small default; native SDKs + ML stack are extras)
pipx install booktoanime

# 2. Put your provider keys in a config.yaml (use config.example.yaml as a base)
cp config.example.yaml config.yaml
export GROQ_API_KEY=...        # whichever provider you picked

# 3. System binaries (one-time)
#    macOS:    brew install ffmpeg tesseract
#    Ubuntu:   sudo apt install ffmpeg tesseract-ocr

# 4. Run
booktoanime                     # starts the local server, opens browser
```

Upload a PDF, pick the anime style + narration voice + depth + length, hit
**Generate**. Progress streams live to the browser via Server-Sent Events.
On failure, hit **Resume** to pick up at the last completed stage.

### Install extras

```bash
pip install "booktoanime[anthropic]"        # native Anthropic SDK
pip install "booktoanime[gemini]"           # google-genai
pip install "booktoanime[all-providers]"    # every native LLM SDK
pip install "booktoanime[kokoro]"           # local TTS (pulls torch)
pip install "booktoanime[visual]"           # SDXL + IP-Adapter (pulls torch + diffusers)
```

The `openai_compatible`, `groq`, `together`, `fireworks`, `mistral`, and
`deepseek` adapters use raw HTTP (no SDK), so they work on the default install.

---

## CLI

```
booktoanime [OPTIONS] COMMAND [ARGS]

Commands:
  run       Start the local FastAPI server and open a browser tab.
  resume    Re-run a previously failed/cancelled job from its last completed stage.
  version   Print the package version.

Options:
  --data-dir PATH    Override job/state directory (default: platformdirs user_data_dir).
  --config, -c PATH  Path to config.yaml (default: ./config.yaml).
```

`booktoanime run` is the default; you usually just type `booktoanime`.

`BOOKTOANIME_DATA_DIR=/some/path` overrides `--data-dir` system-wide.

---

## Pipeline stages

```
parsing → structuring → storyboard → images → audio → assembly
```

Each stage writes a versioned artifact to disk under
`<data_dir>/jobs/<job_id>/`:

```
jobs/<job_id>/
├── source.pdf
├── manifest.json            # job state + per-stage status
├── extracted/parsed.json    # parsed text + tables + image refs (parsing)
├── extracted/img_*.png      # raw images extracted from the PDF
├── structured.json          # depth-aware topic summaries (structuring)
├── storyboard.json          # per-shot narration + prompts (storyboard)
├── images/shot_*.png + index.json
├── audio/shot_*.wav + index.json
├── personas/<style>__<seed>.png   # IP-Adapter narrator reference
├── output.mp4
├── output.srt
├── events.log               # NDJSON of every progress event (SSE source of truth)
└── logs/job.log + ffmpeg.log
```

Resume rule: each stage picks up at the first non-completed entry in
`manifest.json`. The image and audio stages add per-shot resume — they
reconcile their `index.json` against on-disk files so deleted files force
a re-render and orphan files (written by a crash before the index was
flushed) get adopted.

---

## Profiles

| Profile | Concurrency (images / audio) | Notes |
|---|---|---|
| `default`      | 2 | Balanced for an 8 GB GPU + CPU fallback |
| `high_quality` | 1 | Heavier checkpoints, fewer parallel shots |
| `low_vram`     | 1 | Smallest viable models, single shot at a time |

Pass via `profile:` in `config.yaml` or per-job in the upload form.

---

## Bring your own provider

Three small abstract interfaces, one decorator-based registry, no edits to
pipeline code. Drop a file under `src/booktoanime/providers/{language,audio,visual}/`:

```python
from booktoanime.providers import (
    LanguageProvider, CompletionRequest, ImageExplanation, VisionInput,
    register_language_provider,
)

class MyProvider(LanguageProvider):
    name = "myprovider"
    async def complete(self, request: CompletionRequest) -> str: ...
    async def explain_image(self, image: VisionInput, *, max_tokens=400, temperature=0.2) -> ImageExplanation: ...
    async def close(self) -> None: ...

@register_language_provider("myprovider")
def _factory(sub_config):
    return MyProvider(...)
```

Then point at it from `config.yaml`:

```yaml
language:
  active: myprovider
  myprovider:
    api_key_env: MYPROVIDER_API_KEY
    model: my-model-name
```

See `docs/providers.md` for the full contract (error mapping, vision
fallback semantics, cancellation handling).

---

## Models, downloads, and disk space

* All model downloads are **lazy per job** — never on install.
* Model cache lives at `<data_dir>/models/`.
* Heavy ML deps (torch, diffusers) live behind install extras (`[kokoro]`,
  `[visual]`) so the default install stays small.

---

## Resume & failure handling

* Failed stages produce **actionable** error messages in the UI (no stack
  traces). Full stack traces land in `<job_dir>/logs/job.log`.
* `booktoanime resume <job_id>` re-runs from the last completed stage.
* The "Resume" button on the failed-job page does the same thing through
  the API.

Hard failures: encrypted PDFs, corrupted PDFs, image-only PDFs with OCR
disabled. Each surfaces a distinct typed error with a clear next step.

---

## Docker

```bash
docker build -t booktoanime .
docker run --rm -it \
  -p 8765:8765 \
  -v "$HOME/booktoanime-data":/data \
  -v "$PWD/config.yaml":/config.yaml \
  -e BOOKTOANIME_DATA_DIR=/data \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  booktoanime --config /config.yaml run --host 0.0.0.0 --no-open-browser
```

The container ships ffmpeg + tesseract pre-installed. GPU support requires
the NVIDIA Container Toolkit and the `[visual]` extra.

---

## Licensing & content responsibility

* Project code: **Apache-2.0** (see `LICENSE`).
* Default runtime dependencies: all permissively licensed (MIT / BSD /
  Apache-2.0). PyMuPDF (AGPL) is **explicitly excluded**; pull requests
  introducing copyleft runtime deps will not be accepted.
* Source-available / non-commercial models (XTTS-v2, FLUX.1 dev) are
  referenced as **opt-in** providers only. The user installs them and
  accepts the upstream license.
* The user is responsible for the content they process. v1 ships with
  no content moderation by design — you run this locally on your own
  files.

See `NOTICE` for the full third-party attribution.

---

## Development

```bash
git clone https://github.com/ai-with-avinash/book-to-anime
cd book-to-anime

# Project venv (per CLAUDE.md house rules)
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

pytest                # 115 tests, fully mocked, no network
ruff check src tests
mypy src
```

Add a `.env` or export your provider keys before running `booktoanime` for a
real end-to-end check.

---

## Roadmap

* Image-to-video stage under `--profile high-quality` (Wan 2.1)
* Multilingual narration: ship voice presets for non-English Kokoro
  packs once upstream stabilizes
* Per-image bbox-based caption hint matching (currently page-level)
* Wider table support (image overlay rendering for narration)
* `booktoanime models download` pre-fetch command

---

## Contributing

Issues + PRs welcome at
<https://github.com/ai-with-avinash/book-to-anime>. Please:

1. Run `ruff check`, `mypy --strict`, and `pytest` locally before opening a
   PR.
2. Match the existing async style — wrap blocking SDKs in
   `asyncio.to_thread`, never block the event loop.
3. Don't introduce AGPL or other copyleft runtime dependencies.

---

## Acknowledgments

Built on `pdfplumber`, `pypdf`, `Kokoro-82M`, `diffusers`, `IP-Adapter`,
`FastAPI`, and `ffmpeg`. See `NOTICE` for the full list. None of the
referenced upstreams are affiliated with or endorse this project.
