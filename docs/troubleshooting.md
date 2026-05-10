# Troubleshooting

## "No language / audio / visual provider configured."

Cause: missing `config.yaml` or one of the three required sections.

Fix: copy `config.example.yaml` to `config.yaml`, uncomment exactly one
block under each of `language`, `audio`, `visual`, and set the matching
`api_key_env` (or `api_key`).

## "missing API key: set $GROQ_API_KEY or `api_key:` in the provider's config block."

Cause: factory could not resolve the API key for the active provider.

Fix: `export GROQ_API_KEY=...` (or whichever provider you picked) and
re-run, or put `api_key: ...` directly in the provider sub-block.

## "the 'anthropic' package is required for the Anthropic provider."

Cause: you set `language.active: anthropic` without installing the
optional SDK.

Fix: `pip install "booktoanime[anthropic]"` (or `[all-providers]` for
every native SDK).

## "the 'kokoro' package is required for the Kokoro provider."

Cause: kokoro/torch isn't installed (default install ships without
torch to keep the wheel small).

Fix: `pip install "booktoanime[kokoro]"`.

## "the 'diffusers' / 'torch' stack is required for the SDXL provider."

Fix: `pip install "booktoanime[visual]"`. On Linux/macOS without CUDA
this still works — it falls back to CPU (slower).

## "This PDF is encrypted."

Cause: PDF has a password. The parser only tries the empty password
automatically.

Fix: decrypt the PDF first (`qpdf --decrypt input.pdf output.pdf`,
or your PDF reader's "Save without password" option) and try again.

## "This PDF has no text layer (it is image-only)."

Cause: pure scanned PDF + you set `ocr_enabled: false` (or Tesseract
isn't installed).

Fix: install Tesseract (`brew install tesseract` / `apt install tesseract-ocr`)
and leave `ocr_enabled: true` (the default). For non-English text you'll
also need the language pack (`tesseract-ocr-jpn`, etc.).

## "ffmpeg exited 1; see logs/ffmpeg.log"

Cause: ffmpeg was invoked but failed. Common reasons: missing libraries,
unusual aspect ratio (must be evenly divisible), corrupt shot file.

Fix: open `<job_dir>/logs/ffmpeg.log` — the full stderr is recorded
there. The most common causes:

* `Stream specifier ':a' in filtergraph description matches no streams`
  — one of the audio shot files is empty or corrupt.
* `Cannot find a suitable libx264` — your ffmpeg build is missing the
  H.264 encoder. Install `ffmpeg` from a normal package source (Homebrew,
  apt, the project's Docker image).

## "Image generation failed for shot 14: model out of memory."

Cause: GPU VRAM exhausted by SDXL.

Fix:
1. Switch to `profile: low_vram` in `config.yaml`.
2. Reduce `length_preset` (fewer shots → less memory pressure).
3. Reduce `visual.sdxl_diffusers.width` / `height` (e.g. 1280x720).

The pipeline auto-resumes from the failed shot, so the previously rendered
shots aren't lost.

## "no live event stream for this job"

Cause: you opened the SSE URL after the job finished or before it
started.

Fix: refresh the job page — the REST endpoint will fetch the final state
even after the bus is closed.
