# Default vs high-quality model stack

| Stage | `default` profile | `high_quality` profile |
|---|---|---|
| Text + table extraction | pdfplumber | pdfplumber |
| Image extraction        | pypdf | pypdf |
| OCR fallback            | Tesseract | PaddleOCR (extra) |
| LLM                     | (your config — typically Groq Llama / GPT-4o-mini) | Claude Sonnet 4.6 / GPT-4o-class |
| VLM (image understanding) | (your config — defaults to `language.vision_fallback` or text-only) | Qwen2.5-VL (extra) |
| TTS                     | Kokoro-82M | Kokoro-82M (same) |
| Image generation        | small SDXL anime checkpoint + IP-Adapter | Animagine XL / Illustrious + IP-Adapter |
| Image-to-video          | (none) | Wan 2.1 (planned) |
| Assembly                | ffmpeg + Ken Burns + xfade | ffmpeg + xfade + image-to-video clips |

The default stack is chosen so the project ships with **no AGPL or
non-commercial dependencies** and runs on a CPU or modest GPU. The
high-quality profile assumes a 16 GB+ GPU and pulls in heavier weights.

## Disk-space planning

Approximate first-job download sizes:

| Component | Default | High-quality |
|---|---|---|
| Tesseract (system) | ~30 MB | (replaced by PaddleOCR) |
| Kokoro-82M weights | ~300 MB | ~300 MB |
| SDXL base + IP-Adapter | ~7 GB | ~10-12 GB (Animagine XL or Illustrious) |
| Total (excluding LLM weights for local LLMs) | ~7.5 GB | ~13 GB |

Plus whatever local LLM you run (a 70B Q4 quant is ~40 GB; 7B Q4 is ~4 GB).

## Why we don't bundle weights

* They're big (single shipment > GitHub release size limits).
* Licenses vary per checkpoint and may change upstream.
* Users on hosted-only setups (LLM via Groq/Anthropic/etc.) don't need
  any local weights at all.

So everything downloads lazily on first run from the upstream model hub
of record (Hugging Face for SDXL/IP-Adapter/Kokoro). A future
`booktoanime models download` command will pre-fetch on a fast network
for offline boxes; for now you can warm the cache with a single small
end-to-end run on a tiny test PDF.
