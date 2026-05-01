# API costs at a glance

> **Last updated:** 2026-04. Prices change. Verify against the upstream
> pricing page before committing to a provider; this table is a
> rough-magnitudes guide, not a quote.

A "book" here means a 200-300 page non-fiction PDF generating ~30 topics
at the `standard` (15-25 min) length preset. Image generation and TTS
costs are NOT included — those run locally on the default stack.

| Provider | Model | Approx tokens / book | Approx cost / book |
|---|---|---|---|
| Groq        | llama-3.3-70b-versatile          | 200k in + 80k out | $0.05 - $0.30 |
| Gemini      | gemini-2.5-flash                 | 200k in + 80k out | $0.10 - $0.50 |
| OpenAI      | gpt-4o-mini                      | 200k in + 80k out | $0.50 - $2.00 |
| OpenAI      | gpt-4o                           | 200k in + 80k out | $5 - $15 |
| Anthropic   | claude-sonnet-4-6                | 200k in + 80k out | $2 - $8 |
| Anthropic   | claude-opus-4-7                  | 200k in + 80k out | $10 - $30 |
| DeepSeek    | deepseek-chat                    | 200k in + 80k out | $0.10 - $0.50 |
| Together    | Llama-3.3-70B-Instruct-Turbo     | 200k in + 80k out | $0.20 - $0.80 |
| Fireworks   | llama-v3p3-70b-instruct          | 200k in + 80k out | $0.20 - $0.80 |
| Mistral     | mistral-large-latest             | 200k in + 80k out | $1 - $4 |
| Local (Ollama / vLLM via OpenAI-compatible) | any local model | n/a | $0 |

## Why the wide ranges?

* **Depth setting** scales token usage linearly. `eli5` is the cheapest,
  `expert` the most expensive (it produces longer, more detailed
  summaries).
* **Length preset** scales the per-topic budget (and therefore the
  per-topic completion length). `in_depth` (40-60 min) costs ~3x more than
  `short` (5-10 min).
* **Vision calls** are billed separately and per image. A book with many
  figures (technical / scientific) increases cost noticeably; a prose-only
  book stays at the low end.

## Tips to reduce cost

* Use `--profile default` for first-pass drafts; bump to `high_quality`
  only for final renders.
* Pick a cheap-and-fast hosted provider for iteration, then re-run the
  same `job_id` with a higher-quality provider once you're happy with
  structure.
* Run TTS + image generation locally — they're free on the default stack.
  Only the LLM stages call hosted APIs.
* For fully-local pipelines, point `language.openai_compatible.base_url`
  at Ollama, vLLM, LM Studio, or llama.cpp — total marginal cost is $0.
