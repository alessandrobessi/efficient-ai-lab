# Evaluation Framework

Shared model-evaluation framework used from Week 4 onward to score quantization levels
and compare models on a fixed dataset.

- `datasets/` — the evaluation dataset (JSONL). **v1 (100 examples) built in Week 4**
  — see [`datasets/README.md`](datasets/README.md) — across classification,
  information extraction, structured output, summarization, reasoning, and
  instruction following, in English and Italian. Target is 100–200; v1 covers the
  low end and was used as-is in Week 5 — extending toward 200 remains open.
- `runners/` — [`llama_server_runner.py`](runners/llama_server_runner.py): starts
  llama-server once per quantization level and drives the whole dataset through its
  OpenAI-compatible chat endpoint, rather than reloading the model per example (see
  [Week 5 README](../experiments/05-quantization-quality/README.md) §4).
- `metrics/` — [`scorers.py`](metrics/scorers.py): heuristic, regex/string-based
  per-category scorers (exact match, JSON field/value matching, token-F1 lexical
  overlap, numeric/constraint checkers) — deliberately not an LLM-as-judge, per
  FULL-ROADMAP.md's Week 5 brief.
- `prompts/` — [`templates.py`](prompts/templates.py): category-level system prompts
  plus the classification label-set injection (see Week 5 README limitations for why
  that was necessary).
- `analysis/` — statistical analysis lives in
  [`experiments/05-quantization-quality/analysis/`](../experiments/05-quantization-quality/analysis/)
  rather than here, since it's specific to that week's raw data — this directory is
  reserved for analysis code shared across future evaluation runs (Week 6+).
