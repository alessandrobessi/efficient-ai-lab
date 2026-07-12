# Evaluation Framework

Shared model-evaluation framework used from Week 4 onward to score quantization levels
and compare models on a fixed dataset.

- `datasets/` — the evaluation dataset (JSONL). **v1 (100 examples) built in Week 4**
  — see [`datasets/README.md`](datasets/README.md) — across classification,
  information extraction, structured output, summarization, reasoning, and
  instruction following, in English and Italian. Target is 100–200; v1 covers the
  low end, with room to extend before Week 5 puts it to use.
- `runners/` — code that runs a model over the dataset and captures raw output. Not
  started — first used in Week 5.
- `metrics/` — scorers (accuracy, exact match, F1, JSON validity, instruction
  compliance, semantic similarity). Not started — first used in Week 5.
- `prompts/` — prompt templates used by runners. Not started — first used in Week 5.
- `analysis/` — statistical analysis (means, variance, confidence intervals, effect
  sizes, failure categorization) and the quality-vs-performance Pareto plot. Not
  started — first used in Week 5.
