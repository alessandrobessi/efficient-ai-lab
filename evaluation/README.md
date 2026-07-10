# Evaluation Framework

Shared model-evaluation framework used from Week 4 onward to score quantization levels
and compare models on a fixed dataset.

- `datasets/` — the evaluation dataset (JSONL), built starting Week 4 (target: 100–200
  examples across classification, extraction, structured output, summarization, simple
  reasoning, instruction following, Italian, and English).
- `runners/` — code that runs a model over the dataset and captures raw output.
- `metrics/` — scorers (accuracy, exact match, F1, JSON validity, instruction
  compliance, semantic similarity).
- `prompts/` — prompt templates used by runners.
- `analysis/` — statistical analysis (means, variance, confidence intervals, effect
  sizes, failure categorization) and the quality-vs-performance Pareto plot.

Not started — first used in Week 4.
