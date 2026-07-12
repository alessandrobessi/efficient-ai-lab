# Hypotheses — Week 6: Small Model Comparison

## Overall (naive)

More parameters should mean better quality and slower inference, roughly
proportionally, across all 5 models — the standard scaling-law intuition applied
across model families instead of within one.

## Central hypothesis

> Parameter count positively predicts quality score and negatively predicts decode
> speed, consistently across the 5 models tested (Qwen2.5-0.5B/1.5B, Llama-3.2-1B,
> Gemma-2-2B-it, Phi-3.5-mini-instruct — 0.5B to 3.82B parameters, all Q4_K_M).

Falsifiable by: a smaller model outscoring a larger one on quality, a larger model
decoding faster than a smaller one, or either correlation being weak/absent.

## Why this is worth testing rather than assuming

Every prior week in this program has falsified the equivalent naive hypothesis at
least partially:

- **Week 4** found decode speed and peak memory are *not* monotonic with
  quantization level within a single model — Q4_K_M was fastest, not the least
  quantized; Q8_0 used more RAM than unquantized F16.
- **Week 5** found quality does *not* decline monotonically with quantization either
  — several quantized levels scored statistically indistinguishably from (or nominally
  above) F16.
- **This week's own speed/memory benchmark already shows a similar surprise**:
  Llama-3.2-1B-Instruct (1.24B params) uses more peak RAM (5721 MB) than
  Gemma-2-2B-it (2.61B params, 4003 MB) — a smaller model using *more* memory than a
  more-than-twice-as-large one, and Phi-3.5-mini's load time (7.6s) is 4-8x every
  other model tested despite not being disproportionately larger on disk. Cross-model
  comparisons add a new axis of variation (architecture, tokenizer, training data,
  chat template) on top of the parameter-count axis, so there's no strong reason to
  expect the "more parameters, more of everything" intuition to hold cleanly here
  either — especially at n=5 models, where a single architectural outlier can dominate
  a naive correlation.

## Sub-hypotheses (the roadmap's specific analysis questions)

- **Does parameter count predict quality?** Expected: weakly at best — Week 5 already
  showed within-model quantization noise can exceed real quality differences; across
  architecturally different models, training data and instruction-tuning quality
  plausibly matter more than raw parameter count at this small scale (0.5B-3.8B).
- **Does parameter count predict latency?** Expected: more reliably than quality
  (compute scales predictably with parameter count in a way "quality" doesn't), but
  this week's benchmark already shows family/architecture-specific effects (Phi-3.5's
  outsized load time) that a pure parameter-count story wouldn't predict.
- **Which model gives the best quality/performance tradeoff?** No prior expectation —
  this is what the Pareto plot is for.
- **Which model is best in Italian?** Expected: the Qwen family, given Alibaba's
  publicly-documented multilingual training emphasis, but genuinely untested for this
  specific dataset and these specific models.
- **Which model produces the most reliable structured output?** Expected: weakly
  correlated with instruction-tuning quality rather than size — Week 5 found even a
  1.5B model was JSON-reliable down to Q4_K_M, so this may saturate similarly here.
- **Are some families particularly CPU-friendly?** Measured here via decode tokens/sec
  per billion parameters — expected to vary by how well each model's GGUF format and
  architecture map onto this CPU's SIMD kernels, echoing Week 4's finding that kernel
  fit matters more than raw size.
