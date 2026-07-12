# The CPU Small Language Model Benchmark

### Phase II Final Deliverable — Weeks 4–6, Efficient AI Systems

**Central question:** What tradeoffs emerge when Small Language Models are compressed
and compared?

This report synthesizes three weeks of controlled experiments on the same machine as
Phase I (Apple M4, 4 performance + 6 efficiency cores, 10 total; macOS 15.7 arm64),
all via llama.cpp/GGUF on CPU only. Weeks 4-5 held the model constant
(`Qwen2.5-1.5B-Instruct`) and varied quantization level; Week 6 held quantization
constant (Q4_K_M) and varied the model across 5 models spanning 4 families. Full
detail, raw data, and figures live in the linked week READMEs; this report connects
them into one narrative and states what Phase II leaves open for Phase III.

- [Week 4 — Quantization Fundamentals](../../experiments/04-quantization/README.md)
- [Week 5 — Quantization vs Quality](../../experiments/05-quantization-quality/README.md)
- [Week 6 — Small Model Comparison](../../experiments/06-model-comparison/README.md)
- [Open questions tracker](../../docs/methodology/open-questions.md)
- [Phase I report](cpu-inference-performance-report-v1.md)

---

## 1. Quantization Cost: Non-Monotonic From the Start

Week 4 measured disk size, peak RSS, load time, and decode speed across 6
quantization levels of one model (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M) and found
only one of the four behaved as the naive "fewer bits, less of everything except
speed" intuition predicts:

| quant level | disk (MB) | load (s) | peak RSS (MB) | TTFT (s) | decode (tok/s) |
|---|---|---|---|---|---|
| F16 | 3395.5 | 0.653 | 3771.5 | 0.369 | 22.63 |
| Q8_0 | 1806.8 | 1.071 | **4108.5** | 0.263 | 41.98 |
| Q6_K | 1396.3 | 0.906 | 3397.4 | 0.442 | 42.25 |
| Q5_K_M | 1225.9 | 0.860 | 3088.1 | 0.459 | 41.56 |
| Q4_K_M | 1065.6 | 0.866 | 2798.7 | 0.386 | **57.15** |
| Q3_K_M | 881.6 | 0.654 | 2172.4 | 0.357 | 38.99 |

Disk size drops cleanly and monotonically. Nothing else does: **Q8_0 uses more peak
memory than unquantized F16**, load time is *slowest in the middle* of the precision
range rather than scaling with file size, and **Q4_K_M — not the least-quantized
format — is the fastest decoder tested**, 36% faster than its immediate neighbors.
The likely mechanism, not fully confirmed: llama.cpp's CPU backend can *repack*
certain quantized block layouts into a different in-memory format for faster SIMD
access (`REPACK`, reported as a supported feature on this build) — a load-time cost
some formats pay and others don't, and a runtime benefit some formats get more of than
others, independent of raw bit-width.

## 2. Quantization Quality: Free Until It Isn't

Week 5 scored all 6 quantization levels against the same 100-example evaluation
dataset (600 generations) and found quality tracks cost even less than the naive
hypothesis predicted — in the *good* direction, down to a point:

| quant level | mean quality score | 95% CI | Δ vs. F16 | decode tok/s |
|---|---|---|---|---|
| F16 | 0.702 | [0.624, 0.773] | — | 22.63 |
| Q8_0 | 0.715 | [0.640, 0.786] | +0.013 (n.s.) | 41.98 |
| Q6_K | **0.731** | [0.656, 0.801] | +0.029 (n.s.) | 42.25 |
| Q5_K_M | 0.714 | [0.639, 0.788] | +0.012 (n.s.) | 41.56 |
| Q4_K_M | 0.694 | [0.614, 0.772] | −0.008 (n.s.) | **57.15** |
| Q3_K_M | 0.653 | [0.573, 0.733] | −0.049 (n.s., CI edge) | 38.99 |

No quantization level differs significantly from unquantized F16 down through
Q4_K_M — Q6_K nominally scores *highest* of all six. Only Q3_K_M shows a real, if
formally non-significant, decline, and it's also the only level to ever emit invalid
JSON (96.9% validity vs. 100% everywhere else). Combining this with §1's speed
numbers produces a clean two-point Pareto frontier: **Q6_K** (best quality, ~1.9x
F16 speed) and **Q4_K_M** (fastest, quality statistically indistinguishable from
F16) are the only non-dominated quantization levels — every other level, including
F16 itself, is beaten on both axes simultaneously by at least one of these two.

The single most interesting failure mode wasn't a smooth decline but a spike: **7 of
8 Italian summarization examples got answered in English at Q4_K_M specifically**
(vs. 0-2 at every other level), single-handedly dragging that category's Q4_K_M score
to 0.275, the lowest of any (level, category) pair measured this phase. Aggregated
over all categories, Italian scores trail English at every quantization level
(F16: 0.747 EN vs. 0.657 IT), so multilingual capability is both weaker to start and
more fragile under quantization than English-only numbers would suggest.

## 3. Model Comparison: Size Predicts a Lot, Except When It Doesn't

Week 6 repeated Weeks 4-5's full methodology across 5 models (Qwen2.5-0.5B/1.5B,
Llama-3.2-1B, Gemma-2-2B-it, Phi-3.5-mini — 0.5B to 3.82B params, all Q4_K_M):

| model | family | params (B) | decode (tok/s) | mean quality | 95% CI |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Qwen | 0.50 | **116.98** | 0.595 | [0.515, 0.676] |
| Qwen2.5-1.5B-Instruct | Qwen | 1.50 | 62.37 | 0.694 | [0.613, 0.771] |
| Llama-3.2-1B-Instruct | Llama | 1.24 | 77.44 | 0.588 | [0.506, 0.669] |
| Gemma-2-2B-it | Gemma | 2.61 | 34.22 | 0.746 | [0.671, 0.816] |
| Phi-3.5-mini-instruct | Phi | 3.82 | 23.29 | **0.774** | [0.704, 0.840] |

Unlike Phase II's within-model quantization sweeps, parameter count *does* correlate
strongly with both quality (r=0.91) and speed (r=−0.94) across these 5 models — but
n=5 keeps this descriptive, not confirmatory (a single point can swing a
5-point correlation substantially), and there's a clean counterexample:
**Llama-3.2-1B-Instruct is strictly Pareto-dominated by Qwen2.5-0.5B-Instruct** — a
model under half its size that is both faster (117.0 vs. 77.4 tok/s) and
higher-quality (0.595 vs. 0.588, though their CIs nearly fully overlap). Its weakness
is concentrated in `instruction_following` (0.389, the single lowest score any model
got in any category this phase) rather than a general capability gap — its
`structured_output` score (0.917) is the second-best of any model tested.

Multilingual performance directly falsified this phase's own prior expectation:
Qwen — the family most associated with published multilingual training — had the
*largest* English/Italian gap of any model (0.185), while **Phi-3.5-mini-instruct had
essentially none** (0.772 EN vs. 0.777 IT). Family reputation didn't predict this
outcome; nothing about training data composition was directly measured to explain it.

By tokens/sec per billion parameters (a rough CPU-efficiency measure), the Qwen
family was clearly the most CPU-friendly tested (234.0 and 41.6 tok/s/B at its two
sizes) and Phi-3.5-mini the least (6.1) — consistent with Phi-3.5-mini also being a
6-16x outlier on load time (7.65s vs. 0.47-1.23s for every other model) despite a
disk size only 1.4x Gemma-2-2B's. As in §1, this is circumstantial evidence of an
architecture/GGUF-conversion fit effect, not a profiled, confirmed mechanism.

## 4. Two Pareto Frontiers, One Pattern

Both this phase's headline visualizations (Week 5's quantization Pareto plot, Week
6's model Pareto plot) tell the same structural story: **most of the tested space is
dominated, and the frontier is small.** Of 6 quantization levels, only 2 are
Pareto-optimal (Q6_K, Q4_K_M); of 5 models, only 1 is strictly dominated
(Llama-3.2-1B) and the other 4 form the frontier — but that's because the 4
survivors are spread widely across the speed/quality plane, not clustered together,
meaning "best" is genuinely use-case-dependent rather than there being one clear
winner. In neither case does "most aggressive compression" or "fewest parameters"
sit at either end of the frontier in the way a naive intuition would predict — Q3_K_M
(smallest quantization) and Llama-3.2-1B (not the smallest model, but the specific
counterexample) are each dominated, not optimal, on their respective plane.

## 5. Statistical Methodology, Applied Consistently

Both quality analyses (Weeks 5-6) used the same toolkit, established in Week 5 and
reused without modification in Week 6: bootstrap 95% confidence intervals (10,000
resamples) on mean quality scores rather than assuming normality (many per-example
scores are 0/1), paired comparisons where the same 100 examples are shared across
conditions (Week 5's quant-level-vs-F16 diffs), and heuristic, non-LLM-judge scorers
per FULL-ROADMAP.md's explicit brief — see
[`evaluation/metrics/scorers.py`](../../evaluation/metrics/scorers.py). Week 5's
first evaluation pass caught 3 real prompt/scorer bugs via a self-scoring sanity
check (every dataset example scored against its own answer should return 1.0); Week 6
reused the validated, fixed pipeline directly, and the same sanity check was rerun
before trusting its output.

## 6. Observed Anomalies This Phase

In order of how surprising they were relative to the naive hypothesis being tested:

1. **Q4_K_M is the fastest quantization format tested — not the least quantized**
   (Week 4) — the single clearest violation of "fewer bits, more speed, proportionally."
2. **Llama-3.2-1B-Instruct is Pareto-dominated by a model under half its size**
   (Week 6) — parameter count predicts quality on average, but not for every model.
3. **Q4_K_M spikes to 7/8 English-language answers on Italian summarization prompts**
   while its immediate neighbors don't (Week 5) — a non-monotonic, format-specific
   multilingual failure, not a smooth degradation.
4. **Qwen2.5-1.5B-Instruct has the *largest* EN-IT gap of any model tested, while
   Phi-3.5-mini has almost none** (Week 6) — family multilingual reputation didn't
   predict the outcome.
5. **Q8_0 uses more peak memory than unquantized F16** (Week 4), and
   **Llama-3.2-1B-Instruct uses more peak RAM than the more-than-twice-as-large
   Gemma-2-2B-it** (Week 6) — parameter/bit count doesn't predict memory footprint
   either.
6. **Phi-3.5-mini's load time is 6-16x every other model tested** (Week 6) despite a
   disk size only 1.4x Gemma-2-2B's — plausibly architecture/tokenizer-specific, not
   confirmed.

## 7. Unanswered Questions Going Into Phase III

The full, itemized register is
[`docs/methodology/open-questions.md`](../../docs/methodology/open-questions.md).
Still open at the end of Phase II:

- What specifically causes the `REPACK`-linked load-time and memory anomalies (Week
  4), and the Phi-3.5-mini load-time / Llama-3.2-1B memory anomalies (Week 6)? None
  of these were profiled beyond circumstantial evidence.
- Does Llama-3.2-1B-Instruct's instruction-following weakness hold at other
  quantization levels, or is it Q4_K_M-specific?
- Would a harder, larger evaluation dataset (both quality dimensions and Italian
  coverage) change which of this phase's "not statistically significant" or
  "n=5, descriptive only" findings hold up?
- How do llama-server's HTTP-level latencies compare across models under concurrent
  load — directly relevant once Phase III's Weeks 7-9 build a production gateway and
  load-test it.

## 8. Reproducibility

Every number in this report traces back to a raw CSV/JSONL and metadata JSON under
`results/raw/{04-quantization,05-quantization-quality,06-model-comparison}/`,
generated by the scripts in each week's `experiments/*/scripts/` directory and
processed by each week's `analysis/` scripts. See each week's README §4 for exact
commands. Hardware, software versions, and configuration are captured per-experiment
per the metadata standard in root `README.md` §8. All 5 Week 6 GGUF models are
publicly available and non-gated on Hugging Face — exact repo/file names are in
[Week 6's README §4](../../experiments/06-model-comparison/README.md#4-what-is-the-experimental-setup).
