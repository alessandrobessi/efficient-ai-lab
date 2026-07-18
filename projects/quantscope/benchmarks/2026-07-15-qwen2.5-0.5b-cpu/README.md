# Benchmark: Qwen2.5-0.5B-Instruct, 8 formats, Apple M4 CPU

A real `llama-bench`/`llama-quantize`/`llama-perplexity` run — not stub
scripts — using quantscope's own CLI end to end: quantize an F16 base into
7 formats, benchmark all 8 (including the F16 baseline) with CPU forced
(`-ngl 0` on every invocation, bench *and* perplexity), measure quality
loss relative to F16, and rank the result.

![Pareto frontier: file size vs. generation speed, annotated with perplexity delta](frontier.png)

## Setup

| | |
|---|---|
| Hardware | Apple M4, 10 cores, 16GB RAM, macOS 15.7.3 |
| llama.cpp | build `a935fbffe` (version 9960), installed via `brew install llama.cpp` |
| Base model | [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), `qwen2.5-0.5b-instruct-fp16.gguf` (630,167,424 params, confirmed via `model_n_params` in every row of `results.csv`) |
| Formats | F16 (baseline) + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K, all produced from the F16 base by quantscope's own `quantize` command |
| Perplexity dataset | [`wiki-like-sample.txt`](wiki-like-sample.txt) — a real ~154KB public-domain excerpt from Project Gutenberg's *Pride and Prejudice*, committed in this directory (v0.2.0's copy of this file was never actually committed; fixed in v0.2.1) |
| quantscope | this branch (v0.2.1, pre-tag), built from source |
| Bench config | `--threads 8 --rounds 10 --seed 42 --n-prompt 256 --n-gen 64`, CPU forced on every llama.cpp invocation (`-ngl 0` — see the manifest's `n_gpu_layers: 0` despite `gpu_info: "Apple M4"` and `backends: "BLAS,MTL"` both being present) |

Reproduce with:

```bash
cd projects/quantscope
uv sync

MODEL=qwen2.5-0.5b-instruct-fp16.gguf   # download from the HF repo above
uv run quantscope quantize --llama-quantize-bin llama-quantize \
  --input "$MODEL" --output-dir quants Q2_K Q3_K_M Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0

uv run quantscope bench --llama-bench-bin llama-bench \
  --llama-perplexity-bin llama-perplexity \
  --perplexity-dataset wiki-like-sample.txt \
  --perplexity-baseline-format F16 \
  --gguf F16="$MODEL" \
  --gguf Q8_0=quants/qwen2.5-0.5b-instruct-fp16-Q8_0.gguf \
  --gguf Q6_K=quants/qwen2.5-0.5b-instruct-fp16-Q6_K.gguf \
  --gguf Q5_K_M=quants/qwen2.5-0.5b-instruct-fp16-Q5_K_M.gguf \
  --gguf Q4_K_M=quants/qwen2.5-0.5b-instruct-fp16-Q4_K_M.gguf \
  --gguf Q4_0=quants/qwen2.5-0.5b-instruct-fp16-Q4_0.gguf \
  --gguf Q3_K_M=quants/qwen2.5-0.5b-instruct-fp16-Q3_K_M.gguf \
  --gguf Q2_K=quants/qwen2.5-0.5b-instruct-fp16-Q2_K.gguf \
  --threads 8 --rounds 10 --seed 42 --n-prompt 256 --n-gen 64 \
  --output results.csv --plot frontier.png
```

`--rounds 10` runs 10 independently-shuffled measurement rounds — each
round visits all 8 formats in a freshly-randomized order (recorded in
`results_manifest.json`'s `round_orders`) instead of the fixed
F16→Q8_0→Q6_K→... sequence v0.2.0 used, so thermal/background/caching
drift over the run gets spread evenly across formats instead of
correlating with format identity. Perplexity is measured once per format
afterward, not repeated per round (it's deterministic).

## What the data shows

| Format | Size (MB) | Gen tok/s (mean ± stddev, 10 rounds) | PPL | PPL delta vs. F16 | Pareto-optimal |
|---|---|---|---|---|---|
| F16 (baseline) | 1207.8 | 29.4 ± 21.7 | 20.323 | +0.000 | yes |
| Q8_0 | 644.4 | 59.9 ± 45.7 | 20.375 | +0.052 | no |
| Q6_K | 620.2 | 60.6 ± 47.2 | 20.379 | +0.056 | yes |
| Q5_K_M | 498.0 | 65.0 ± 48.8 | 20.591 | +0.268 | yes |
| Q4_K_M | 468.6 | 67.7 ± 49.3 | 20.920 | +0.597 | yes |
| Q3_K_M | 412.0 | 78.5 ± 52.3 | 21.829 | +1.506 | yes |
| Q2_K | 395.9 | 78.0 ± 56.8 | 22.843 | +2.520 | yes |
| Q4_0 | 408.9 | 92.9 ± 61.9 | 22.587 | +2.264 | yes |

**The honest headline: this run cannot support a confident "format X is
fastest" claim, and that's itself the real finding.** Q4_0 posted the
highest mean generation throughput, roughly tracking file size this time
(smaller formats measured faster, close to monotonically) — the opposite
pattern from v0.2.0's run, which found Q8_0 fastest despite being larger
than four K-quants. Neither result should be trusted at face value: look
at the stddevs. Every format's round-to-round standard deviation is 60-90%
of its own mean, an order of magnitude noisier than v0.2.0's 0.7-8.1 tok/s
stddevs. The difference between adjacent rows in the table above is
routinely smaller than either row's own stddev — well within
`--pareto-epsilon`'s blind spot, since the epsilon-dominance check compares
means against a flat 2% band, not each point's own confidence interval
(a documented limitation — see [ROADMAP.md](../../ROADMAP.md#explicitly-deferred-stretch-goal-not-v10-scope)).

**Why this run is noisier than v0.2.0's, and why that's the correct
tradeoff, not a regression.** v0.2.0 called `llama-bench -r 3` *once per
format* — three repetitions back-to-back inside one warm process, model
already loaded and page-cached from the first repetition. That produces
tight numbers, but every format's three repetitions ran in a single
contiguous block, so any drift over the run (thermal, background load,
OS scheduling) lands on whichever format happened to run early or late —
exactly the confound the second external review flagged. v0.2.1 instead
launches a fresh `llama-bench -r 1` process per (round, format) pair, in a
newly-shuffled order each round: model load, page-cache warmup, and thread
pool startup happen fresh every single measurement. That's real,
structural round-to-round noise on top of whatever drift exists — for a
0.5B model with a short `-n 64` generation phase, the fixed per-launch
overhead is a large fraction of the total measured time, so it shows up as
large variance. The tradeoff is deliberate: v0.2.1 trades tight-looking
numbers for order-independence, and the honest result is that this
model/hardware/config combination needs more than 10 single-shot rounds
(or repetitions *within* each round) to separate formats with confidence —
not that quantization format doesn't matter.

**What *did* stay stable across both runs: perplexity.**
`ppl_delta` is nearly identical to v0.2.0's numbers (e.g. Q4_K_M: 0.597 now
vs. 0.591 then; Q2_K: 2.520 now vs. 2.480 then) — expected, since
perplexity is a deterministic single pass over the same dataset, unaffected
by process-launch or scheduling noise. `perplexity_error` (llama-perplexity's
own `+/- ...` uncertainty) is tight and consistent across formats
(0.41-0.47), new in v0.2.1 — v0.2.0 discarded this number entirely.

```
$ quantscope recommend --csv results.csv --max-ppl-delta 0.3
format  file_size_mb  gen_tokens_per_second  ppl_delta
Q5_K_M        498.0                   65.0     0.268
  Q6_K        620.2                   60.6     0.056
  Q8_0        644.4                   59.9     0.052
   F16       1207.8                   29.4     0.000
```

```
$ quantscope recommend --csv results.csv --max-size-gb 0.5
format  file_size_mb  gen_tokens_per_second  ppl_delta
  Q4_0        408.9                   92.9     2.264   <- highest mean, but see noise caveat above
  Q2_K        395.9                   78.0     2.520
Q3_K_M        412.0                   78.5     1.506
Q4_K_M        468.6                   67.7     0.597
Q5_K_M        498.0                   65.0     0.268
```

`recommend` still does its job — narrowing 8 Pareto-ish candidates to the
ones meeting a stated constraint — but at this noise level, treat its
sort order as "roughly this tier," not a confident ranking within a tier.

## Known limitations of this specific run

- **Round-to-round variance is large (60-90% of the mean) and dominates
  the differences between most formats.** See "Why this run is noisier
  than v0.2.0's" above. A future refinement worth trying: `llama-bench -r
  N` *within* each round (not just `-r 1`), trading some of the
  order-randomization's granularity for tighter per-round measurements —
  not implemented in this pass, noted in
  [ROADMAP.md](../../ROADMAP.md#explicitly-deferred-stretch-goal-not-v10-scope)
  as a candidate for a future version.
- **This machine was an actively-used development laptop, not a dedicated
  benchmark rig.** Background load varied over the ~2 hour run (other
  applications competing for CPU); the randomized-round design spreads
  that load's effect evenly across formats rather than letting it
  correlate with format identity, but it doesn't eliminate the load's
  contribution to variance.
- **One thread count (8), one hardware target, one model size (0.5B).**
  Whether any format's speed edge generalizes to other models/thread
  counts/CPUs is an open question this single run doesn't answer.
- **Perplexity measured on a modest, non-standard text sample** (a
  Pride-and-Prejudice excerpt, not the standard wikitext-2 test set used in
  most published llama.cpp perplexity numbers). Valid for *this* run's own
  same-model, same-tokenizer relative comparison (exactly what `ppl_delta`
  is for), not for comparing these absolute PPL numbers against numbers
  published elsewhere.
- **No IQ*-format / imatrix-calibrated run included** — `quantize --imatrix`
  is implemented and tested (see `../../ROADMAP.md`) but wasn't exercised
  in this particular benchmark, which only covers the legacy/K-quant
  families.
