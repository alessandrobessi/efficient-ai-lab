# Hypotheses — Week 1: Transformer Inference Fundamentals

## Overall

Prefill (processing the prompt) and decode (generating tokens one at a time) are
different computational regimes, and that difference should be visible in simple
timing measurements without touching any internals of the model:

- Prefill cost should scale with **prompt length**, because it's one forward pass over
  the whole prompt.
- Decode cost per token should be roughly **constant**, because after prefill, each
  step is a fixed-size forward pass over one new token plus the cached keys/values —
  independent of how long the prompt was or how many tokens have already been decoded.
- Therefore total latency for a fixed prompt should scale roughly **linearly** with the
  number of tokens generated, dominated by the constant per-token decode cost.

## Experiment 1.1 — Model Loading Time

> Model loading time and memory footprint are stable across repeated loads of the same
> model, after accounting for OS-level file cache warm-up on the first run.

Falsifiable by: load time or memory footprint drifting upward/downward over repeated
runs beyond what a first-run cache effect explains.

## Experiment 1.2 — Prompt Length

> Time to First Token grows with prompt length (prefill scales with prompt size), while
> steady-state decode speed (tokens/s) stays roughly constant regardless of prompt
> length, since decode only depends on the KV cache grown during prefill, not on
> re-processing the prompt.

Falsifiable by: decode speed dropping substantially as prompt length grows (would
suggest decode cost is not actually independent of context length at this scale), or
TTFT not growing with prompt length (would suggest prefill is not the bottleneck being
measured).

## Experiment 1.3 — Output Length

> With the prompt fixed, total latency grows approximately linearly with the number of
> tokens generated, and TTFT (a one-time prefill cost) stays roughly constant across
> output lengths.

Falsifiable by: total latency growing super-linearly or sub-linearly with output length,
or TTFT varying with output length (it shouldn't, since output length is decided after
the prefill step already happened).
