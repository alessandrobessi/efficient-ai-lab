# quantscope

**Status: planned — not yet built.** See [`ROADMAP.md`](ROADMAP.md) for the full
development plan.

A CLI that answers a question this program's own research raised but never
fully automated: **for this model, on this CPU, which GGUF quantization format
is actually fastest — and what do you give up to get there?**

## The problem

[Week 4](../../experiments/04-quantization/README.md) and
[Week 6](../../experiments/06-model-comparison/README.md) found repeatedly
that quantization speed and memory footprint don't correlate cleanly with bit
width or parameter count. The leading explanation is that llama.cpp's CPU
backend can repack certain quantized block layouts (`REPACK`) into different
in-memory formats for faster SIMD access on a given CPU — but the evidence for
this in the program is a single log line, never actually profiled. Finding the
fast format for a given model+CPU pair currently means manually running
`llama-bench` across every candidate quantization and reading a CSV by hand.

## What quantscope does

- **Benchmarks, doesn't guess, by default.** Wraps `llama-bench` and
  `llama-quantize` across a model's available (or producible) quantization
  formats and reports a ranked, Pareto-frontier table of speed vs. memory vs.
  quality — the same non-dominated-point framing this program used throughout
  Weeks 5, 6, and its decision framework.
- **Detects what your CPU and build actually use**, not just what the CPU
  supports — cross-checking llama.cpp's own reported SIMD feature line against
  independent OS-level CPU feature detection, which surfaces a genuinely new
  diagnostic: "this CPU supports AVX2 but this llama.cpp build isn't using it."
- **Offers a heuristic `predict` mode**, clearly labeled as a heuristic with a
  confidence caveat rather than a measurement — this program's own evidence is
  that naive bit-width heuristics are often wrong, so quantscope doesn't
  pretend otherwise.
- **Wraps, doesn't reimplement.** All actual benchmarking and quantization
  goes through llama.cpp's own `llama-bench`/`llama-quantize` binaries.

## Relationship to this repo

Generalizes the sweep-over-GGUF → CSV → pandas/matplotlib pipeline built in
`../../experiments/02-llama-cpp/`, `../../experiments/04-quantization/`, and
`../../experiments/06-model-comparison/` into a reusable, installable tool
instead of one-off experiment scripts. See `ROADMAP.md` for the exact reuse
plan.

## License

MIT — see [`../../LICENSE`](../../LICENSE).
