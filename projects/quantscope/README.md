# quantscope

**Status: core implemented, pre-release.** Installable, tested end-to-end
(`bench`, `predict`, `quantize`, `report`, `formats`, `cpu-info` all work).
Not yet packaged as a tagged release (no CI, no `CONTRIBUTING.md`) — see
[`ROADMAP.md`](ROADMAP.md) for what's left.

A CLI that answers a question this program's own research raised but never
fully automated: **for this model, on this CPU, which GGUF quantization format
is actually fastest — and what do you give up to get there?**

## Quickstart

```bash
cd projects/quantscope
uv sync

# what does this llama-quantize binary support, and what's usable without an imatrix?
uv run quantscope formats --llama-quantize-bin /path/to/llama-quantize

# is this CPU's hardware being fully used by this llama.cpp build?
uv run quantscope cpu-info --llama-bench-bin /path/to/llama-bench

# produce missing formats, then benchmark them:
uv run quantscope quantize --llama-quantize-bin /path/to/llama-quantize \
  --input model-f16.gguf --output-dir /tmp/quants Q4_K_M Q5_K_M Q8_0
uv run quantscope bench --llama-bench-bin /path/to/llama-bench \
  --gguf Q4_K_M=/tmp/quants/model-f16-Q4_K_M.gguf \
  --gguf Q5_K_M=/tmp/quants/model-f16-Q5_K_M.gguf \
  --gguf Q8_0=/tmp/quants/model-f16-Q8_0.gguf \
  --output results.csv --plot frontier.png

# quick heuristic guess without benchmarking anything (see the caveat it prints):
uv run quantscope predict Q4_0 Q4_K_M Q5_K_M Q8_0 F16
```

Run `uv run quantscope <subcommand> -h` for every flag.

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
