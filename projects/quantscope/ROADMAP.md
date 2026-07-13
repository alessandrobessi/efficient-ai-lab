# quantscope — Development Roadmap

**Status: planned.** This document describes the architecture and milestones
for a tool that does not exist yet. No code or `pyproject.toml` has been
created — this is what a future implementation session should build against.

## Why this exists

[Week 4](../../experiments/04-quantization/README.md) (quantization
fundamentals) and [Week 6](../../experiments/06-model-comparison/README.md)
(small model comparison) both found that quantization format and model choice
interact with SIMD/CPU-kernel fit in ways that don't correlate with bit-width
or parameter count — see
[`docs/methodology/glossary.md`](../../docs/methodology/glossary.md)'s `REPACK`
entry and
[§9 of the Week 4 README](../../experiments/04-quantization/README.md). The
leading explanation (llama.cpp's CPU backend repacking certain quantized block
layouts for faster SIMD access) is documented as "circumstantial, indirect"
evidence — a CPU-feature log line, not an actual profile — because confirming
it would need deeper instrumentation than either week's scope allowed. What
both weeks actually needed, and didn't have, was a tool that sweeps the
candidate formats for a given model+CPU and reports the tradeoffs, instead of
a person doing it by hand across many benchmark runs.

## Reuse from `experiments/`

| Component | Path | Disposition |
|---|---|---|
| `llama-bench` subprocess wrapper | `experiments/02-llama-cpp/scripts/llama_cpp_runner.py` (`run_llama_bench`, `run_llama_bench_single_rep`) | Strong foundation — CSV output already matches the "wrap llama-bench, don't reimplement GGML kernels" principle |
| Sweep-over-format pipeline | `experiments/04-quantization/scripts/quantization_benchmark.py`, `experiments/04-quantization/analysis/analyze.py` | Sweep → CSV → pandas summary → matplotlib figure pattern, directly generalizable |
| Sweep-over-model pipeline | `experiments/06-model-comparison/scripts/model_benchmark.py`, `experiments/06-model-comparison/analysis/analyze_benchmark.py` | Same pattern, second precedent |
| llama.cpp vendoring | `experiments/02-llama-cpp/scripts/setup_llama_cpp.sh` | CMake, CPU-only build — llama.cpp stays vendored/shelled-out-to, never reimplemented |

## Architecture

Independent Python package (`pyproject.toml` + `uv`,
`requires-python >= 3.11`, `[tool.uv] package = false` to match this repo's
existing convention).

```
projects/quantscope/
├── pyproject.toml
├── quantscope/
│   ├── cli.py                    entry point: bench, predict, quantize, report subcommands
│   ├── cpu_detect.py              two-layer CPU feature detection (see below)
│   ├── formats.py                 query llama-quantize for supported formats; filter by GGUF applicability
│   ├── llama_bin.py               subprocess wrapper (evolves llama_cpp_runner.py)
│   ├── bench.py                   measure: sweep llama-bench across formats
│   ├── predict.py                 heuristic-only, no benchmarking, confidence-labeled
│   ├── quantize.py                auto-produce missing formats via llama-quantize
│   └── report.py                 Pareto-frontier ranking, table + plot output
├── tests/
└── ROADMAP.md, README.md, LICENSE
```

**`cpu_detect.py`** — two-layer detection: parse llama.cpp's own reported
feature line at startup (ground truth for what *this specific build* actually
does) and cross-check it against independent OS-level detection (`sysctl` on
macOS, `/proc/cpuinfo` on Linux). Divergence between the two — CPU supports a
feature the build doesn't report using — is surfaced directly, which is a
diagnostic this program never had.

**`formats.py`** — queries the installed `llama-quantize` binary for its
currently supported format list at runtime rather than hardcoding one, since
llama.cpp adds and renames formats across versions; filters that list against
the target GGUF's own metadata (architecture, existing quantization) for
applicability before including a format in a sweep.

**`bench.py` / `predict.py` split** — `bench` always measures via
`llama-bench`; `predict` is an explicitly labeled heuristic (bit-width and
architecture-based estimate, no benchmark run) with a stated confidence
caveat, kept deliberately separate so a fast-but-approximate mode is never
confused with a measured one — direct response to this program's own finding
that naive bit-width heuristics are frequently wrong.

**`quantize.py`** — shells out to `llama-quantize` to produce formats missing
from a model's existing GGUF set, so `bench`/`report` can cover the full
candidate space without requiring the user to pre-generate every format by
hand.

**`report.py`** — ranks formats on a Pareto frontier (speed vs. memory vs.
quality), reusing the same "dominated" / "Pareto-optimal" language this
program used in Weeks 5, 6, and the architecture decision framework, so a user
already familiar with that framing gets the same mental model here.

## Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0** | `bench` CLI wrapping `llama-bench` over a fixed/supplied format list, ranked table output | Reproduces Week 4's own numbers within noise, against the same model/format set |
| **M1** | CPU feature detection + heuristic `predict` mode | Feature-line parsing matches a fixture table of real captured strings across llama.cpp versions |
| **M2** | `quantize` (auto-produce missing formats) + metadata-based format applicability filtering | Round-trip test: quantize a format, bench it, confirm it appears correctly ranked |
| **M3** | Pareto report/plot polish; optional `--quality-eval` wrapping `llama-perplexity` | Non-dominated-sort output matches hand-computed expected sets on fixture data |
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` (`uv build`) | `uv run pytest` green in CI on a fresh clone without a real llama.cpp build present |

## Testing strategy

- **Mocked-subprocess tests** for `llama_bin.py` using canned `llama-bench`/
  `llama-quantize` output, so most of the suite runs without a real llama.cpp
  build in CI.
- **Feature-line regex tests** against a fixture table of real feature strings
  captured across multiple llama.cpp versions, since the exact log format is
  the single most version-fragile dependency in the tool.
- **Pareto/non-dominated-sort tests** against hand-computed expected sets.
- **One real end-to-end smoke test**, gated behind an integration marker and
  skipped by default in CI — run manually before a release, against an actual
  llama.cpp build and a small real GGUF.

## Explicitly deferred (stretch goal, not v1.0 scope)

Confirming which code path actually executes per quantization format
(REPACK-path confirmation) needs an instrumented llama.cpp fork or a real
A/B toggle — a genuinely open-ended research effort, not a v1.0 engineering
task. v1.0 ships with the same "circumstantial, log-line-only" evidence
caveat this program's own research already carries, documented honestly
rather than either overclaiming the mechanism or blocking the tool's real
value (which is the sweep-and-rank workflow, not the mechanistic explanation)
on resolving it.

## License

MIT — see [`../../LICENSE`](../../LICENSE); each project carries its own copy
of the same license text.
