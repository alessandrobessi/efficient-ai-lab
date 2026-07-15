# quantscope

[![quantscope CI](https://github.com/alessandrobessi/efficient-ai-lab/actions/workflows/quantscope-ci.yml/badge.svg)](https://github.com/alessandrobessi/efficient-ai-lab/actions/workflows/quantscope-ci.yml)

A CLI that answers a question this program's own research raised but never
fully automated: **for this model, on this CPU, which GGUF quantization
format is actually fastest — and what do you give up to get there?**

**Status: [`quantscope/v0.2.0`](https://github.com/alessandrobessi/efficient-ai-lab/tree/quantscope/v0.2.0) tagged.**
Installable, tested end-to-end, CPU inference always forced (`-ngl 0`,
never GPU-offloaded — see [Why this matters](#why-this-matters)). No
package published to PyPI or as a release artifact yet — install from
source (below). See [`ROADMAP.md`](ROADMAP.md) for full design rationale
and [Known limitations](#known-limitations) for what's still open.

## Table of contents

- [Why this matters](#why-this-matters)
- [30-second demo](#30-second-demo)
- [A real quantization sweep](#a-real-quantization-sweep)
- [Installation](#installation)
- [Concepts](#concepts)
- [CLI reference](#cli-reference)
- [Output formats](#output-formats)
- [Architecture](#architecture)
- [Known limitations](#known-limitations)
- [License](#license)

## Why this matters

Say you have a language model and want to run it locally. GGUF quantization
lets you shrink it — trading precision for a smaller file and (usually)
faster inference. The obvious assumption is: the more aggressively you
shrink it, the faster it runs. That assumption is often wrong.

[Week 4](../../experiments/04-quantization/README.md) and
[Week 6](../../experiments/06-model-comparison/README.md) of this program
found that quantization speed doesn't reliably track file size or bit
width — a format that's nominally "more compressed" can measure *slower*
than one that's nominally larger, because how well a format's memory layout
fits the CPU's actual SIMD instructions matters more than how few bits it
uses. The only way to know which format is genuinely fastest on your
specific model and your specific CPU is to actually run it and measure —
guessing from the format's name is a coin flip. [A real run below](#a-real-quantization-sweep)
shows exactly this: an 8-bit format beating four smaller ones.

quantscope automates that measurement instead of leaving it as something
you do by hand with a stopwatch and a spreadsheet: produce the missing
quantization formats, benchmark every one on your actual CPU (never
silently on a GPU — see the next point), measure how much output quality
each one costs relative to the original, and rank the tradeoffs so you can
pick with evidence instead of a guess.

## 30-second demo

```bash
cd projects/quantscope
uv sync
uv run quantscope bench --llama-bench-bin llama-bench \
  --gguf Q4_K_M=model-Q4_K_M.gguf --gguf Q8_0=model-Q8_0.gguf \
  --output results.csv --plot frontier.png
```

```
format  file_size_mb  gen_tokens_per_second  pareto_optimal
  Q8_0    644.408051             166.662051            True
Q4_K_M    468.635590             162.907600            True
```

(Both Pareto-optimal here — Q8_0 is bigger but faster, Q4_K_M is smaller but
slower, a genuine tradeoff at just two formats. The real 8-format sweep
below is where things get more interesting.)

Every `bench` run always benchmarks CPU-only — `-ngl 0` is passed to
`llama-bench` unconditionally, regardless of whether your build has
Metal/CUDA/Vulkan support compiled in, because a "CPU-kernel-fit profiler"
that silently benchmarks GPU offload isn't answering the question it claims
to.

## A real quantization sweep

Not synthetic numbers: a real `llama-bench`/`llama-quantize`/
`llama-perplexity` run, 8 formats of Qwen2.5-0.5B-Instruct, on an Apple M4
CPU.

![Pareto frontier: file size vs. generation speed, annotated with perplexity delta](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/frontier.png)

**Q8_0 is the fastest format tested — faster than Q4_K_M, Q5_K_M, and
Q6_K, despite being nominally less compressed than all three** (166.7
tok/s vs. Q4_K_M's 162.9, on this exact hardware). This is the "measure,
don't guess" thesis made concrete, not hypothetical. Full setup, raw CSV,
manifest, and honest limitations:
[`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/).

## Installation

Requires Python 3.11+ and a llama.cpp build
(`llama-bench`/`llama-quantize`, and `llama-perplexity` for quality
evaluation) on your machine somewhere — quantscope never bundles or builds
llama.cpp itself; see
`../../experiments/02-llama-cpp/scripts/setup_llama_cpp.sh` for one way to
build it, or `brew install llama.cpp` on macOS.

```bash
cd projects/quantscope
uv sync
uv run quantscope --version
uv run quantscope -h
```

Or install it as a standalone tool from a local build:

```bash
uv build
uv tool install dist/quantscope-0.2.0-py3-none-any.whl
quantscope -h
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for building, testing, and
extending quantscope (e.g. adding a new subcommand).

## Concepts

**CPU is forced, always.** `bench` passes `-ngl 0` to every `llama-bench`
invocation unconditionally — no flag needed, no way to turn it off.
llama-bench's own default (`-ngl -1`) maximally offloads to GPU if the
build supports one; on a machine with Metal/CUDA/Vulkan compiled in
(common — Homebrew's llama.cpp ships Metal support on macOS by default),
that would silently answer a different question than the one this tool
exists to answer. Every `bench` run's manifest records `n_gpu_layers` so
this is independently verifiable, not just asserted.

**Pareto frontier / non-dominated points, with a noise tolerance.** A
format is "Pareto-optimal" if no other format is at least as good on every
objective and *materially* better on at least one — "materially" meaning
beyond a configurable relative tolerance (`--pareto-epsilon`, default 2%),
not just numerically different. `llama-bench` reports a stddev alongside
every average because run-to-run noise is real; without a tolerance, two
runs that are statistically indistinguishable could still flip which one
"wins" based on noise alone. This reuses the "dominated"/"Pareto-optimal"
framing this program used in Weeks 5, 6, and its architecture decision
framework, extended with epsilon-dominance rather than treating every
benchmark average as if it were exact.

**`recommend`: Pareto isn't always a single answer.** When every format
that gets smaller also gets slower or lower-quality, *all* of them are
Pareto-optimal simultaneously — a true, correct, and not always useful
answer to "which should I pick?" `recommend` filters an already-ranked
table down to what actually satisfies constraints you state explicitly
(`--max-size-gb`, `--min-tokens-per-second`, `--max-ppl-delta`), turning
"here are 8 valid points" into "here's what's left given what you actually
care about."

**`bench` vs. `estimate-size`.** `bench` always measures, by running
`llama-bench` against real GGUF files. `estimate-size` ranks formats by
approximate bits-per-weight with zero benchmarking — but it is deliberately
*not* framed as a speed estimate: this program's own data (see
[Why this matters](#why-this-matters)) shows speed doesn't track bit width
reliably, so a command that ranked by bit width and called it a "speed
prediction" would directly contradict quantscope's own reason for existing.
`estimate-size` answers "will this roughly fit in memory," not "will this
be fast" — only `bench` answers that.

**Perplexity is a quantization-loss proxy relative to a baseline, not
"quality."** A bare perplexity number is only meaningful relative to a
same-model, same-tokenizer reference — llama.cpp's own docs are explicit
about this. Passing `--perplexity-baseline-format` (e.g. `F16`) to `bench`
turns every format's raw perplexity into `ppl_delta`/`ppl_ratio` against
that reference, which is what "how much quality did this quantization
cost" actually means. Without a baseline, `bench` still reports raw
perplexity, but it isn't comparable to anything outside that one run.

**Same-model validation.** `bench` cross-checks `model_n_params` (from
llama-bench's own CSV output) across every `--gguf` entry in a sweep and
refuses to compare files that don't match — quantization never changes
parameter count, so a mismatch means the files aren't the same base model
quantized differently, and comparing their speed/quality would be
comparing apples to oranges.

**imatrix.** `IQ*` formats need an importance matrix to quantize well;
`quantize --imatrix PATH` passes a calibration file through to
`llama-quantize`'s own `--imatrix` flag. Requesting an `IQ*` format without
one is a hard error (`--allow-iq-without-imatrix` overrides it) — silently
producing a known-poor-quality file was the previous, wrong default.

**CPU feature divergence (`cpu-info`).** llama.cpp binaries print a
`system_info:` line at startup reporting which SIMD features *this
specific build* detected and will use. `cpu-info` cross-checks that
against independent OS-level detection (`sysctl` on macOS, `/proc/cpuinfo`
on Linux), restricted to a controlled vocabulary of SIMD/compute-relevant
flags (AVX, AVX2, FMA, NEON, ...) — not every OS-reported flag, since
`/proc/cpuinfo` lists dozens unrelated to GGML's compute kernels (APIC,
MSR, PAE, ...) that llama.cpp was never going to report on either way.

## CLI reference

Run `uv run quantscope <subcommand> -h` for the authoritative list;
summarized here.

### `bench` — sweep llama-bench across pre-quantized GGUF files (CPU only)

| Flag | Default | Description |
|---|---|---|
| `--llama-bench-bin` | *(required)* | Path to the `llama-bench` binary |
| `--gguf FORMAT=path` | *(required, repeatable)* | One entry per already-quantized GGUF file to benchmark |
| `--n-prompt` | `512` | Prompt-processing test size (tokens) |
| `--n-gen` | `128` | Token-generation test size (tokens) |
| `--threads` | `4` | Thread count passed to llama-bench |
| `--repetitions` | `3` | llama-bench's own internal repetition count (`-r`) |
| `--output` | *(none)* | Write the ranked table to this CSV path, plus `<path>_manifest.json` |
| `--skip-hash` | off | Skip sha256 hashing GGUF files for the manifest (faster on huge files) |
| `--plot` | *(none)* | Write a Pareto-frontier plot to this path |
| `--pareto-epsilon` | `0.02` | Relative tolerance before a difference counts as material, not noise |
| `--llama-perplexity-bin` | *(none)* | Also measure quality via `llama-perplexity`; requires `--perplexity-dataset` too |
| `--perplexity-dataset` | *(none)* | Text file passed to `llama-perplexity -f` |
| `--perplexity-baseline-format` | *(none)* | One of `--gguf`'s formats to treat as the quality reference; adds `ppl_delta`/`ppl_ratio` |

### `estimate-size` — rank formats by approximate storage size, NOT speed

`quantscope estimate-size FORMAT [FORMAT ...]` — positional list of format
names. Always prints a note that this is not a speed estimate; see
[Concepts](#concepts).

### `quantize` — produce missing GGUF formats

| Flag | Default | Description |
|---|---|---|
| `--llama-quantize-bin` | *(required)* | Path to the `llama-quantize` binary |
| `--input` | *(required)* | Source GGUF to quantize from (typically F16/F32) |
| `--output-dir` | *(required)* | Directory to write `<input-stem>-<FORMAT>.gguf` files into |
| *(positional)* | *(required)* | One or more format names to produce |
| `--existing FORMAT=path` | *(none, repeatable)* | Formats already on disk, used with `--skip-existing` |
| `--skip-existing` | off | Skip formats already present in `--existing` |
| `--imatrix PATH` | *(none)* | Calibration file passed to `llama-quantize --imatrix`; required for `IQ*` formats |
| `--allow-iq-without-imatrix` | off | Produce `IQ*` formats even without `--imatrix`, accepting the quality risk |

### `report` — rank an existing CSV on a Pareto frontier

| Flag | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to a CSV with one row per format |
| `--minimize` | *(none, repeatable)* | Column(s) where smaller is better |
| `--maximize` | *(none, repeatable)* | Column(s) where larger is better |
| `--pareto-epsilon` | `0.02` | Relative tolerance before a difference counts as material |
| `--plot` | *(none)* | Write a Pareto-frontier plot to this path |

### `recommend` — filter a bench CSV to formats meeting explicit constraints

| Flag | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to a CSV with one row per format |
| `--minimize` / `--maximize` | `file_size_mb` / `gen_tokens_per_second` | Objectives for the underlying Pareto ranking |
| `--max-size-gb` | *(none)* | Only formats no larger than this |
| `--min-tokens-per-second` | *(none)* | Only formats at least this fast |
| `--max-ppl-delta` | *(none)* | Only formats within this much perplexity of the baseline (needs `bench --perplexity-baseline-format`) |

### `formats` — list what a llama-quantize binary supports

| Flag | Default | Description |
|---|---|---|
| `--llama-quantize-bin` | *(required)* | Path to the `llama-quantize` binary |
| `--imatrix PATH` | *(none)* | If given, `IQ*` formats are listed as applicable |

### `cpu-info` — cross-check CPU features this build uses vs. the OS

| Flag | Default | Description |
|---|---|---|
| `--llama-bench-bin` | *(required)* | Path to the `llama-bench` binary |

## Output formats

**Table** (stdout, always) — a pandas-rendered table, Pareto-optimal rows
sorted first, with a `pareto_optimal` boolean column.

**CSV** (`bench --output` / read by `report`/`recommend --csv`) — one row
per format: `format, gguf_path, file_size_mb, model_size_bytes,
model_n_params, model_type, prompt_tokens_per_second(_stddev),
gen_tokens_per_second(_stddev), perplexity, ppl_delta, ppl_ratio,
pareto_optimal` (columns present depend on what was benchmarked).

**Manifest** (`bench --output`, written as `<path>_manifest.json`) — the
shared environment every format in the sweep ran under (llama.cpp build
commit, CPU/GPU info, backends, thread/batch config, `n_gpu_layers`) plus a
sha256 per GGUF file — the reproducibility half of a bench result,
deliberately kept separate from the CSV since it's identical across every
row in one sweep rather than repeated per-row.

**Plot** (`--plot`, `bench`/`report`) — a PNG scatter plot, Pareto-optimal
points in a different color, each labeled with its format name and (when
available) its quality delta.

## Architecture

```
projects/quantscope/
├── pyproject.toml
├── quantscope/
│   ├── cli.py            argparse entry point: bench, estimate-size, quantize, report, recommend, formats, cpu-info
│   ├── llama_bin.py       subprocess wrapper around llama-bench/llama-quantize/llama-perplexity
│   ├── manifest.py        run-manifest + sha256 generation (reproducibility provenance)
│   ├── cpu_detect.py      two-layer CPU feature detection (system_info line vs. OS, relevant-vocabulary filtered)
│   ├── formats.py         discovers supported formats from llama-quantize --help
│   ├── bench.py           sweeps llama-bench across pre-quantized GGUF files; validates same-model identity
│   ├── estimate.py        heuristic bit-width-based size ranking, no benchmarking
│   ├── quantize.py        produces missing formats via llama-quantize; enforces imatrix for IQ*
│   └── report.py          Pareto-frontier ranking (epsilon-tolerant) + recommend + plotting
├── benchmarks/            real (not stub) sweep runs, one dir per run
└── tests/                 57 tests: mocked subprocess, fixture parsing, Pareto logic, CLI end-to-end
```

See [`ROADMAP.md`](ROADMAP.md) for the design rationale behind each module
and what was reused from this program's own `experiments/` scripts.

## Known limitations

- **No live GPU comparison mode.** CPU is always forced; there's no
  `--device auto` to compare against GPU offload in the same run (a
  deliberate scope choice — see [Why this matters](#why-this-matters)).
- **CPU feature vocabulary (`cpu-info`) is a fixed, curated list**, not
  exhaustive across every architecture — see `cpu_detect.py`'s
  `_RELEVANT_FEATURES`.
- **`recommend`'s ranking is constraint filtering, not a weighted
  optimizer** — it narrows candidates and sorts by your primary objective;
  it doesn't compute a single blended "best overall" score.
- **Epsilon-dominance uses one flat tolerance (2% default), not per-point
  confidence intervals** derived from each format's own stddev — a
  reasonable v1 given llama-bench's own averages already carry real noise,
  but not as statistically rigorous as CI-aware dominance would be.
- **No published package.** No PyPI publish, no GitHub Release with a
  downloadable wheel attached — install from source.

See [`ROADMAP.md`](ROADMAP.md) for the full rationale behind each, and what
came from this program's `experiments/02-llama-cpp/`,
`experiments/04-quantization/`, and `experiments/06-model-comparison/`.

## License

MIT — see [`LICENSE`](LICENSE) (this project's own copy) or the
[repo root](../../LICENSE).
