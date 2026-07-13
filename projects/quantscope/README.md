# quantscope

**Status: core implemented, pre-release.** Installable, tested end-to-end
(`bench`, `predict`, `quantize`, `report`, `formats`, `cpu-info` all work).
Not yet packaged as a tagged release (no CI, no `CONTRIBUTING.md`) — see
[`ROADMAP.md`](ROADMAP.md) for what's left and for the full architecture
rationale behind everything summarized here.

A CLI that answers a question this program's own research raised but never
fully automated: **for this model, on this CPU, which GGUF quantization format
is actually fastest — and what do you give up to get there?**

## Table of contents

- [The problem](#the-problem)
- [What quantscope does](#what-quantscope-does)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [CLI reference](#cli-reference)
- [Output formats](#output-formats)
- [Architecture](#architecture)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Relationship to this repo](#relationship-to-this-repo)
- [License](#license)

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

## Installation

Requires Python 3.11+ and a llama.cpp build (`llama-bench`/`llama-quantize`)
on your machine somewhere — quantscope never bundles or builds llama.cpp
itself; see `../../experiments/02-llama-cpp/scripts/setup_llama_cpp.sh` for
one way to build it.

```bash
cd projects/quantscope
uv sync
uv run quantscope -h
```

Or install it as a standalone tool (e.g. into another project, or with
`pipx`/`uv tool install`):

```bash
uv build                      # produces dist/quantscope-0.1.0-py3-none-any.whl
uv tool install dist/quantscope-0.1.0-py3-none-any.whl
quantscope -h
```

## Quickstart

```bash
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

# re-rank an existing CSV (e.g. from `bench`, or hand-assembled) on different objectives:
uv run quantscope report --csv results.csv --minimize file_size_mb --maximize gen_tokens_per_second --plot frontier.png
```

Example `bench` output (formats are illustrative, not measured on real
hardware):

```
format  file_size_mb  gen_tokens_per_second  pareto_optimal
   F16         14000                     60            True
  Q8_0          8100                     55            True
Q4_K_M          4370                     42            True
  Q4_0          4200                     40            True
```

Example `predict` output (note the caveat — this is never presented as a
substitute for `bench`):

```
rank  format        approx bpw
   1  Q4_0                 4.5
   2  Q4_K_M               4.8
   3  Q5_K_M               5.5
   4  Q8_0                 8.5
   5  F16                 16.0

NOTE: heuristic only, not a measurement: this program's own benchmarking (Weeks 4
and 6) found quantization speed does not correlate cleanly with bit width -- SIMD/
kernel-fit effects can make a nominally larger format faster than a smaller one.
Run `quantscope bench` for a real, measured answer.
```

## Concepts

**Pareto frontier / non-dominated points.** A format is "Pareto-optimal"
(`pareto_optimal: True`) if no other format is at least as good on every
objective (e.g. smaller file size *and* higher tokens/sec) and strictly
better on at least one. A dominated format is never the right choice
regardless of how you weigh size vs. speed vs. quality — quantscope flags
these rather than making you eyeball a table. This reuses the same framing
this program used in Weeks 5, 6, and its architecture decision framework, on
purpose, so it's a familiar mental model rather than new vocabulary.

**`bench` vs. `predict`.** `bench` always measures, by actually running
`llama-bench` against real GGUF files — slower, but a real answer. `predict`
is a heuristic based only on approximate bits-per-weight, with zero
benchmarking — fast, but explicitly labeled low-confidence, because this
program's own data shows quantization speed frequently *doesn't* track bit
width (see [The problem](#the-problem)). The two are deliberately separate
commands so a quick guess is never mistaken for a measurement.

**CPU feature divergence (`cpu-info`).** llama.cpp binaries print a
`system_info:` line at startup reporting which SIMD features *this specific
build* detected and will use (e.g. `AVX2 = 1`). That's different from what
the CPU itself is capable of. `cpu-info` cross-checks the two (via `sysctl`
on macOS, `/proc/cpuinfo` on Linux) and reports any feature the OS says the
CPU has that this build doesn't report using — a build/CPU mismatch a
benchmark number alone can't distinguish from "the CPU doesn't have this
feature."

**imatrix and `IQ*` formats.** `IQ*` quantization formats (e.g. `IQ2_XXS`)
need an importance matrix to quantize well; without one, `llama-quantize`
will still produce a file, but its quality is known to be poor. `formats`
excludes `IQ*` formats from its "applicable" list by default for exactly
this reason — pass `--imatrix` if you have one and want them included in
that list. This is advisory only: `quantize` does not itself check or
enforce it — naming an `IQ*` format there will produce it regardless (see
[Known limitations](#known-limitations)).

## CLI reference

Run `uv run quantscope <subcommand> -h` for the authoritative list;
summarized here.

### `bench` — sweep llama-bench across pre-quantized GGUF files

| Flag | Default | Description |
|---|---|---|
| `--llama-bench-bin` | *(required)* | Path to the `llama-bench` binary |
| `--gguf FORMAT=path` | *(required, repeatable)* | One entry per already-quantized GGUF file to benchmark, e.g. `--gguf Q4_K_M=./m-q4.gguf` |
| `--n-prompt` | `512` | Prompt-processing test size (tokens) |
| `--n-gen` | `128` | Token-generation test size (tokens) |
| `--threads` | `4` | Thread count passed to llama-bench |
| `--repetitions` | `3` | llama-bench's own internal repetition count (`-r`) |
| `--output` | *(none)* | Write the ranked table to this CSV path |
| `--plot` | *(none)* | Write a Pareto-frontier plot (file size vs. gen tokens/sec) to this path |

### `predict` — heuristic ranking, no benchmarking

`quantscope predict FORMAT [FORMAT ...]` — positional list of format names.
Always prints `CONFIDENCE_CAVEAT`; see [Concepts](#concepts).

### `quantize` — produce missing GGUF formats

| Flag | Default | Description |
|---|---|---|
| `--llama-quantize-bin` | *(required)* | Path to the `llama-quantize` binary |
| `--input` | *(required)* | Source GGUF to quantize from (typically F16/F32) |
| `--output-dir` | *(required)* | Directory to write `<input-stem>-<FORMAT>.gguf` files into |
| *(positional)* | *(required)* | One or more format names to produce, e.g. `Q4_K_M Q5_K_M` |
| `--existing FORMAT=path` | *(none, repeatable)* | Formats already on disk, used with `--skip-existing` |
| `--skip-existing` | off | Skip formats already present in `--existing` instead of re-producing them |

### `report` — rank an existing CSV on a Pareto frontier

| Flag | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to a CSV with one row per format (e.g. `bench`'s `--output`, or hand-assembled) |
| `--minimize` | *(none, repeatable)* | Column(s) where smaller is better (e.g. `file_size_mb`) |
| `--maximize` | *(none, repeatable)* | Column(s) where larger is better (e.g. `gen_tokens_per_second`) |
| `--plot` | *(none)* | Write a Pareto-frontier plot to this path |

### `formats` — list what a llama-quantize binary supports

| Flag | Default | Description |
|---|---|---|
| `--llama-quantize-bin` | *(required)* | Path to the `llama-quantize` binary |
| `--imatrix` | off | Include `IQ*` formats in the "applicable" list |

### `cpu-info` — cross-check CPU features this build uses vs. the OS

| Flag | Default | Description |
|---|---|---|
| `--llama-bench-bin` | *(required)* | Path to the `llama-bench` binary (run with no model, to capture its startup `system_info:` line) |

## Output formats

**`bench`/`report` table** (stdout, always) — a pandas-rendered table with
one row per format, sorted Pareto-optimal-first, with a `pareto_optimal`
boolean column (see the example under [Quickstart](#quickstart)).

**CSV** (`bench --output` / read by `report --csv`) — one row per format:
`format, gguf_path, file_size_mb, model_size_bytes, prompt_tokens_per_second,
gen_tokens_per_second, pareto_optimal` (exact columns depend on what was
benchmarked; `report` accepts any CSV with the columns you pass to
`--minimize`/`--maximize`, not just `bench`'s own output).

**Plot** (`--plot`, `bench` or `report`) — a PNG scatter plot, Pareto-optimal
points in a different color from dominated ones, each labeled with its
format name.

## Architecture

```
projects/quantscope/
├── pyproject.toml
├── quantscope/
│   ├── cli.py            argparse entry point: bench, predict, quantize, report, formats, cpu-info
│   ├── llama_bin.py       subprocess wrapper around llama-bench/llama-quantize
│   ├── cpu_detect.py      two-layer CPU feature detection (system_info line vs. OS)
│   ├── formats.py         discovers supported formats from llama-quantize --help
│   ├── bench.py           sweeps llama-bench across pre-quantized GGUF files
│   ├── predict.py         heuristic bit-width-based ranking, no benchmarking
│   ├── quantize.py        produces missing formats via llama-quantize
│   └── report.py          Pareto-frontier ranking + plotting
└── tests/                 26 tests: mocked subprocess, fixture parsing, Pareto logic, CLI end-to-end
```

See [`ROADMAP.md`](ROADMAP.md) for the design rationale behind each module,
what was reused from this program's own `experiments/` scripts, and the
three deliberate deviations from the original plan (installable package
config, imatrix-only format filtering, no `--quality-eval` yet).

## Testing

```bash
uv run pytest -v
uv build          # verifies the package actually builds into an installable wheel
```

Since no real llama.cpp build is assumed to be present, every test either
mocks `subprocess.run` directly (`tests/test_llama_bin.py`), tests pure
parsing/ranking logic against fixture strings (`tests/test_cpu_detect.py`,
`tests/test_formats.py`, `tests/test_report.py`), or runs the full CLI
(`tests/test_cli.py`) against small stub shell scripts standing in for
`llama-bench`/`llama-quantize`. A real end-to-end run against an actual
llama.cpp build and GGUF file is a manual pre-release step, not something
the automated suite depends on.

## Known limitations

- **Format-applicability filtering only exists in `formats`, and is
  imatrix-based only.** `formats --imatrix` controls what it lists as
  "applicable," but `quantize` doesn't consult this itself — it will
  produce whatever format you name, `IQ*` included, without an imatrix or
  an architecture-compatibility check. Treat `formats`' output as advisory
  when choosing what to pass to `quantize`, not as an enforced guardrail.
- **No `--quality-eval` mode.** Wrapping `llama-perplexity` to add a
  measured quality axis (not just speed/size) to the Pareto ranking was
  planned but not built.
- **Not validated against a real llama.cpp build.** None was available in
  the environment this was built in; all tests use stub binaries. Numbers
  produced against a real `llama-bench`/`llama-quantize` haven't been
  sanity-checked against this program's own Week 4/6 findings yet.
- **CPU feature name mapping (`cpu-info`) is a small, non-exhaustive alias
  table**, not a complete mapping between every OS-reported flag name and
  every llama.cpp-reported one — expect some real divergences to go
  unflagged on CPUs/features outside the common ones this program worked
  with (see `quantscope/cpu_detect.py`'s `_ALIASES`).

See [`ROADMAP.md`](ROADMAP.md) for the full list of what M4 still needs (CI,
`CONTRIBUTING.md`, tagged release).

## Relationship to this repo

Generalizes the sweep-over-GGUF → CSV → pandas/matplotlib pipeline built in
`../../experiments/02-llama-cpp/`, `../../experiments/04-quantization/`, and
`../../experiments/06-model-comparison/` into a reusable, installable tool
instead of one-off experiment scripts. See `ROADMAP.md` for the exact reuse
plan and what's cited from where.

## License

MIT — see [`LICENSE`](LICENSE) (this project's own copy) or the
[repo root](../../LICENSE).
