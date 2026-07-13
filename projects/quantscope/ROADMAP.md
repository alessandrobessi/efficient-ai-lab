# quantscope — Development Roadmap

**Status: M0-M4 implemented — `quantscope/v0.1.0` tagged.** The architecture
below reflects what was actually built (see `quantscope/`), not just a plan.
Two deliberate deviations from the original plan remain, noted inline where
they occur: `pyproject.toml` builds an installable package
(`[build-system]`/hatchling, `project.scripts`) rather than using
`[tool.uv] package = false`; and format applicability filtering is
imatrix-based only, not full GGUF-metadata/architecture-based.
`--quality-eval` (wrapping `llama-perplexity`) — previously deferred — was
built as part of closing out M3.

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

Independent Python package (`pyproject.toml` + `uv`, `requires-python >= 3.11`).
**Deviation:** builds as a real installable package (`[build-system]` via
hatchling, `project.scripts` entry point `quantscope`), not
`[tool.uv] package = false` as originally planned — `uv build` producing a
wheel is one of M4's own exit criteria, so the package needed to be
installable from the start rather than retrofitted later.

```
projects/quantscope/
├── pyproject.toml
├── quantscope/
│   ├── cli.py                    entry point: bench, predict, quantize, report, formats, cpu-info subcommands
│   ├── cpu_detect.py              two-layer CPU feature detection (see below)
│   ├── formats.py                 query llama-quantize for supported formats; filter by imatrix availability
│   ├── llama_bin.py               subprocess wrapper (evolves llama_cpp_runner.py); also wraps llama-perplexity
│   ├── bench.py                   measure: sweep llama-bench (+ optional llama-perplexity) across formats
│   ├── predict.py                 heuristic-only, no benchmarking, confidence-labeled
│   ├── quantize.py                auto-produce missing formats via llama-quantize
│   └── report.py                 Pareto-frontier ranking, table + plot output
├── tests/                        34 tests: mocked-subprocess, fixture-based parsing, Pareto logic, CLI end-to-end via stub scripts
├── CONTRIBUTING.md
└── ROADMAP.md, README.md, LICENSE
```

`cli.py` grew two subcommands beyond the original plan: `formats` (list what
a `llama-quantize` binary supports/excludes) and `cpu-info` (print the
`cpu_detect` divergence check directly) — both were internal building blocks
the plan already required; exposing them as subcommands cost little and
makes each independently useful/testable rather than only reachable as a
side effect of `bench`.

**`cpu_detect.py`** — two-layer detection: parse llama.cpp's own reported
feature line at startup (ground truth for what *this specific build* actually
does) and cross-check it against independent OS-level detection (`sysctl` on
macOS, `/proc/cpuinfo` on Linux). Divergence between the two — CPU supports a
feature the build doesn't report using — is surfaced directly, which is a
diagnostic this program never had.

**`formats.py`** — queries the installed `llama-quantize` binary for its
currently supported format list at runtime rather than hardcoding one, since
llama.cpp adds and renames formats across versions. **Deviation:**
applicability filtering as implemented is imatrix-based only (`IQ*` formats
need one to quantize well and are excluded unless `--imatrix` is passed) —
full GGUF-metadata/architecture-based filtering (e.g. excluding formats
incompatible with a specific model architecture) was not built; every format
`llama-quantize` reports is otherwise treated as applicable.

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

**`--quality-eval`** — `bench.sweep()` takes optional `llama_perplexity_bin`/
`perplexity_dataset` arguments; when both are given, it runs
`llama_bin.run_llama_perplexity` (parses llama-perplexity's
`Final estimate: PPL = X.XXXX` summary line) per format and adds a
`perplexity` field to `FormatResult`, which `cli.py`'s `cmd_bench` then adds
to the Pareto ranking's `minimize` objectives alongside file size. Kept as
opt-in (not run by default) because perplexity evaluation over a real
dataset is much slower than a speed benchmark — a plain `bench` sweep stays
fast, and quality is added deliberately, not automatically.

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| **M0** | `bench` CLI wrapping `llama-bench` over a fixed/supplied format list, ranked table output | **Done** — `cmd_bench`/`sweep`, tested against a stub `llama-bench` binary end-to-end (`test_cli.py::test_cli_bench_end_to_end`); reproducing Week 4's actual numbers requires a real llama.cpp build, not available in this environment, so that specific exit criterion is unverified rather than failed |
| **M1** | CPU feature detection + heuristic `predict` mode | **Done** — `cpu_detect.py`, `predict.py`; feature-line parsing tested against hand-written fixture strings (not yet against multiple *real* captured llama.cpp version strings — only one representative fixture) |
| **M2** | `quantize` (auto-produce missing formats) + format applicability filtering | **Done**, with the imatrix-only filtering deviation noted above |
| **M3** | Pareto report/plot polish; optional `--quality-eval` wrapping `llama-perplexity` | **Done** — Pareto ranking/plotting (`report.py`); `--quality-eval` implemented (`run_llama_perplexity`, `bench.sweep`'s optional args, `cmd_bench`'s validation that both flags are given together), tested at both the subprocess-parsing layer and full CLI end-to-end via a stub `llama-perplexity` script |
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` (`uv build`) | **Done** — `.github/workflows/quantscope-ci.yml` (`uv sync --locked`, `pytest`, `uv build` on every push/PR touching `projects/quantscope/**`), `CONTRIBUTING.md`, a `--version` flag, git tag `quantscope/v0.1.0`. No PyPI publish and no GitHub Release with a downloadable wheel attached — that step was deliberately held back pending the maintainer's own call on publishing publicly (same reasoning as `llmpace`'s M4) |

## Testing strategy

- **Mocked-subprocess tests** for `llama_bin.py` (`tests/test_llama_bin.py`)
  using canned `llama-bench`/`llama-quantize`/`llama-perplexity` output — no
  real llama.cpp build needed.
- **Fixture-based parsing tests** for the feature-line parser
  (`tests/test_cpu_detect.py`) and the `--help` format-list parser
  (`tests/test_formats.py`).
- **Pareto/non-dominated-sort tests** (`tests/test_report.py`) against a
  hand-computed expected set (4 points, one deliberately dominated).
- **CLI end-to-end tests** (`tests/test_cli.py`) using small stub shell
  scripts standing in for `llama-bench`/`llama-quantize`/`llama-perplexity`,
  covering every subcommand through `cli.main()` rather than only the
  underlying functions — including `bench --quality-eval`'s full path and
  the validation error when only one of the two required flags is given.
  This is the "no real llama.cpp build needed" tier the roadmap called for;
  a real end-to-end run against an actual llama.cpp build and GGUF was not
  performed in this environment (none was available) and remains a genuine
  manual pre-release step, as planned.
- **CI** (`.github/workflows/quantscope-ci.yml`) runs `uv sync --locked`,
  `uv run pytest`, and `uv build` on every push/PR touching
  `projects/quantscope/**` — confirmed passing against GitHub's own runners,
  not just locally.

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
