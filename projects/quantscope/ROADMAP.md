# quantscope — Development Roadmap

**Status: M0-M6 implemented — `quantscope/v0.2.1` tagged.** The architecture
below reflects what was actually built (see `quantscope/`), not just a plan.
One deliberate deviation from the original plan remains: `pyproject.toml`
builds an installable package (`[build-system]`/hatchling, `project.scripts`)
rather than using `[tool.uv] package = false`.

## v0.2.1: fixes from a second external review

A second external review — this time of v0.2.0's real benchmark, Pareto
ranking, and manifest — found seven more gaps, all addressed:

1. **Benchmark order was confounded.** Each format was benchmarked then
   perplexity-evaluated immediately, then the next format — sequential, not
   randomized, so thermal throttling/background load/OS caching drift
   correlated with format identity instead of being spread evenly across all
   formats. Fixed: `sweep()` now runs `--rounds` (default 10) independently
   shuffled rounds, one `llama-bench -r 1` call per (round, format) pair,
   aggregating mean/stddev *across rounds* for real between-round
   uncertainty instead of within-round repetition noise; `--seed` makes the
   shuffle reproducible. Perplexity (deterministic, much slower) stays a
   separate phase, run once per format after all rounds finish, not repeated
   per round.
2. **Epsilon-dominance used one relative tolerance for everything**,
   including `ppl_delta` — meaningless around the baseline format's own
   entry, where `ppl_delta = 0` and a relative tolerance around zero is
   zero, so the advertised noise protection silently didn't apply to the one
   row that needed it most. Fixed: `_tolerance_band()` supports a fixed
   *absolute* tolerance per objective (`--ppl-absolute-tolerance`, default
   0.05) layered on top of the existing relative epsilon; `bench`, `report`,
   and `recommend` all thread it through.
3. **`_validate_same_model` only checked `model_n_params`** — necessary but
   not sufficient, since a base model and its instruct fine-tune (or two
   unrelated architectures) can share a parameter count. Fixed: added a soft
   secondary check on a normalized `model_type` prefix (architecture + size,
   with the quant-format suffix stripped), raising `ModelIdentityError` on
   mismatch. Full GGUF-metadata validation (tokenizer hash, tensor shapes)
   is deferred — see "Explicitly deferred" below.
4. **`run_llama_perplexity` never forced `-ngl 0`**, unlike
   `run_llama_bench` — a Metal/CUDA-enabled build could silently evaluate
   perplexity with GPU offload despite quantscope's "CPU is always forced"
   claim. Fixed: perplexity now always passes `-ngl 0` too.
5. **Perplexity's own uncertainty was discarded.** llama-perplexity reports
   `Final estimate: PPL = X +/- Y`, but only `X` was kept. Fixed:
   `run_llama_perplexity` returns `PerplexityResult(value, error)`;
   `perplexity_error` is now a CSV column.
6. **The manifest was missing key provenance** — `n_prompt`, `n_gen`,
   `rounds`, the actual per-round visiting order, the perplexity dataset's
   sha256, and the Pareto objectives/epsilon/tolerance used — and recorded
   absolute machine-specific temp paths instead of basenames, plus a stale
   `quantscope_version` left over from a prior release. Fixed:
   `RunManifest` now records all of the above (`round_orders`,
   `perplexity_dataset_sha256`, `pareto_minimize`/`pareto_maximize`/
   `pareto_epsilon`/`pareto_ppl_absolute_tolerance`, etc.); `ModelEntry`
   stores a basename, never an absolute path.
7. **Two engineering nits.** `bench` hashed every GGUF file even when
   `--output` wasn't given (wasted work on large files for a manifest that
   was never going to be written); `report --csv` allowed zero
   minimize/maximize columns, which crashed downstream instead of failing
   clearly. Fixed: hashing is now gated on `--output` being present
   (`--skip-hash` still works regardless); `rank_table` raises a clear
   `ValueError` when both objective lists are empty.
8. **The committed benchmark referenced a dataset file that didn't
   exist.** The reproduce command named `wiki-like-sample.txt`, but it was
   never actually committed. Fixed: re-ran the full benchmark end to end
   with a real, committed dataset (a ~150KB public-domain excerpt from
   Project Gutenberg's *Pride and Prejudice*) using the new randomized-round
   architecture — see
   [`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/)
   for the updated methodology and numbers. The rerun surfaced a real,
   unplanned finding: round-to-round variance for a 0.5B model under the
   new one-process-per-round design is large (60-90% of the mean),
   because launching a fresh `llama-bench` process every round pays a
   model-load/cache-warmup cost that v0.2.0's single-invocation-with-
   internal-repetition design didn't. v0.2.0's "Q8_0 is fastest" headline
   didn't survive the rerun — the honest result is that this run can't
   support a confident speed-ranking claim at all, documented as such
   rather than replaced with an equally overconfident new headline. A
   candidate fix (`llama-bench -r N` within each round, not just `-r 1`)
   is noted below under "Explicitly deferred," not implemented in v0.2.1.

## v0.2.0: fixes from external review

An external review of v0.1.0 found seven real correctness/design gaps and
asked for a real benchmark instead of stub-only evidence. All addressed:

1. **`bench` never disabled GPU offloading.** llama-bench's own default
   (`-ngl -1`) maximally offloads to GPU if the build supports one; on a
   machine with Metal/CUDA compiled in (common — this program's own dev
   environment has Metal), quantscope was silently answering "which format
   is fastest with GPU offload" while claiming to profile CPU. Fixed:
   `run_llama_bench` now always passes `-ngl 0`, no flag, no way to turn it
   off — matching the user's explicit choice that quantscope should target
   CPU-only inference specifically, not general local hardware.
2. **`cpu_detect`'s divergence check produced false positives.** It flagged
   *every* OS-reported feature absent from llama.cpp's own output as
   "unused by the build," including dozens of `/proc/cpuinfo` flags
   (APIC, MSR, PAE, MTRR, ...) that have nothing to do with GGML's compute
   kernels. Fixed: `unused_but_supported` now only compares a controlled,
   SIMD-relevant vocabulary (`_RELEVANT_FEATURES`) — an OS flag outside it
   is "not compared," not "unused."
3. **No model-identity validation.** `bench --gguf FORMAT=path` accepted
   arbitrary files with no check that they were the same base model
   quantized differently. Fixed: `bench` now cross-checks `model_n_params`
   (from llama-bench's own CSV) across every format and refuses to compare
   mismatched files — see `ModelIdentityError`. (A real benchmark run
   caught a real bug in this very fix: `model_type` embeds the quantization
   format itself, e.g. `"qwen2 1B Q8_0"` vs. `"qwen2 1B F16"` for the exact
   same model, so it's deliberately *not* part of the comparison — only
   `model_n_params` is reliable.)
4. **Benchmark provenance and uncertainty were discarded**, keeping only
   `avg_ts` and dropping build commit, CPU/GPU info, backend, batch config,
   and even `model_n_params`. Fixed: `manifest.py` writes a
   `<output>_manifest.json` alongside every `bench --output` CSV, capturing
   the full shared environment plus a sha256 per GGUF file; `BenchRow`/
   `FormatResult` now carry `stddev` alongside every average.
5. **Pareto ranking treated every benchmark average as exact**, letting
   ranking flip on measurement noise. Fixed: `_dominates` now takes a
   relative tolerance (`--pareto-epsilon`, default 2%) — a difference
   within tolerance no longer counts as domination.
6. **No real `--imatrix` support.** `quantize` had no way to pass a
   calibration file to `llama-quantize`, and would happily produce `IQ*`
   formats without one despite `formats --imatrix` (a bare boolean)
   implying otherwise. Fixed: `quantize --imatrix PATH` passes it through
   to `llama-quantize --imatrix`; `IQ*` without one is now a hard error
   (`ImatrixRequiredError`, overridable via `--allow-iq-without-imatrix`).
7. **Perplexity was reported as raw "quality," not relative to anything**,
   and the plot only ever showed 2 of a possible 3 Pareto dimensions (a
   point could be Pareto-optimal because of an invisible perplexity
   advantage that never showed up on the size/speed chart). Fixed:
   `bench --perplexity-baseline-format` computes `ppl_delta`/`ppl_ratio`
   against a reference format; `plot_frontier`'s `quality_col` annotates
   each point with its quality value so 2D charts don't hide a 3D decision.
8. **"Pareto-optimal" wasn't always an answer** — when every smaller format
   is also slower, *all* formats can be simultaneously Pareto-optimal
   (confirmed in the real benchmark below: all 8 formats landed on the
   frontier). Added `recommend`, which filters to what actually satisfies
   stated constraints (`--max-size-gb`, `--min-tokens-per-second`,
   `--max-ppl-delta`).
9. **`predict` structurally contradicted quantscope's own thesis** — it
   ranked by bit width and called that a "prediction," while the whole
   point of the project is that bit width doesn't predict speed reliably.
   Renamed to `estimate-size` (module `predict.py` → `estimate.py`,
   `Prediction`/`predicted_rank` → `SizeEstimate`/`size_rank`), explicitly
   documented as a storage-size ranking, never a speed estimate.
10. **No real end-to-end run.** Added
    [`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/):
    a real `llama-bench`/`llama-quantize`/`llama-perplexity` sweep across 8
    formats of Qwen2.5-0.5B-Instruct on an Apple M4 CPU. Found, concretely,
    that Q8_0 is faster than four smaller K-quant formats — the project's
    own "measure, don't guess" thesis demonstrated with real numbers, not
    just argued for.

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
│   ├── cli.py                    entry point: bench, estimate-size, quantize, report, recommend, formats, cpu-info
│   ├── cpu_detect.py              two-layer CPU feature detection, relevant-vocabulary filtered (see below)
│   ├── formats.py                 query llama-quantize for supported formats; filter by imatrix availability
│   ├── llama_bin.py               subprocess wrapper (evolves llama_cpp_runner.py); also wraps llama-perplexity
│   ├── manifest.py                run-manifest (build/CPU/GPU/config provenance) + sha256 generation
│   ├── bench.py                   measure: sweep llama-bench (+ optional llama-perplexity), validate same-model identity
│   ├── estimate.py                heuristic size-only ranking, no benchmarking, explicitly not a speed estimate
│   ├── quantize.py                auto-produce missing formats via llama-quantize; enforces imatrix for IQ*
│   └── report.py                 Pareto-frontier ranking (epsilon-tolerant), recommend, table + plot output
├── benchmarks/                    real (not stub) sweep runs, one dir per run
├── tests/                        57 tests: mocked-subprocess, fixture-based parsing, Pareto logic, CLI end-to-end via stub scripts
├── CONTRIBUTING.md
└── ROADMAP.md, README.md, LICENSE
```

`cli.py` grew subcommands beyond the original plan: `formats` (list what a
`llama-quantize` binary supports/excludes), `cpu-info` (print the
`cpu_detect` divergence check directly), and `recommend` (v0.2.0, constraint
filtering) — the first two were internal building blocks the plan already
required; exposing them as subcommands cost little and makes each
independently useful/testable rather than only reachable as a side effect
of `bench`.

**`cpu_detect.py`** — two-layer detection: parse llama.cpp's own reported
feature line at startup (ground truth for what *this specific build* actually
does) and cross-check it against independent OS-level detection (`sysctl` on
macOS, `/proc/cpuinfo` on Linux), restricted to `_RELEVANT_FEATURES` (v0.2.0
fix — see above; without this restriction, every OS-reported flag outside
llama.cpp's small vocabulary was a false positive). Divergence within that
vocabulary — CPU supports a feature the build doesn't report using — is
surfaced directly.

**`formats.py`** — queries the installed `llama-quantize` binary for its
currently supported format list at runtime rather than hardcoding one, since
llama.cpp adds and renames formats across versions. Applicability filtering
is imatrix-based (`IQ*` formats need one to quantize well and are excluded
from the "applicable" list unless `--imatrix PATH` is given) — this list is
advisory; `quantize.py` (not `formats.py`) is what actually enforces it (see
below). Full GGUF-metadata/architecture-based filtering (e.g. excluding
formats incompatible with a specific model architecture) was not built.

**`bench.py` / `estimate.py` split** — `bench` always measures via
`llama-bench`; `estimate.py` (renamed from `predict.py` in v0.2.0 — see
above) is an explicitly labeled size-only heuristic, no benchmark run, no
speed claim — direct response to this program's own finding that naive
bit-width heuristics don't predict speed reliably. `bench.py` also now
validates (`_validate_same_model`) that every `--gguf` entry in a sweep
reports the same `model_n_params` before returning anything, and builds a
`manifest.RunManifest` (build/CPU/GPU/config provenance + per-file sha256)
alongside the per-format `FormatResult` list.

**`quantize.py`** — shells out to `llama-quantize` to produce formats missing
from a model's existing GGUF set, so `bench`/`report` can cover the full
candidate space without requiring the user to pre-generate every format by
hand. Validates every requested format *before* quantizing any of them
(`_requires_imatrix`): an `IQ*` format without `--imatrix` and without
`--allow-iq-without-imatrix` raises `ImatrixRequiredError` up front rather
than wasting time quantizing earlier formats before failing on a later one.

**`report.py`** — ranks formats on a Pareto frontier (speed vs. memory vs.
quality), reusing the same "dominated" / "Pareto-optimal" language this
program used in Weeks 5, 6, and the architecture decision framework, so a
user already familiar with that framing gets the same mental model here.
`_dominates` takes a relative `epsilon` (default 2%, `DEFAULT_EPSILON`) so a
difference within measurement noise doesn't flip Pareto membership.
`recommend()` filters an already-ranked table to rows meeting explicit
constraints (max size, min speed, max quality loss) — the direct answer to
"Pareto-optimal isn't always a single recommendation" (see above).
`plot_frontier`'s `quality_col` param annotates each point with a third
value (e.g. `ppl_delta`) when the Pareto ranking used more than the plotted
x/y axes, so a 2D chart doesn't hide a 3D decision.

**Quality evaluation** — `bench.sweep()` takes optional
`llama_perplexity_bin`/`perplexity_dataset`/`perplexity_baseline_format`
arguments; when the first two are given, it runs
`llama_bin.run_llama_perplexity` (parses llama-perplexity's
`Final estimate: PPL = X.XXXX` summary line) per format. If
`perplexity_baseline_format` is also given, every result's raw perplexity
becomes `ppl_delta`/`ppl_ratio` against that reference — llama.cpp's own
docs are explicit that a bare perplexity number is only meaningful relative
to a same-model, same-tokenizer baseline, not as an absolute score. Kept
opt-in (not run by default) because perplexity evaluation over a real
dataset is much slower than a speed benchmark. **Note:** there is no single
`--quality-eval` flag — v0.1.0's docs referenced one that didn't exist;
v0.2.0's docs describe the actual two-or-three-flag mechanism instead of a
shorthand name for it.

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| **M0** | `bench` CLI wrapping `llama-bench` over a fixed/supplied format list, ranked table output | **Done** — `cmd_bench`/`sweep`, tested against a stub `llama-bench` binary end-to-end (`test_cli.py::test_cli_bench_end_to_end`); reproducing Week 4's actual numbers requires a real llama.cpp build, not available in this environment, so that specific exit criterion is unverified rather than failed |
| **M1** | CPU feature detection + heuristic `predict` mode | **Done** — `cpu_detect.py`, `estimate.py` (renamed in v0.2.0); feature-line parsing tested against hand-written fixture strings (not yet against multiple *real* captured llama.cpp version strings — only one representative fixture) |
| **M2** | `quantize` (auto-produce missing formats) + format applicability filtering | **Done**, now with real `--imatrix` enforcement (v0.2.0) |
| **M3** | Pareto report/plot polish; optional `--quality-eval` wrapping `llama-perplexity` | **Done** — Pareto ranking/plotting (`report.py`) now epsilon-tolerant; quality evaluation implemented and, as of v0.2.0, baseline-relative (`ppl_delta`/`ppl_ratio`) rather than a raw number |
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` (`uv build`) | **Done** — `.github/workflows/quantscope-ci.yml`, `CONTRIBUTING.md`, a `--version` flag, git tag `quantscope/v0.1.0`. No PyPI publish and no GitHub Release with a downloadable wheel attached — deliberately held back pending the maintainer's own call on publishing publicly (same reasoning as `llmpace`'s M4) |
| **M5** | v0.2.0: fixes from external review — force CPU, fix CPU-feature false positives, validate model identity, preserve provenance/uncertainty, real imatrix support, baseline-relative quality, `recommend`, rename `predict`, a real benchmark | **Done** — see [v0.2.0: fixes from external review](#v020-fixes-from-external-review) above |
| **M6** | v0.2.1: fixes from a second external review — randomized multi-round benchmark architecture, absolute Pareto tolerance for zero baselines, stronger model-identity check, CPU-only enforcement on perplexity too, perplexity uncertainty, manifest completeness, engineering nits, a real benchmark rerun with a committed dataset | **Done** — see [v0.2.1: fixes from a second external review](#v021-fixes-from-a-second-external-review) above |

## Testing strategy

- **Mocked-subprocess tests** for `llama_bin.py` (`tests/test_llama_bin.py`)
  using canned `llama-bench`/`llama-quantize`/`llama-perplexity` output — no
  real llama.cpp build needed. Includes an explicit assertion
  (`test_run_llama_bench_always_forces_ngl_zero`) that `-ngl 0` is always
  in the constructed command.
- **Fixture-based parsing tests** for the feature-line parser
  (`tests/test_cpu_detect.py` — now including a test proving irrelevant
  OS flags like APIC/MSR/PAE are never flagged) and the `--help`
  format-list parser (`tests/test_formats.py`).
- **Pareto/non-dominated-sort tests** (`tests/test_report.py`), including
  epsilon-dominance cases distinguishing a real size/speed tradeoff from a
  noise-level difference that shouldn't affect ranking, plus `recommend`
  and `plot_frontier`'s quality-annotation path.
- **Model-identity and manifest tests** (`tests/test_bench.py`): a
  same-model sweep passes, a mismatched-`model_n_params` sweep raises
  `ModelIdentityError`, sha256 hashing is verified against a known digest,
  and `ppl_delta`/`ppl_ratio` are checked against hand-computed values.
- **imatrix enforcement tests** (`tests/test_quantize.py`): `IQ*` without
  an imatrix raises before any subprocess call is made; with `--imatrix` or
  `--allow-iq-without-imatrix` it proceeds.
- **CLI end-to-end tests** (`tests/test_cli.py`) using small stub shell
  scripts standing in for `llama-bench`/`llama-quantize`/`llama-perplexity`,
  covering every subcommand through `cli.main()` — including the
  model-identity-mismatch error, the imatrix-required error, and
  `recommend`'s filtering. This is the "no real llama.cpp build needed"
  tier; a real end-to-end run (see
  [`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/))
  was performed manually as part of v0.2.0 and remains a manual pre-release
  step, not something CI depends on.
- **CI** (`.github/workflows/quantscope-ci.yml`) runs `uv sync --locked`,
  `uv run pytest`, and `uv build` on every push/PR touching
  `projects/quantscope/**` — confirmed passing against GitHub's own runners,
  not just locally.

## Explicitly deferred (stretch goal, not v1.0 scope)

Confirming which code path actually executes per quantization format
(REPACK-path confirmation) needs an instrumented llama.cpp fork or a real
A/B toggle — a genuinely open-ended research effort, not a v1.0 engineering
task. quantscope ships with the same "circumstantial, log-line-only"
evidence caveat this program's own research already carries, documented
honestly rather than either overclaiming the mechanism or blocking the
tool's real value (which is the sweep-and-rank workflow, not the
mechanistic explanation) on resolving it.

**Deep GGUF model-identity validation** (v0.2.1). `_validate_same_model`
checks `model_n_params` (exact) and a normalized `model_type` prefix (soft)
— together enough to catch the concrete failure mode the second review
raised (a base model and its fine-tune sharing a parameter count), but
still not a guarantee. A stronger future validator would parse GGUF
metadata directly and compare tokenizer hash and per-tensor shapes, which
needs a GGUF-binary-parsing dependency this project doesn't currently have
(see `pyproject.toml`) and is a large-enough addition to warrant its own
pass rather than folding it into a review-response cycle.

Not addressed in v0.2.1, left for a future pass if real need arises:
per-point confidence-interval-aware dominance (still one flat
epsilon/absolute-tolerance band, not each format's own stddev), a
`--device auto` GPU-comparison mode (CPU-only remains the deliberate
scope), `recommend` computing a single weighted "best overall" score
rather than filtering candidates, and reducing per-round measurement noise
via `llama-bench -r N` *within* each shuffled round (not just `-r 1`) —
the real benchmark rerun (see
[v0.2.1's item 8](#v021-fixes-from-a-second-external-review)) found
round-to-round variance large enough that this is now a concrete, known
gap rather than a hypothetical one, but implementing it needs its own
pass to decide the right N and re-validate the aggregation math, not a
same-cycle fix.

## License

MIT — see [`../../LICENSE`](../../LICENSE); each project carries its own copy
of the same license text.
