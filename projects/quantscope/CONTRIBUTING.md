# Contributing to quantscope

quantscope is a small, single-maintainer Python project. This doc is a
practical guide to building, testing, and extending it — see
[`ROADMAP.md`](ROADMAP.md) for the design rationale behind why things are
built the way they are, and [`README.md`](README.md) for what it does and
how to run it.

## Building and testing

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). No real
llama.cpp build is needed to run the test suite — everything is mocked or
run against small stub shell scripts (see
[Testing philosophy](#testing-philosophy)).

```bash
cd projects/quantscope
uv sync
uv run pytest -v
uv build          # verifies the package builds into an installable wheel
```

CI (`.github/workflows/quantscope-ci.yml`) runs `uv sync --locked`,
`uv run pytest`, and `uv build` on every push/PR touching
`projects/quantscope/**`. Run the same commands locally before pushing. If
you change `pyproject.toml`'s dependencies, run `uv lock` and commit the
updated `uv.lock` — `--locked` in CI fails otherwise.

## Code layout

See the [Architecture section of README.md](README.md#architecture) for the
package tree. In short: `llama_bin.py` is the only module that shells out to
llama.cpp binaries; every other module (`cpu_detect.py`, `formats.py`,
`bench.py`, `predict.py`, `quantize.py`, `report.py`) is pure logic that
takes already-parsed data in and returns already-parsed data out — this
split is what makes almost the whole test suite mock-free and fast.

## Adding a new subcommand

1. Add the actual logic as a pure function/module first (see `predict.py`
   for the simplest example — no subprocess calls at all).
2. If it needs a llama.cpp binary, add a thin wrapper to `llama_bin.py` that
   runs the subprocess and parses its output, raising `LlamaBinError` on
   failure (see `run_llama_perplexity` for the most recent example — regex
   over combined stdout+stderr, since llama.cpp binaries are inconsistent
   about which stream they write to).
3. Wire it into `cli.py`: a `cmd_<name>` function plus an
   `sub.add_parser(...)` block in `build_parser()`.
4. Add tests at both layers: mock `subprocess.run` (or the wrapper function
   directly, if calling it from a higher layer) for the parsing logic, and
   a CLI end-to-end test using a stub shell script standing in for the
   llama.cpp binary (see `tests/test_cli.py`'s `LLAMA_BENCH_STUB` and
   `LLAMA_PERPLEXITY_STUB` for the pattern — a `#!/bin/sh` script that
   prints canned output and exits).
5. Update `README.md`'s CLI reference table for the new subcommand.

## Testing philosophy

Every test either mocks `subprocess.run` directly, tests pure parsing/
ranking logic against a hand-written fixture string, or runs the full CLI
against small stub shell scripts standing in for `llama-bench`/
`llama-quantize`/`llama-perplexity` — never a real llama.cpp build. This
keeps the suite fast and dependency-free, at the cost of not catching a real
llama.cpp version changing its output format; that's why
`tests/test_cpu_detect.py` and `tests/test_formats.py` test against
hand-captured *realistic* fixture strings rather than synthetic ones, and
why a real end-to-end run against an actual llama.cpp build is a documented
manual pre-release step (see `ROADMAP.md`'s testing strategy section) rather
than something CI depends on.

## Reporting issues / proposing changes

This is a solo-maintainer project inside a larger research monorepo
([efficient-ai-lab](../../README.md)) — open an issue or PR against
[alessandrobessi/efficient-ai-lab](https://github.com/alessandrobessi/efficient-ai-lab)
with `quantscope:` in the title.

## License

By contributing, you agree your contribution is licensed under this
project's [MIT license](LICENSE).
