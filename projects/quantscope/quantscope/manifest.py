"""Run-manifest generation: the environment/provenance half of a bench
sweep's output, kept separate from the per-format CSV rows (bench.py's
FormatResult) since it's identical across every format in one sweep — CPU,
GPU, llama.cpp build, thread/batch config don't vary row to row, so
repeating them per-row would be redundant and, worse, easy to silently drop
(which is exactly what quantscope did before this module existed: only
avg_ts survived into the final result, discarding build commit, CPU/GPU
info, backend, batch config, and even model_n_params).

A benchmark result without this is not reproducible by someone else, or by
the same person six months later.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

from quantscope import __version__
from quantscope.llama_bin import BenchRow


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Hashes a file in fixed-size chunks so multi-gigabyte GGUF files never
    need to be read into memory whole.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelEntry:
    format: str
    # Basename only, not the full path -- an earlier version stored
    # whatever path the sweep was invoked with, which for a real run is
    # typically some machine-specific temp directory (e.g.
    # "/Users/alice/.cache/tmp/qs-bench/..."). That leaks nothing useful for
    # reproducibility (the sha256 below is the actual identity anchor) while
    # publishing a committed benchmark artifact that references a directory
    # only the original machine ever had.
    filename: str
    sha256: str
    model_size_bytes: int
    model_n_params: int


@dataclass
class RunManifest:
    quantscope_version: str
    timestamp: str
    os: str
    llama_cpp_build_commit: str
    cpu_info: str
    gpu_info: str
    backends: str
    n_threads: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    # Sweep parameters needed to reconstruct the experiment -- without these
    # a committed benchmark's CSV numbers can't be reproduced even with the
    # same hardware and GGUF files, since n_prompt/n_gen/rounds all affect
    # the result.
    n_prompt: int = 0
    n_gen: int = 0
    rounds: int = 0
    # The actual per-round visiting order used, format names in the order
    # llama-bench was invoked -- the direct evidence that format order was
    # randomized (and how), not just an assertion that it was.
    round_orders: list[list[str]] = field(default_factory=list)
    # Perplexity phase provenance -- absent (empty/zero) when bench wasn't
    # run with quality evaluation at all.
    perplexity_dataset_sha256: str = ""
    perplexity_baseline_format: str = ""
    perplexity_threads: int = 0
    perplexity_n_gpu_layers: int = 0
    # The Pareto configuration actually used to produce this run's
    # `pareto_optimal` column -- without this, the column can't be
    # reconstructed or sanity-checked later (e.g. after report.py's
    # epsilon-handling changes).
    pareto_minimize: list[str] = field(default_factory=list)
    pareto_maximize: list[str] = field(default_factory=list)
    pareto_epsilon: float = 0.0
    pareto_ppl_absolute_tolerance: float | None = None
    models: list[ModelEntry] = field(default_factory=list)


def build_manifest(
    env: BenchRow,
    entries: list[ModelEntry],
    n_prompt: int = 0,
    n_gen: int = 0,
    rounds: int = 0,
    round_orders: list[list[str]] | None = None,
    perplexity_dataset_sha256: str = "",
    perplexity_baseline_format: str = "",
    perplexity_threads: int = 0,
    perplexity_n_gpu_layers: int = 0,
) -> RunManifest:
    """env is any BenchRow from the sweep (they all share the same
    build/CPU/GPU/backend/config fields; only the per-format numbers
    differ), used purely as a source for that shared environment.

    Pareto configuration (pareto_minimize/maximize/epsilon) isn't known at
    sweep() time -- it's chosen at the CLI/report layer -- so callers set
    those three fields on the returned RunManifest directly before writing
    it (see cli.py's cmd_bench).
    """
    return RunManifest(
        quantscope_version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        os=platform.platform(),
        llama_cpp_build_commit=env.build_commit,
        cpu_info=env.cpu_info,
        gpu_info=env.gpu_info,
        backends=env.backends,
        n_threads=env.n_threads,
        n_batch=env.n_batch,
        n_ubatch=env.n_ubatch,
        n_gpu_layers=env.n_gpu_layers,
        n_prompt=n_prompt,
        n_gen=n_gen,
        rounds=rounds,
        round_orders=round_orders or [],
        perplexity_dataset_sha256=perplexity_dataset_sha256,
        perplexity_baseline_format=perplexity_baseline_format,
        perplexity_threads=perplexity_threads,
        perplexity_n_gpu_layers=perplexity_n_gpu_layers,
        models=entries,
    )


def write_manifest(manifest: RunManifest, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2)
