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
    path: str
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
    models: list[ModelEntry] = field(default_factory=list)


def build_manifest(env: BenchRow, entries: list[ModelEntry]) -> RunManifest:
    """env is any BenchRow from the sweep (they all share the same
    build/CPU/GPU/backend/config fields; only the per-format numbers
    differ), used purely as a source for that shared environment.
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
        models=entries,
    )


def write_manifest(manifest: RunManifest, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2)
