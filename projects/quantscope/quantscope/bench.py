"""Sweeps llama-bench across a set of GGUF files (one per quantization
format) and collects one row per format — the measured half of quantscope,
as opposed to predict.py's heuristic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from quantscope.llama_bin import run_llama_bench, run_llama_perplexity


@dataclass
class FormatResult:
    format: str
    gguf_path: str
    file_size_mb: float
    model_size_bytes: int
    prompt_tokens_per_second: float
    gen_tokens_per_second: float
    # None unless quality_eval's binary+dataset args are both given to
    # sweep() -- deliberately optional, since perplexity evaluation is much
    # slower than a speed benchmark and not every sweep needs a quality axis.
    perplexity: float | None = None


def sweep(
    llama_bench_bin: str,
    gguf_paths: dict[str, str],
    n_prompt: int = 512,
    n_gen: int = 128,
    threads: int = 4,
    repetitions: int = 3,
    llama_perplexity_bin: str | None = None,
    perplexity_dataset: str | None = None,
) -> list[FormatResult]:
    """Runs llama-bench once per (format, gguf_path) pair. gguf_paths maps a
    format name (e.g. "Q4_K_M") to the path of an already-quantized GGUF
    file in that format — producing missing ones is quantize.py's job, kept
    separate so a sweep over pre-existing files never silently quantizes.

    If both llama_perplexity_bin and perplexity_dataset are given, also runs
    llama-perplexity per format and records the result as a measured quality
    axis alongside speed/size — this is `bench --quality-eval`'s mechanism.
    """
    results = []
    for fmt, path in gguf_paths.items():
        rows = run_llama_bench(llama_bench_bin, path, n_prompt, n_gen, threads, repetitions)
        prompt_tps = next((r.avg_tokens_per_second for r in rows if r.n_prompt > 0 and r.n_gen == 0), 0.0)
        gen_tps = next((r.avg_tokens_per_second for r in rows if r.n_gen > 0 and r.n_prompt == 0), 0.0)
        model_size = next((r.model_size_bytes for r in rows if r.model_size_bytes), 0)

        perplexity = None
        if llama_perplexity_bin and perplexity_dataset:
            perplexity = run_llama_perplexity(llama_perplexity_bin, path, perplexity_dataset, threads)

        results.append(
            FormatResult(
                format=fmt,
                gguf_path=path,
                file_size_mb=os.path.getsize(path) / (1024 * 1024),
                model_size_bytes=model_size,
                prompt_tokens_per_second=prompt_tps,
                gen_tokens_per_second=gen_tps,
                perplexity=perplexity,
            )
        )
    return results
