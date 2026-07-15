"""Sweeps llama-bench across a set of GGUF files (one per quantization
format) and collects one row per format — the measured half of quantscope,
as opposed to estimate.py's heuristic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from quantscope.llama_bin import BenchRow, run_llama_bench, run_llama_perplexity
from quantscope.manifest import ModelEntry, RunManifest, build_manifest, sha256_file


class ModelIdentityError(ValueError):
    """Raised when --gguf entries in one sweep don't appear to be the same
    base model quantized differently -- e.g. different parameter counts,
    which quantization never changes. Comparing speed/quality across
    different models under one "which format is best for this model"
    question would be silently meaningless.
    """


@dataclass
class FormatResult:
    format: str
    gguf_path: str
    file_size_mb: float
    model_size_bytes: int
    model_n_params: int
    model_type: str
    prompt_tokens_per_second: float
    prompt_tokens_per_second_stddev: float
    gen_tokens_per_second: float
    gen_tokens_per_second_stddev: float
    # None unless perplexity args are given to sweep() -- deliberately
    # optional, since perplexity evaluation is much slower than a speed
    # benchmark and not every sweep needs a quality axis.
    perplexity: float | None = None
    # Set only when perplexity_baseline_format is also given: perplexity
    # alone is not comparable across models/tokenizers, but a delta/ratio
    # against a same-model, same-tokenizer reference (e.g. F16) is exactly
    # what "how much quality did this quantization cost" means. The
    # baseline's own entry gets ppl_delta=0.0, ppl_ratio=1.0.
    ppl_delta: float | None = None
    ppl_ratio: float | None = None


def _validate_same_model(rows_by_format: dict[str, BenchRow]) -> None:
    """Cross-checks model_n_params (and model_type, if present) reported by
    llama-bench's own CSV across every format in the sweep -- quantization
    never changes parameter count, so a mismatch means the files are not
    the same base model. Entries where llama-bench didn't report
    model_n_params (older builds; value 0) are skipped rather than treated
    as a false mismatch.

    Deliberately does NOT compare model_type: confirmed against a real
    llama-bench run that model_type embeds the quantization format itself
    (e.g. "qwen2 1B F16" vs "qwen2 1B Q8_0" for the exact same base model),
    so comparing it directly would reject every legitimate same-model sweep
    -- model_n_params is the only signal quantization doesn't change.
    """
    known = {fmt: row for fmt, row in rows_by_format.items() if row.model_n_params > 0}
    if len(known) < 2:
        return
    reference_fmt, reference_row = next(iter(known.items()))
    for fmt, row in known.items():
        if row.model_n_params != reference_row.model_n_params:
            raise ModelIdentityError(
                f"'{fmt}' reports {row.model_n_params} parameters but '{reference_fmt}' reports "
                f"{reference_row.model_n_params} -- these don't look like the same base model "
                f"quantized differently. Refusing to compare them."
            )


def sweep(
    llama_bench_bin: str,
    gguf_paths: dict[str, str],
    n_prompt: int = 512,
    n_gen: int = 128,
    threads: int = 4,
    repetitions: int = 3,
    llama_perplexity_bin: str | None = None,
    perplexity_dataset: str | None = None,
    perplexity_baseline_format: str | None = None,
    compute_hashes: bool = True,
) -> tuple[list[FormatResult], RunManifest]:
    """Runs llama-bench once per (format, gguf_path) pair. gguf_paths maps a
    format name (e.g. "Q4_K_M") to the path of an already-quantized GGUF
    file in that format — producing missing ones is quantize.py's job, kept
    separate so a sweep over pre-existing files never silently quantizes.

    Validates that every file reports the same model_n_params/model_type
    before returning anything (raises ModelIdentityError otherwise -- see
    _validate_same_model).

    If both llama_perplexity_bin and perplexity_dataset are given, also runs
    llama-perplexity per format. If perplexity_baseline_format is also given
    (must be one of gguf_paths' keys), every result's perplexity is turned
    into a delta/ratio against that format's own perplexity, since a bare
    perplexity number is only meaningful relative to a same-model reference.

    Returns (per-format results, a RunManifest capturing the shared
    environment + a sha256 per file for reproducibility).
    """
    if perplexity_baseline_format is not None and perplexity_baseline_format not in gguf_paths:
        raise ValueError(f"perplexity_baseline_format {perplexity_baseline_format!r} is not one of the swept formats")

    results = []
    rows_by_format: dict[str, BenchRow] = {}
    env_row: BenchRow | None = None

    for fmt, path in gguf_paths.items():
        rows = run_llama_bench(llama_bench_bin, path, n_prompt, n_gen, threads, repetitions)
        prompt_row = next((r for r in rows if r.n_prompt > 0 and r.n_gen == 0), None)
        gen_row = next((r for r in rows if r.n_gen > 0 and r.n_prompt == 0), None)
        representative = prompt_row or gen_row or rows[0]
        rows_by_format[fmt] = representative
        env_row = env_row or representative

        perplexity = None
        if llama_perplexity_bin and perplexity_dataset:
            perplexity = run_llama_perplexity(llama_perplexity_bin, path, perplexity_dataset, threads)

        results.append(
            FormatResult(
                format=fmt,
                gguf_path=path,
                file_size_mb=os.path.getsize(path) / (1024 * 1024),
                model_size_bytes=representative.model_size_bytes,
                model_n_params=representative.model_n_params,
                model_type=representative.model_type,
                prompt_tokens_per_second=prompt_row.avg_tokens_per_second if prompt_row else 0.0,
                prompt_tokens_per_second_stddev=prompt_row.stddev_tokens_per_second if prompt_row else 0.0,
                gen_tokens_per_second=gen_row.avg_tokens_per_second if gen_row else 0.0,
                gen_tokens_per_second_stddev=gen_row.stddev_tokens_per_second if gen_row else 0.0,
                perplexity=perplexity,
            )
        )

    _validate_same_model(rows_by_format)

    if perplexity_baseline_format is not None:
        baseline = next(r for r in results if r.format == perplexity_baseline_format)
        if baseline.perplexity is None:
            raise ValueError("perplexity_baseline_format was given but perplexity was not computed for it")
        for r in results:
            if r.perplexity is None:
                continue
            r.ppl_delta = r.perplexity - baseline.perplexity
            r.ppl_ratio = r.perplexity / baseline.perplexity

    entries = [
        ModelEntry(
            format=r.format,
            path=r.gguf_path,
            sha256=sha256_file(r.gguf_path) if compute_hashes else "",
            model_size_bytes=r.model_size_bytes,
            model_n_params=r.model_n_params,
        )
        for r in results
    ]
    manifest = build_manifest(env_row, entries)

    return results, manifest
