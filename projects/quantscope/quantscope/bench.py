"""Sweeps llama-bench across a set of GGUF files (one per quantization
format) and collects one row per format — the measured half of quantscope,
as opposed to estimate.py's heuristic.
"""

from __future__ import annotations

import os
import random
import re
import statistics
from dataclasses import dataclass, field

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
    # Raw per-round samples behind the mean/stddev above -- one launch of
    # llama-bench per round, each in an independently-randomized format
    # order (see sweep()). Kept off the flat CSV (cli.py pops these before
    # writing rows) but preserved in FormatResult so a caller with the
    # Python API can see the actual distribution, not just its summary.
    prompt_tokens_per_second_samples: list[float] = field(default_factory=list)
    gen_tokens_per_second_samples: list[float] = field(default_factory=list)
    # None unless perplexity args are given to sweep() -- deliberately
    # optional, since perplexity evaluation is much slower than a speed
    # benchmark and not every sweep needs a quality axis.
    perplexity: float | None = None
    # llama-perplexity's own reported +/- uncertainty for `perplexity`.
    perplexity_error: float | None = None
    # Set only when perplexity_baseline_format is also given: perplexity
    # alone is not comparable across models/tokenizers, but a delta/ratio
    # against a same-model, same-tokenizer reference (e.g. F16) is exactly
    # what "how much quality did this quantization cost" means. The
    # baseline's own entry gets ppl_delta=0.0, ppl_ratio=1.0.
    ppl_delta: float | None = None
    ppl_ratio: float | None = None


# Matches llama-bench's model_type convention, e.g. "qwen2 1B Q8_0" or
# "qwen2 1B Q4_K - Medium" -- architecture and size come first, the
# quantization format (and its optional " - Size" qualifier) always last.
# Used to compare "is this the same base model" independently of
# model_n_params, without hardcoding a list of known format tokens (which
# would need updating every time llama.cpp adds one).
_MODEL_TYPE_PREFIX_RE = re.compile(r"^(\S+\s+\S+)")


def _model_type_prefix(model_type: str) -> str:
    match = _MODEL_TYPE_PREFIX_RE.match(model_type.strip())
    return match.group(1) if match else model_type.strip()


def _validate_same_model(rows_by_format: dict[str, BenchRow]) -> None:
    """Cross-checks model_n_params (exact) and a normalized model_type
    prefix (arch+size, soft) reported by llama-bench's own CSV across every
    format in the sweep -- quantization never changes parameter count, so a
    mismatch there means the files are not the same base model. Entries
    where llama-bench didn't report model_n_params (older builds; value 0)
    are skipped rather than treated as a false mismatch.

    The model_type prefix check is a *soft* guardrail, not proof of
    identity: two different models can share both an architecture and a
    parameter count (e.g. a base model and its instruct fine-tune are
    frequently identical on both axes). Confirming these files are truly
    the same weights would need GGUF-level metadata (tokenizer hash, tensor
    names/shapes) -- deliberately not implemented here, see ROADMAP.md's
    "Explicitly deferred" section. This check only catches the more obvious
    case: an outright different architecture or size class slipping into
    one sweep.

    Deliberately does NOT compare the full model_type string: confirmed
    against a real llama-bench run that model_type embeds the quantization
    format itself (e.g. "qwen2 1B F16" vs "qwen2 1B Q8_0" for the exact same
    base model), so comparing it directly would reject every legitimate
    same-model sweep -- only the arch+size prefix survives across formats.
    """
    known = {fmt: row for fmt, row in rows_by_format.items() if row.model_n_params > 0}
    if len(known) < 2:
        return
    reference_fmt, reference_row = next(iter(known.items()))
    reference_prefix = _model_type_prefix(reference_row.model_type)
    for fmt, row in known.items():
        if row.model_n_params != reference_row.model_n_params:
            raise ModelIdentityError(
                f"'{fmt}' reports {row.model_n_params} parameters but '{reference_fmt}' reports "
                f"{reference_row.model_n_params} -- these don't look like the same base model "
                f"quantized differently. Refusing to compare them."
            )
        prefix = _model_type_prefix(row.model_type)
        if reference_prefix and prefix and prefix != reference_prefix:
            raise ModelIdentityError(
                f"'{fmt}' reports model_type {row.model_type!r} (prefix {prefix!r}) but '{reference_fmt}' "
                f"reports {reference_row.model_type!r} (prefix {reference_prefix!r}) -- these don't look "
                f"like the same base model quantized differently. Refusing to compare them. (This is a "
                f"soft check on architecture+size naming, not a GGUF-metadata identity proof -- see "
                f"ROADMAP.md.)"
            )


def sweep(
    llama_bench_bin: str,
    gguf_paths: dict[str, str],
    n_prompt: int = 512,
    n_gen: int = 128,
    threads: int = 4,
    rounds: int = 10,
    rng_seed: int | None = None,
    llama_perplexity_bin: str | None = None,
    perplexity_dataset: str | None = None,
    perplexity_baseline_format: str | None = None,
    compute_hashes: bool = True,
) -> tuple[list[FormatResult], RunManifest]:
    """Runs llama-bench across `rounds` independent repetitions per format,
    visiting formats in an independently-randomized order each round (each
    call uses `-r 1`, one measurement per launch) -- not one long
    sequential pass per format. This matters: benchmarking every format's
    reps back-to-back before moving to the next confounds format identity
    with the machine's thermal/background/caching state at whatever point
    in the sweep that format happened to run, which is exactly what an
    external review of quantscope caught in its earlier sequential design
    (see ROADMAP.md's v0.2.1 section). Interleaving in random order spreads
    that drift evenly across every format instead.

    Speed and perplexity are measured in separate phases: all speed rounds
    for every format complete first, then perplexity (if requested) runs
    once per format. Perplexity is a deterministic forward-pass loss over a
    fixed dataset, not a timing measurement -- repeating it round-over-round
    the way speed is wouldn't average out anything real, it would just cost
    a lot more wall-clock time for a much slower operation.

    gguf_paths maps a format name (e.g. "Q4_K_M") to the path of an
    already-quantized GGUF file in that format — producing missing ones is
    quantize.py's job, kept separate so a sweep over pre-existing files
    never silently quantizes.

    Validates that every file reports the same model_n_params (and a
    normalized model_type prefix) before returning anything (raises
    ModelIdentityError otherwise -- see _validate_same_model).

    Returns (per-format results with cross-round mean/stddev, a RunManifest
    capturing the shared environment, every round's actual visiting order,
    and a sha256 per file for reproducibility).
    """
    if perplexity_baseline_format is not None and perplexity_baseline_format not in gguf_paths:
        raise ValueError(f"perplexity_baseline_format {perplexity_baseline_format!r} is not one of the swept formats")

    rng = random.Random(rng_seed)
    formats = list(gguf_paths.keys())
    round_orders: list[list[str]] = []
    prompt_samples: dict[str, list[float]] = {fmt: [] for fmt in formats}
    gen_samples: dict[str, list[float]] = {fmt: [] for fmt in formats}
    rows_by_format: dict[str, BenchRow] = {}
    env_row: BenchRow | None = None

    for _ in range(rounds):
        order = formats[:]
        rng.shuffle(order)
        round_orders.append(order)
        for fmt in order:
            path = gguf_paths[fmt]
            rows = run_llama_bench(llama_bench_bin, path, n_prompt, n_gen, threads, repetitions=1)
            prompt_row = next((r for r in rows if r.n_prompt > 0 and r.n_gen == 0), None)
            gen_row = next((r for r in rows if r.n_gen > 0 and r.n_prompt == 0), None)
            if prompt_row:
                prompt_samples[fmt].append(prompt_row.avg_tokens_per_second)
            if gen_row:
                gen_samples[fmt].append(gen_row.avg_tokens_per_second)
            representative = prompt_row or gen_row or rows[0]
            rows_by_format[fmt] = representative
            env_row = env_row or representative

    _validate_same_model(rows_by_format)

    results = []
    for fmt, path in gguf_paths.items():
        row = rows_by_format[fmt]
        p_samples = prompt_samples[fmt]
        g_samples = gen_samples[fmt]
        results.append(
            FormatResult(
                format=fmt,
                gguf_path=path,
                file_size_mb=os.path.getsize(path) / (1024 * 1024),
                model_size_bytes=row.model_size_bytes,
                model_n_params=row.model_n_params,
                model_type=row.model_type,
                prompt_tokens_per_second=statistics.mean(p_samples) if p_samples else 0.0,
                prompt_tokens_per_second_stddev=statistics.stdev(p_samples) if len(p_samples) > 1 else 0.0,
                gen_tokens_per_second=statistics.mean(g_samples) if g_samples else 0.0,
                gen_tokens_per_second_stddev=statistics.stdev(g_samples) if len(g_samples) > 1 else 0.0,
                prompt_tokens_per_second_samples=p_samples,
                gen_tokens_per_second_samples=g_samples,
            )
        )

    perplexity_threads = 0
    if llama_perplexity_bin and perplexity_dataset:
        perplexity_threads = threads
        for r in results:
            ppl = run_llama_perplexity(llama_perplexity_bin, r.gguf_path, perplexity_dataset, threads)
            r.perplexity = ppl.value
            r.perplexity_error = ppl.error

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
            filename=os.path.basename(r.gguf_path),
            sha256=sha256_file(r.gguf_path) if compute_hashes else "",
            model_size_bytes=r.model_size_bytes,
            model_n_params=r.model_n_params,
        )
        for r in results
    ]
    manifest = build_manifest(
        env_row,
        entries,
        n_prompt=n_prompt,
        n_gen=n_gen,
        rounds=rounds,
        round_orders=round_orders,
        perplexity_dataset_sha256=sha256_file(perplexity_dataset) if (perplexity_dataset and compute_hashes) else "",
        perplexity_baseline_format=perplexity_baseline_format or "",
        perplexity_threads=perplexity_threads,
        perplexity_n_gpu_layers=0,
    )

    return results, manifest
