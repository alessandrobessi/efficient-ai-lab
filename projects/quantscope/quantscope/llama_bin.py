"""Thin subprocess wrappers around llama.cpp's own `llama-bench`,
`llama-quantize`, and `llama-perplexity` binaries — quantscope never
reimplements GGML kernel benchmarking or quantization itself, the same
"wrap, don't reimplement" principle behind
`experiments/02-llama-cpp/scripts/llama_cpp_runner.py`'s `run_llama_bench`/
`run_llama_bench_single_rep`, which this module evolves.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import dataclass

# Matches llama-perplexity's final summary line, e.g.:
#   Final estimate: PPL = 5.9070 +/- 0.03166
# Captures both the point estimate and its own reported uncertainty -- a
# quantscope v0.2.0 bug discarded the "+/- 0.03166" half, making it
# impossible to tell whether a small perplexity delta between two formats
# was a real difference or within llama-perplexity's own measurement noise.
_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


class LlamaBinError(RuntimeError):
    """Raised when a llama.cpp subprocess exits non-zero or produces output
    quantscope cannot parse. Always carries the subprocess's own stderr so
    the underlying llama.cpp error is never swallowed.
    """


@dataclass
class BenchRow:
    """One row of llama-bench's own CSV output, for one test (a
    prompt-processing or token-generation test). Keeps the full provenance
    llama-bench itself reports (build/CPU/GPU/backend/config), not just the
    speed number — a benchmark result without this is not reproducible by
    someone else, or by the same person six months later.
    """

    n_prompt: int
    n_gen: int
    avg_tokens_per_second: float
    stddev_tokens_per_second: float
    model_size_bytes: int
    model_n_params: int
    model_type: str
    build_commit: str
    cpu_info: str
    gpu_info: str
    backends: str
    n_threads: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    flash_attn: str
    use_mmap: str


def run_llama_bench(
    llama_bench_bin: str,
    gguf_path: str,
    n_prompt: int,
    n_gen: int,
    threads: int,
    repetitions: int = 3,
) -> list[BenchRow]:
    """Runs llama-bench once, requesting both a prompt-processing test
    (n_prompt tokens) and a token-generation test (n_gen tokens), and
    returns both as separate rows — llama-bench reports the average over
    `repetitions` internally (`-r`), we don't re-average here.

    Always passes `-ngl 0`: quantscope profiles CPU inference specifically
    (its own name and purpose), and llama-bench's own default
    (`-ngl -1`, maximal offload) would silently benchmark GPU-accelerated
    inference on any machine with Metal/CUDA/Vulkan support compiled in,
    answering a different question than the one quantscope claims to.
    """
    cmd = [
        llama_bench_bin,
        "-m", gguf_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-r", str(repetitions),
        "-ngl", "0",
        "-o", "csv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LlamaBinError(f"llama-bench failed (exit {proc.returncode}):\n{proc.stderr}")
    return _parse_bench_csv(proc.stdout)


def _parse_bench_csv(csv_text: str) -> list[BenchRow]:
    rows = []
    for record in csv.DictReader(io.StringIO(csv_text)):
        try:
            rows.append(
                BenchRow(
                    n_prompt=int(record["n_prompt"]),
                    n_gen=int(record["n_gen"]),
                    avg_tokens_per_second=float(record["avg_ts"]),
                    stddev_tokens_per_second=float(record.get("stddev_ts", 0.0) or 0.0),
                    model_size_bytes=int(record.get("model_size", 0)),
                    model_n_params=int(record.get("model_n_params", 0)),
                    model_type=record.get("model_type", ""),
                    build_commit=record.get("build_commit", ""),
                    cpu_info=record.get("cpu_info", ""),
                    gpu_info=record.get("gpu_info", ""),
                    backends=record.get("backends", ""),
                    n_threads=int(record.get("n_threads", 0) or 0),
                    n_batch=int(record.get("n_batch", 0) or 0),
                    n_ubatch=int(record.get("n_ubatch", 0) or 0),
                    n_gpu_layers=int(record.get("n_gpu_layers", 0) or 0),
                    flash_attn=record.get("flash_attn", ""),
                    use_mmap=record.get("use_mmap", ""),
                )
            )
        except (KeyError, ValueError) as e:
            raise LlamaBinError(f"unrecognized llama-bench CSV columns: {e}\ncsv:\n{csv_text}") from e
    if not rows:
        raise LlamaBinError(f"llama-bench produced no parseable rows:\n{csv_text}")
    return rows


def run_llama_quantize(
    llama_quantize_bin: str,
    input_gguf: str,
    output_gguf: str,
    format_name: str,
    imatrix_path: str | None = None,
) -> None:
    """Produces output_gguf in format_name from input_gguf. Raises
    LlamaBinError on failure; quantscope never guesses at a quantized
    file's existence, it either has one or it invokes this to make one.

    imatrix_path, if given, is passed through to llama-quantize's own
    `--imatrix` flag — required for IQ* formats to quantize well (see
    quantize.py's applicability check, which enforces this before ever
    calling here).
    """
    cmd = [llama_quantize_bin]
    if imatrix_path:
        cmd += ["--imatrix", imatrix_path]
    cmd += [input_gguf, output_gguf, format_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LlamaBinError(
            f"llama-quantize failed to produce {format_name} (exit {proc.returncode}):\n{proc.stderr}"
        )


def get_quantize_help(llama_quantize_bin: str) -> str:
    """Returns llama-quantize's own --help text (stdout+stderr combined,
    since llama.cpp binaries are inconsistent about which stream they use),
    the ground truth for formats.list_supported_formats.
    """
    proc = subprocess.run([llama_quantize_bin, "--help"], capture_output=True, text=True)
    # --help conventionally exits non-zero in some llama.cpp builds; the
    # text is what matters, not the exit code.
    return proc.stdout + "\n" + proc.stderr


@dataclass
class PerplexityResult:
    """llama-perplexity's `Final estimate: PPL = value +/- error` line, kept
    as two numbers rather than collapsing to just `value` -- a quantscope
    v0.2.0 bug did exactly that, making it impossible to tell whether a
    small ppl_delta between two formats (e.g. +0.0083) was a real quality
    difference or within llama-perplexity's own reported measurement noise.
    """

    value: float
    error: float


def run_llama_perplexity(
    llama_perplexity_bin: str,
    gguf_path: str,
    dataset_path: str,
    threads: int = 4,
) -> PerplexityResult:
    """Runs llama-perplexity over dataset_path and returns the final PPL
    estimate plus its own reported uncertainty — the raw
    quantization-loss-proxy measurement `bench`'s perplexity options add
    alongside speed/size. Callers are responsible for turning `.value` into
    a delta/ratio against a reference format (see bench.py) — llama.cpp's
    own docs are explicit that a bare perplexity number is only meaningful
    relative to a same-model, same-tokenizer reference, not as an absolute
    "quality" score.

    Always passes `-ngl 0`, for the same reason run_llama_bench does:
    without it, a Metal/CUDA/Vulkan-enabled build would silently evaluate
    perplexity with GPU offload. This doesn't change the perplexity *value*
    (a forward pass produces the same loss regardless of which device ran
    it, modulo tiny floating-point reduction-order differences) but keeps
    quantscope's "CPU is forced, always" claim actually true of every
    llama.cpp invocation it makes, not just the speed benchmark.
    """
    cmd = [
        llama_perplexity_bin,
        "-m", gguf_path,
        "-f", dataset_path,
        "-t", str(threads),
        "-ngl", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LlamaBinError(f"llama-perplexity failed (exit {proc.returncode}):\n{proc.stderr}")
    combined = proc.stdout + "\n" + proc.stderr
    match = _PPL_RE.search(combined)
    if not match:
        raise LlamaBinError(f"could not find a 'Final estimate: PPL = ...' line in llama-perplexity output:\n{combined}")
    return PerplexityResult(value=float(match.group(1)), error=float(match.group(2)))


def get_system_info(llama_bench_bin: str) -> str:
    """Runs llama-bench with no model to capture its startup system_info
    log line (CPU feature flags this specific build detected), the ground
    truth cpu_detect.py cross-checks against OS-level detection. llama-bench
    prints system_info and exits non-zero without -m; that's expected.
    """
    proc = subprocess.run([llama_bench_bin], capture_output=True, text=True)
    return proc.stdout + "\n" + proc.stderr
