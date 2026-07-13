"""Thin subprocess wrappers around llama.cpp's own `llama-bench` and
`llama-quantize` binaries — quantscope never reimplements GGML kernel
benchmarking or quantization itself, the same "wrap, don't reimplement"
principle behind `experiments/02-llama-cpp/scripts/llama_cpp_runner.py`'s
`run_llama_bench`/`run_llama_bench_single_rep`, which this module evolves.
"""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass


class LlamaBinError(RuntimeError):
    """Raised when a llama.cpp subprocess exits non-zero or produces output
    quantscope cannot parse. Always carries the subprocess's own stderr so
    the underlying llama.cpp error is never swallowed.
    """


@dataclass
class BenchRow:
    """One row of llama-bench's own CSV output, for one test (a prompt-processing
    or token-generation test), narrowed to what quantscope actually uses.
    """

    n_prompt: int
    n_gen: int
    avg_tokens_per_second: float
    model_size_bytes: int
    model_n_params: int


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
    """
    cmd = [
        llama_bench_bin,
        "-m", gguf_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-r", str(repetitions),
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
                    model_size_bytes=int(record.get("model_size", 0)),
                    model_n_params=int(record.get("model_n_params", 0)),
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
) -> None:
    """Produces output_gguf in format_name from input_gguf. Raises
    LlamaBinError on failure; quantscope never guesses at a quantized
    file's existence, it either has one or it invokes this to make one.
    """
    cmd = [llama_quantize_bin, input_gguf, output_gguf, format_name]
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


def get_system_info(llama_bench_bin: str) -> str:
    """Runs llama-bench with no model to capture its startup system_info
    log line (CPU feature flags this specific build detected), the ground
    truth cpu_detect.py cross-checks against OS-level detection. llama-bench
    prints system_info and exits non-zero without -m; that's expected.
    """
    proc = subprocess.run([llama_bench_bin], capture_output=True, text=True)
    return proc.stdout + "\n" + proc.stderr
