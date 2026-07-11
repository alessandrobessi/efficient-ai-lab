"""Helpers for invoking llama-cli as a subprocess and parsing its timing output.

llama-cli reports prefill ("prompt eval time") and decode ("eval time") directly
via its own instrumentation (`--perf`), which we trust over hand-rolled timing.
Model load time is *not* explicitly reported by this llama.cpp version in a single,
version-stable log line, so we derive it as:

    load_time_s = wall_clock_time_for_whole_process - reported_total_inference_time

This is coarser than Week 1's direct Python measurement — it also includes process
startup, tokenizer init, and shutdown — but it's robust to internal log format
changes and documented as such (see the experiment README's limitations section).
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
)
EVAL_RE = re.compile(
    r"(?<!prompt )\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
)
TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
PEAK_RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


@dataclass
class LlamaCliResult:
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    decode_time_s: float
    total_time_s: float
    load_time_s: float
    decode_tokens_per_s: float
    peak_rss_mb: float
    wall_time_s: float


def run_llama_cli(
    llama_cli_bin: str,
    gguf_path: str,
    prompt: str,
    max_new_tokens: int,
    threads: int,
) -> LlamaCliResult:
    cmd = [
        "/usr/bin/time",
        "-l",
        llama_cli_bin,
        "-m",
        gguf_path,
        "-p",
        prompt,
        "-n",
        str(max_new_tokens),
        "-t",
        str(threads),
        "--no-conversation",
        "--single-turn",
        "--perf",
        "--no-warmup",
        "-v",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.perf_counter() - t0

    log = proc.stderr

    prompt_match = PROMPT_EVAL_RE.search(log)
    eval_match = EVAL_RE.search(log)
    total_match = TOTAL_RE.search(log)
    rss_match = PEAK_RSS_RE.search(log)

    if not (prompt_match and eval_match and total_match and rss_match):
        raise RuntimeError(
            "Failed to parse llama-cli timing output. stdout:\n"
            f"{proc.stdout}\n--- stderr ---\n{log}"
        )

    prompt_ms, prompt_tokens, _prompt_tps = prompt_match.groups()
    eval_ms, gen_tokens, decode_tps = eval_match.groups()
    total_ms, _total_tokens = total_match.groups()
    peak_rss_bytes = int(rss_match.group(1))

    total_time_s = float(total_ms) / 1000
    return LlamaCliResult(
        prompt_tokens=int(prompt_tokens),
        generated_tokens=int(gen_tokens),
        ttft_s=float(prompt_ms) / 1000,
        decode_time_s=float(eval_ms) / 1000,
        total_time_s=total_time_s,
        load_time_s=max(wall_time - total_time_s, 0.0),
        decode_tokens_per_s=float(decode_tps),
        peak_rss_mb=peak_rss_bytes / (1024 * 1024),
        wall_time_s=wall_time,
    )


def run_llama_bench(
    llama_bench_bin: str,
    gguf_path: str,
    n_prompt: int,
    n_gen: int,
    threads: list[int],
    repetitions: int,
    out_csv: Path,
    no_warmup: bool = False,
) -> None:
    cmd = [
        llama_bench_bin,
        "-m",
        gguf_path,
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-t",
        ",".join(str(t) for t in threads),
        "-r",
        str(repetitions),
        "-o",
        "csv",
    ]
    if no_warmup:
        cmd.append("--no-warmup")

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{proc.stderr}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(proc.stdout)
    print(f"Wrote llama-bench CSV -> {out_csv}")


def run_llama_bench_single_rep(
    llama_bench_bin: str,
    gguf_path: str,
    n_prompt: int,
    n_gen: int,
    threads: int,
) -> list[dict]:
    """Run llama-bench for exactly one repetition (no internal averaging), no warmup.

    Returns the parsed CSV rows (one for the pp test, one for the tg test) as dicts,
    so callers can invoke this repeatedly and keep every individual measurement —
    llama-bench's own `-r N` only ever reports the aggregate across N reps.
    """
    cmd = [
        llama_bench_bin,
        "-m",
        gguf_path,
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-t",
        str(threads),
        "-r",
        "1",
        "-o",
        "csv",
        "--no-warmup",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{proc.stderr}")
    return list(csv.DictReader(io.StringIO(proc.stdout)))
