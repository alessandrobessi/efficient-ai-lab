"""Experiment 2.1 — Python vs llama.cpp.

Question: under matched conditions (same model, same effective prompt, same thread
count, same output length), how do the Hugging Face Transformers (Python) baseline
from Week 1 and llama.cpp compare on model loading time, peak memory, Time to First
Token, and decode speed?

Both engines run as fresh subprocesses per repetition — each wrapped in
`/usr/bin/time -l` — so peak RSS is measured identically by the OS on both sides.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_cpp_runner import run_llama_cli  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "02-llama-cpp"

PEAK_RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


def run_python_once(prompt: str, max_new_tokens: int) -> dict:
    script = EXPERIMENT_DIR / "scripts" / "_python_single_run.py"
    cmd = ["/usr/bin/time", "-l", "uv", "run", "python", str(script), prompt, str(max_new_tokens)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Python single run failed:\n{proc.stderr}")

    rss_match = PEAK_RSS_RE.search(proc.stderr)
    if not rss_match:
        raise RuntimeError(f"Could not find peak RSS in output:\n{proc.stderr}")

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["peak_rss_mb"] = int(rss_match.group(1)) / (1024 * 1024)
    return result


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_cli_bin = str(REPO_ROOT / config["llama_cli_bin"])
    prompt = config["prompt"]
    threads = config["threads"]
    max_new_tokens = config["experiment_2_1"]["max_new_tokens"]
    reps = config["experiment_2_1"]["repetitions"]

    rows = []

    for i in range(reps):
        r = run_llama_cli(llama_cli_bin, gguf_path, prompt, max_new_tokens, threads)
        row = {
            "engine": "llama.cpp",
            "run": i + 1,
            "prompt_tokens": r.prompt_tokens,
            "generated_tokens": r.generated_tokens,
            "load_time_s": round(r.load_time_s, 4),
            "ttft_s": round(r.ttft_s, 4),
            "decode_time_s": round(r.decode_time_s, 4),
            "total_time_s": round(r.total_time_s, 4),
            "decode_tokens_per_s": round(r.decode_tokens_per_s, 2),
            "peak_rss_mb": round(r.peak_rss_mb, 1),
        }
        rows.append(row)
        print(f"[llama.cpp] run {i + 1}/{reps}: ttft={row['ttft_s']:.3f}s decode_tps={row['decode_tokens_per_s']:.1f} rss={row['peak_rss_mb']:.0f}MB")

    for i in range(reps):
        r = run_python_once(prompt, max_new_tokens)
        row = {
            "engine": "python",
            "run": i + 1,
            "prompt_tokens": r["prompt_tokens"],
            "generated_tokens": r["generated_tokens"],
            "load_time_s": round(r["load_time_s"], 4),
            "ttft_s": round(r["ttft_s"], 4),
            "decode_time_s": round(r["decode_time_s"], 4),
            "total_time_s": round(r["total_time_s"], 4),
            "decode_tokens_per_s": round(r["decode_tokens_per_s"], 2),
            "peak_rss_mb": round(r["peak_rss_mb"], 1),
        }
        rows.append(row)
        print(f"[python]    run {i + 1}/{reps}: ttft={row['ttft_s']:.3f}s decode_tps={row['decode_tokens_per_s']:.1f} rss={row['peak_rss_mb']:.0f}MB")

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "exp_2_1_python_vs_llamacpp.csv"
    with out_csv.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "2.1",
            "title": "Python vs llama.cpp",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "llama.cpp achieves higher decode throughput and lower peak memory "
                "than the Hugging Face Transformers baseline for the same model, "
                "since it's a purpose-built CPU inference engine (quantized kernels, "
                "no Python/autograd overhead) vs a general-purpose training/inference "
                "framework running eagerly in fp32."
            ),
        },
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "gguf": config["gguf_path"], "quantization": "F16 (llama.cpp) vs fp32 (Python)"},
        "inference": {"threads": threads, "max_new_tokens": max_new_tokens, "prompt_tokens": rows[0]["prompt_tokens"]},
        "measurement": {"repetitions": reps, "metrics": ["load_time_s", "ttft_s", "decode_time_s", "peak_rss_mb", "decode_tokens_per_s"]},
    }
    (RESULTS_RAW_DIR / "exp_2_1_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
