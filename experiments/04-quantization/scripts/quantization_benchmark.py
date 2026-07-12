"""Experiments 4.1-4.4 — Quantization Benchmark.

Covers all four of this week's numbered experiments in one pass per quantization
level (disk size, memory, loading time, inference performance), since they all come
from the same underlying measurement — reusing Week 2's `run_llama_cli` — rather than
re-running the benchmark four separate times for the same six quantization levels.

- 4.1 Model Size: disk_size_mb column.
- 4.2 Memory Consumption: peak_rss_mb column.
- 4.3 Loading Performance: load_time_s column.
- 4.4 Inference Performance: ttft_s, decode_tokens_per_s, total_time_s columns.
"""

import csv
import json
import sys
import time
from pathlib import Path

import yaml

WEEK2_SCRIPTS = Path(__file__).resolve().parents[2] / "02-llama-cpp" / "scripts"
sys.path.insert(0, str(WEEK2_SCRIPTS))
from llama_cpp_runner import run_llama_cli  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "04-quantization"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    llama_cli_bin = str(REPO_ROOT / config["llama_cli_bin"])
    prompt = config["prompt"]
    threads = config["threads"]
    max_new_tokens = config["experiment_4"]["max_new_tokens"]
    reps = config["experiment_4"]["repetitions"]

    rows = []
    for level in config["quant_levels"]:
        gguf_path = REPO_ROOT / level["file"]
        if not gguf_path.exists():
            print(f"skip {level['name']}: {gguf_path} not found")
            continue
        disk_size_mb = gguf_path.stat().st_size / (1024 * 1024)

        for i in range(reps):
            r = run_llama_cli(llama_cli_bin, str(gguf_path), prompt, max_new_tokens, threads)
            row = {
                "quant_level": level["name"],
                "run": i + 1,
                "disk_size_mb": round(disk_size_mb, 1),
                "load_time_s": round(r.load_time_s, 4),
                "peak_rss_mb": round(r.peak_rss_mb, 1),
                "ttft_s": round(r.ttft_s, 4),
                "decode_tokens_per_s": round(r.decode_tokens_per_s, 2),
                "total_time_s": round(r.total_time_s, 4),
                "prompt_tokens": r.prompt_tokens,
                "generated_tokens": r.generated_tokens,
            }
            rows.append(row)
            print(
                f"{level['name']:>8} run {i + 1}/{reps}: disk={disk_size_mb:.0f}MB "
                f"load={row['load_time_s']:.2f}s ttft={row['ttft_s']:.3f}s "
                f"decode_tps={row['decode_tokens_per_s']:.1f} rss={row['peak_rss_mb']:.0f}MB"
            )

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "quantization_benchmark.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "4.1-4.4",
            "title": "Quantization Benchmark (Size, Memory, Loading, Inference)",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "Disk size, peak memory, and load time all shrink roughly "
                "monotonically from F16 down to Q3_K_M, while decode speed increases "
                "as there is less data to move per token (memory-bandwidth-bound "
                "decode benefits from smaller weights), at some cost to quality not "
                "measured until Week 5."
            ),
        },
        "inference": {"threads": threads, "max_new_tokens": max_new_tokens, "quant_levels": [l["name"] for l in config["quant_levels"]]},
        "measurement": {"repetitions": reps, "metrics": ["disk_size_mb", "load_time_s", "peak_rss_mb", "ttft_s", "decode_tokens_per_s"]},
    }
    (RESULTS_RAW_DIR / "quantization_benchmark_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
