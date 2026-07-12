"""Week 6 speed/memory benchmark: disk size, peak RSS, load time, TTFT, and decode
speed for each of the 5 compared models, all at Q4_K_M.

Directly adapted from Week 4's quantization_benchmark.py (same methodology,
same continuity prompt, same `run_llama_cli` helper) but sweeping over *models*
instead of *quantization levels* of a single model — this week's controlled
variable is quantization level (held at Q4_K_M throughout), not the model.
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
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "06-model-comparison"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    llama_cli_bin = str(REPO_ROOT / config["llama_cli_bin"])
    prompt = config["prompt"]
    threads = config["threads"]
    max_new_tokens = config["benchmark"]["max_new_tokens"]
    reps = config["benchmark"]["repetitions"]

    rows = []
    for model in config["models"]:
        gguf_path = REPO_ROOT / model["file"]
        if not gguf_path.exists():
            print(f"skip {model['name']}: {gguf_path} not found")
            continue
        disk_size_mb = gguf_path.stat().st_size / (1024 * 1024)

        for i in range(reps):
            r = run_llama_cli(llama_cli_bin, str(gguf_path), prompt, max_new_tokens, threads)
            row = {
                "model": model["name"],
                "family": model["family"],
                "params_b": model["params_b"],
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
                f"{model['name']:>24} run {i + 1}/{reps}: disk={disk_size_mb:.0f}MB "
                f"load={row['load_time_s']:.2f}s ttft={row['ttft_s']:.3f}s "
                f"decode_tps={row['decode_tokens_per_s']:.1f} rss={row['peak_rss_mb']:.0f}MB"
            )

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "model_benchmark.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "6.1",
            "title": "Model Comparison Benchmark (Size, Memory, Loading, Inference)",
            "date": time.strftime("%Y-%m-%d"),
        },
        "inference": {"threads": threads, "max_new_tokens": max_new_tokens, "models": [m["name"] for m in config["models"]]},
        "measurement": {"repetitions": reps, "metrics": ["disk_size_mb", "load_time_s", "peak_rss_mb", "ttft_s", "decode_tokens_per_s"]},
    }
    (RESULTS_RAW_DIR / "model_benchmark_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
