"""Week 10 speed/memory benchmark for the 2 local systems, directly adapted
from Week 4/6's quantization/model benchmark scripts (same methodology, same
continuity prompt, same `run_llama_cli` helper) — sweeping over *systems*
(different models entirely) rather than quantization levels or same-family
model sizes.
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
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "10-slm-vs-llm"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    llama_cli_bin = str(REPO_ROOT / config["llama_cli_bin"])
    prompt = (
        "Explain, in a few sentences, why running a language model on a CPU has "
        "different performance characteristics from running it on a GPU."
    )
    threads = config["threads"]
    max_new_tokens = config["benchmark"]["max_new_tokens"]
    reps = config["benchmark"]["repetitions"]

    rows = []
    for system in config["local_systems"]:
        gguf_path = REPO_ROOT / system["file"]
        if not gguf_path.exists():
            print(f"skip {system['name']}: {gguf_path} not found")
            continue
        disk_size_mb = gguf_path.stat().st_size / (1024 * 1024)
        for shard in system.get("extra_shards", []):
            disk_size_mb += (REPO_ROOT / shard).stat().st_size / (1024 * 1024)

        for i in range(reps):
            r = run_llama_cli(llama_cli_bin, str(gguf_path), prompt, max_new_tokens, threads)
            row = {
                "system": system["name"],
                "params_b": system["params_b"],
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
                f"{system['name']:>14} run {i + 1}/{reps}: disk={disk_size_mb:.0f}MB "
                f"load={row['load_time_s']:.2f}s ttft={row['ttft_s']:.3f}s "
                f"decode_tps={row['decode_tokens_per_s']:.1f} rss={row['peak_rss_mb']:.0f}MB"
            )

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "system_benchmark.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "10.2",
            "title": "SLM vs LLM — Speed/Memory Benchmark",
            "date": time.strftime("%Y-%m-%d"),
        },
        "inference": {"threads": threads, "max_new_tokens": max_new_tokens, "systems": [s["name"] for s in config["local_systems"]]},
        "measurement": {"repetitions": reps, "metrics": ["disk_size_mb", "load_time_s", "peak_rss_mb", "ttft_s", "decode_tokens_per_s"]},
    }
    (RESULTS_RAW_DIR / "system_benchmark_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
