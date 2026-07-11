"""Experiment 2.2 — Thread Count.

Question: how does llama.cpp's prompt-processing (prefill) and text-generation
(decode) throughput scale with thread count on this machine (Apple M4, 10 cores)?

Uses llama-bench, which natively sweeps thread counts and repetitions and reports
avg/stddev tokens/s per configuration — no custom timing needed.
"""

import time
from pathlib import Path

import yaml
from llama_cpp_runner import run_llama_bench

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "02-llama-cpp"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_bench_bin = str(REPO_ROOT / config["llama_bench_bin"])
    cfg = config["experiment_2_2"]

    out_csv = RESULTS_RAW_DIR / "exp_2_2_thread_scaling.csv"
    run_llama_bench(
        llama_bench_bin,
        gguf_path,
        n_prompt=cfg["n_prompt"],
        n_gen=cfg["n_gen"],
        threads=cfg["thread_counts"],
        repetitions=cfg["repetitions"],
        out_csv=out_csv,
    )

    meta = {
        "experiment": {
            "id": "2.2",
            "title": "Thread Count",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "Both prefill (pp) and decode (tg) throughput increase with thread "
                "count up to the physical core count (10), with diminishing returns "
                "as threading overhead and memory bandwidth contention start to "
                "dominate."
            ),
        },
        "inference": {"n_prompt": cfg["n_prompt"], "n_gen": cfg["n_gen"], "thread_counts": cfg["thread_counts"]},
        "measurement": {"repetitions": cfg["repetitions"], "metrics": ["avg_ts (tokens/s)", "stddev_ts"]},
    }
    import json

    (RESULTS_RAW_DIR / "exp_2_2_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
