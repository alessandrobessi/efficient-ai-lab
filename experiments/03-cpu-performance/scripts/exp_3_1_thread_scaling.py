"""Experiment 3.1 — Thread Scaling.

Question: how does llama.cpp throughput scale with thread count at finer granularity
than Week 2's Experiment 2.2 (which only tested 1/2/4/8/10), and does the collapse
point line up with this machine's performance/efficiency core split?

Follows up on open questions from docs/methodology/open-questions.md:
- Q3 (Week 1): at what thread count does decode speed stop improving?
- Q5 (Week 2): is the Exp 2.2 thread-scaling collapse P-core/E-core scheduling?
"""

import json
import sys
import time
from pathlib import Path

import yaml

WEEK2_SCRIPTS = Path(__file__).resolve().parents[2] / "02-llama-cpp" / "scripts"
sys.path.insert(0, str(WEEK2_SCRIPTS))
from llama_cpp_runner import run_llama_bench  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "03-cpu-performance"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_bench_bin = str(REPO_ROOT / config["llama_bench_bin"])
    cfg = config["experiment_3_1"]

    out_csv = RESULTS_RAW_DIR / "exp_3_1_thread_scaling.csv"
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
            "id": "3.1",
            "title": "Thread Scaling",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "Throughput increases with thread count up to roughly the "
                "performance-core count (4 on this machine), then flattens or "
                "declines as efficiency cores and scheduling overhead dominate — "
                "following up on Week 2 Exp 2.2's finding that throughput peaked at "
                "2 threads and degraded by 8-10."
            ),
        },
        "hardware": {
            "performance_cores": config["performance_cores"],
            "efficiency_cores": config["efficiency_cores"],
        },
        "inference": {"n_prompt": cfg["n_prompt"], "n_gen": cfg["n_gen"], "thread_counts": cfg["thread_counts"]},
        "measurement": {"repetitions": cfg["repetitions"], "metrics": ["avg_ts (tokens/s)", "stddev_ts"]},
    }
    (RESULTS_RAW_DIR / "exp_3_1_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
