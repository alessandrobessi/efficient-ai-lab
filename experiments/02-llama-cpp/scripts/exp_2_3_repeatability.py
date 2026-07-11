"""Experiment 2.3 — Repeatability.

Question: how much does llama.cpp benchmark throughput vary from one full benchmark
execution to the next, and is there a visible warm-up effect in the first few runs?

llama-bench's own `-r N` only reports the mean/stddev *across* N internal repetitions
in a single process — it doesn't expose the individual measurements. To see
run-to-run variance and warm-up effects (e.g. cold file cache on the very first
run, mirroring Week 1 Experiment 1.1), this script instead invokes llama-bench as
>=20 independent process executions, each doing exactly one (--no-warmup) pp/tg
measurement, and keeps every individual result.
"""

import csv
import json
import time
from pathlib import Path

import yaml
from llama_cpp_runner import run_llama_bench_single_rep

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "02-llama-cpp"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_bench_bin = str(REPO_ROOT / config["llama_bench_bin"])
    threads = config["threads"]
    cfg = config["experiment_2_3"]
    reps = cfg["repetitions"]

    rows = []
    for i in range(reps):
        bench_rows = run_llama_bench_single_rep(
            llama_bench_bin, gguf_path, cfg["n_prompt"], cfg["n_gen"], threads
        )
        for r in bench_rows:
            test_type = "pp" if int(r["n_prompt"]) > 0 else "tg"
            row = {
                "run": i + 1,
                "test_type": test_type,
                "tokens_per_s": round(float(r["avg_ts"]), 3),
                "avg_ns": int(r["avg_ns"]),
            }
            rows.append(row)
        print(
            f"run {i + 1}/{reps}: "
            + ", ".join(f"{r['test_type']}={r['tokens_per_s']:.1f} tok/s" for r in rows[-2:])
        )

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "exp_2_3_repeatability.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "2.3",
            "title": "Repeatability",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "Run-to-run variance is small (single-digit percent coefficient of "
                "variation) once the OS file cache is warm, and the first "
                "independent invocation is measurably slower than subsequent ones."
            ),
        },
        "inference": {"n_prompt": cfg["n_prompt"], "n_gen": cfg["n_gen"], "threads": threads, "warmup": False},
        "measurement": {"repetitions": reps, "metrics": ["tokens_per_s per independent run"]},
    }
    (RESULTS_RAW_DIR / "exp_2_3_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
