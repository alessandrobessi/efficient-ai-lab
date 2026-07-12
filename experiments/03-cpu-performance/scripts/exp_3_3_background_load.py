"""Experiment 3.3 — Background Load.

Question: how much does llama.cpp inference throughput degrade when the CPU is
under concurrent, unrelated load — at the machine's throughput-optimal thread count
(2, per Week 2 Exp 2.2)?

Background load is generated with plain `yes > /dev/null` processes (no `stress`/
`stress-ng` available on this machine) — each is a single-threaded, CPU-bound busy
loop, so N processes approximate N fully-loaded cores.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

WEEK2_SCRIPTS = Path(__file__).resolve().parents[2] / "02-llama-cpp" / "scripts"
sys.path.insert(0, str(WEEK2_SCRIPTS))
from llama_cpp_runner import run_llama_bench_single_rep  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "03-cpu-performance"


def spawn_hogs(n: int) -> list[subprocess.Popen]:
    return [subprocess.Popen(["yes"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]


def kill_hogs(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_bench_bin = str(REPO_ROOT / config["llama_bench_bin"])
    cfg = config["experiment_3_3"]
    threads = cfg["threads"]
    reps = cfg["repetitions"]

    rows = []
    for n_hogs in cfg["background_hog_counts"]:
        hogs = spawn_hogs(n_hogs)
        try:
            if n_hogs > 0:
                time.sleep(1.0)  # let the hogs ramp up before measuring
            for i in range(reps):
                bench_rows = run_llama_bench_single_rep(llama_bench_bin, gguf_path, cfg["n_prompt"], cfg["n_gen"], threads)
                for r in bench_rows:
                    test_type = "pp" if int(r["n_prompt"]) > 0 else "tg"
                    row = {
                        "background_hogs": n_hogs,
                        "run": i + 1,
                        "test_type": test_type,
                        "tokens_per_s": round(float(r["avg_ts"]), 3),
                    }
                    rows.append(row)
                print(
                    f"hogs={n_hogs} run {i + 1}/{reps}: "
                    + ", ".join(f"{r['test_type']}={r['tokens_per_s']:.1f} tok/s" for r in rows[-2:])
                )
        finally:
            kill_hogs(hogs)

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "exp_3_3_background_load.csv"
    with out_csv.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "3.3",
            "title": "Background Load",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "Inference throughput degrades monotonically as concurrent background "
                "CPU load increases, since background processes compete for the same "
                "physical cores llama.cpp's inference threads need."
            ),
        },
        "inference": {"threads": threads, "n_prompt": cfg["n_prompt"], "n_gen": cfg["n_gen"], "background_hog_counts": cfg["background_hog_counts"]},
        "measurement": {"repetitions": reps, "metrics": ["tokens_per_s per run per hog-count"]},
    }
    (RESULTS_RAW_DIR / "exp_3_3_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
