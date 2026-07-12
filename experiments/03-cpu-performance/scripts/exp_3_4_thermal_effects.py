"""Experiment 3.4 — Thermal Effects.

Question: does the strong throughput decline-then-plateau pattern Week 2 Experiment
2.3 found (at 10 threads, a suboptimal configuration) also happen at this machine's
throughput-optimal thread count (2, per Week 2 Exp 2.2)? If the decline persists even
at the "good" configuration, that's much stronger evidence for a genuine thermal
effect rather than an artifact of thread oversubscription.

Follows up on Q7/Q8 (Week 2): is the Exp 2.3 decline actually thermal, and does it
depend on thread count?

`pmset -g therm` is sampled alongside throughput as a best-effort thermal signal —
Apple Silicon doesn't expose live temperature/throttle percentage without root
(`powermetrics` needs sudo), so this is opportunistic, not the primary measurement.
"""

import csv
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


def sample_therm() -> str:
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as e:  # best-effort only
        return f"<pmset failed: {e}>"


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_bench_bin = str(REPO_ROOT / config["llama_bench_bin"])
    cfg = config["experiment_3_4"]
    threads = cfg["threads"]
    reps = cfg["repetitions"]

    t_start = time.perf_counter()
    therm_log = [f"run 0 (start, t=0.0s): {sample_therm()}"]

    rows = []
    for i in range(reps):
        bench_rows = run_llama_bench_single_rep(llama_bench_bin, gguf_path, cfg["n_prompt"], cfg["n_gen"], threads)
        elapsed = time.perf_counter() - t_start
        for r in bench_rows:
            test_type = "pp" if int(r["n_prompt"]) > 0 else "tg"
            row = {
                "run": i + 1,
                "elapsed_s": round(elapsed, 1),
                "test_type": test_type,
                "tokens_per_s": round(float(r["avg_ts"]), 3),
            }
            rows.append(row)
        if (i + 1) % 10 == 0:
            therm_log.append(f"run {i + 1} (t={elapsed:.1f}s): {sample_therm()}")
        print(
            f"run {i + 1}/{reps} (t={elapsed:.0f}s): "
            + ", ".join(f"{r['test_type']}={r['tokens_per_s']:.1f} tok/s" for r in rows[-2:])
        )

    therm_log.append(f"run {reps} (end, t={time.perf_counter() - t_start:.1f}s): {sample_therm()}")

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "exp_3_4_thermal_effects.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    (RESULTS_RAW_DIR / "exp_3_4_therm_log.txt").write_text("\n".join(therm_log) + "\n")

    meta = {
        "experiment": {
            "id": "3.4",
            "title": "Thermal Effects",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "If Week 2 Exp 2.3's decline was mainly a thread-oversubscription "
                "artifact (10 threads on a 4P+6E core machine), throughput at the "
                "optimal 2 threads should stay much more stable across the same "
                "number of independent runs. If it still declines similarly, thermal "
                "throttling under sustained load is the better explanation."
            ),
        },
        "inference": {"threads": threads, "n_prompt": cfg["n_prompt"], "n_gen": cfg["n_gen"], "warmup": False},
        "measurement": {"repetitions": reps, "metrics": ["tokens_per_s per independent run", "pmset -g therm (best-effort)"]},
    }
    (RESULTS_RAW_DIR / "exp_3_4_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
