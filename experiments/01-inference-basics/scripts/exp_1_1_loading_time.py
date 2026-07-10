"""Experiment 1.1 — Model Loading Time.

Question: how long does it take to load the model, and how much process memory does
loading consume? Repeated across multiple runs to observe variance (e.g. from OS file
cache warm-up after the first load).
"""

import gc

from common import CONFIG, MODEL_ID, RESULTS_RAW_DIR, environment_metadata, load_model, write_csv, write_json


def main() -> None:
    reps = CONFIG["experiment_1_1"]["repetitions"]
    rows = []

    for i in range(reps):
        result = load_model(MODEL_ID)
        row = {
            "run": i + 1,
            "model_id": MODEL_ID,
            "load_time_s": round(result.load_time_s, 4),
            "mem_before_mb": round(result.mem_before_mb, 2),
            "mem_after_mb": round(result.mem_after_mb, 2),
            "mem_delta_mb": round(result.mem_after_mb - result.mem_before_mb, 2),
        }
        rows.append(row)
        print(
            f"run {i + 1}/{reps}: load_time={row['load_time_s']:.2f}s "
            f"mem_before={row['mem_before_mb']:.0f}MB mem_after={row['mem_after_mb']:.0f}MB "
            f"delta={row['mem_delta_mb']:.0f}MB"
        )
        del result
        gc.collect()

    write_csv(rows, RESULTS_RAW_DIR / "exp_1_1_loading_time.csv")

    meta = environment_metadata(
        experiment_id="1.1",
        title="Model Loading Time",
        hypothesis=(
            "Model loading time and memory footprint are stable across repeated loads "
            "of the same model, after accounting for OS-level file cache warm-up on the "
            "first run."
        ),
        measurement={
            "repetitions": reps,
            "warmup_runs": 0,
            "metrics": ["load_time_s", "mem_before_mb", "mem_after_mb", "mem_delta_mb"],
        },
    )
    write_json(meta, RESULTS_RAW_DIR / "exp_1_1_metadata.json")


if __name__ == "__main__":
    main()
