"""Apply evaluation/metrics/scorers.py to the raw model outputs from
run_evaluation.py, producing one row per (quant_level, example) with a score
in [0, 1] and a failure-category detail string.

Kept separate from run_evaluation.py so raw generations never need to be
regenerated just because scoring logic changes.
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.scorers import score_example  # noqa: E402

RAW_PATH = REPO_ROOT / "results" / "raw" / "05-quantization-quality" / "raw_outputs.jsonl"
DATASET_PATH = REPO_ROOT / "evaluation" / "datasets" / "v1.jsonl"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "05-quantization-quality"


def main() -> None:
    metadata_by_id = {}
    with DATASET_PATH.open() as f:
        for line in f:
            obj = json.loads(line)
            metadata_by_id[obj["id"]] = obj["metadata"]

    rows = []
    with RAW_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            metadata = metadata_by_id[r["id"]]
            result = score_example(r["category"], r["output"], r["expected"], metadata)
            rows.append(
                {
                    "quant_level": r["quant_level"],
                    "id": r["id"],
                    "category": r["category"],
                    "language": r["language"],
                    "score": round(result["score"], 4),
                    "detail": result["detail"],
                    "valid_json": result.get("valid_json", ""),
                    "output": r["output"],
                    "expected": r["expected"],
                    "ttft_s": r["ttft_s"],
                    "total_time_s": r["total_time_s"],
                    "decode_tokens_per_s": r["decode_tokens_per_s"],
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "scored_outputs.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
