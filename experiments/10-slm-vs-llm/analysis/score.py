"""Apply evaluation/metrics/scorers.py to the raw outputs from
run_evaluation.py, producing one row per (system, example) with a score in
[0, 1]. Directly adapted from Week 5/6's score.py — same scorer, keyed by
`system` instead of `quant_level`/`model`.
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.scorers import score_example  # noqa: E402

RAW_PATH = REPO_ROOT / "results" / "raw" / "10-slm-vs-llm" / "raw_outputs.jsonl"
DATASET_PATH = REPO_ROOT / "evaluation" / "datasets" / "v1.jsonl"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "10-slm-vs-llm"


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
                    "system": r["system"],
                    "params_b": r["params_b"],
                    "id": r["id"],
                    "category": r["category"],
                    "language": r["language"],
                    "score": round(result["score"], 4),
                    "detail": result["detail"],
                    "valid_json": result.get("valid_json", ""),
                    "output": r["output"],
                    "expected": r["expected"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
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
