"""Week 6 quality evaluation: run all 100 dataset v1 examples through each of the
5 compared models (all Q4_K_M) via llama-server, recording raw output plus timing.

Directly adapted from Week 5's run_evaluation.py — the shared
evaluation/runners, evaluation/prompts, and evaluation/metrics code is fully
model-agnostic (it only needs a GGUF path), so this script only needed to change
which axis (models, not quant levels) it sweeps over.
"""

import json
import time
from pathlib import Path

import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.prompts.templates import build_messages  # noqa: E402
from evaluation.runners.llama_server_runner import LlamaServer, ServerStartupError  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "06-model-comparison"


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    server_bin = str(REPO_ROOT / config["llama_server_bin"])
    threads = config["threads"]
    ctx_size = config["ctx_size"]
    port = config["port"]
    gen = config["generation"]

    dataset = load_dataset(REPO_ROOT / config["dataset"])
    print(f"Loaded {len(dataset)} dataset examples")

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_RAW_DIR / "raw_outputs.jsonl"
    rows = []

    for model in config["models"]:
        gguf_path = REPO_ROOT / model["file"]
        if not gguf_path.exists():
            print(f"skip {model['name']}: {gguf_path} not found")
            continue

        print(f"\n=== {model['name']} ===")
        try:
            with LlamaServer(server_bin, str(gguf_path), port, threads, ctx_size) as server:
                for i, example in enumerate(dataset, start=1):
                    messages = build_messages(example)
                    t0 = time.perf_counter()
                    result = server.chat(
                        messages,
                        temperature=gen["temperature"],
                        seed=gen["seed"],
                        max_tokens=gen["max_tokens"],
                    )
                    wall_s = time.perf_counter() - t0
                    row = {
                        "model": model["name"],
                        "family": model["family"],
                        "params_b": model["params_b"],
                        "id": example["id"],
                        "category": example["category"],
                        "language": example["language"],
                        "expected": example["expected"],
                        "output": result.content,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "ttft_s": round(result.ttft_s, 4),
                        "total_time_s": round(result.total_time_s, 4),
                        "wall_time_s": round(wall_s, 4),
                        "decode_tokens_per_s": round(result.decode_tokens_per_s, 2),
                    }
                    rows.append(row)
                    if i % 20 == 0 or i == len(dataset):
                        print(f"  {i}/{len(dataset)} examples done")
        except ServerStartupError as e:
            print(f"  FAILED to start server for {model['name']}: {e}")
            continue

        # Write incrementally so a crash partway through doesn't lose completed models.
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} rows -> {out_path}")

    meta = {
        "experiment": {
            "id": "6.2",
            "title": "Model Comparison — Evaluation Pipeline",
            "date": time.strftime("%Y-%m-%d"),
        },
        "dataset": {"path": config["dataset"], "n_examples": len(dataset)},
        "generation": gen,
        "inference": {"threads": threads, "ctx_size": ctx_size, "models": [m["name"] for m in config["models"]]},
    }
    (RESULTS_RAW_DIR / "raw_outputs_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
