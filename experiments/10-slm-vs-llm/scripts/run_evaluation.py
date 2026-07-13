"""Week 10 quality evaluation: run all 100 dataset v1 examples through each of
the 2 locally-run systems (CPU-SLM, Larger-Local) via llama-server, recording
raw output plus timing.

Directly adapted from Week 5/6's run_evaluation.py — the shared
evaluation/runners, evaluation/prompts, and evaluation/metrics code is fully
model-agnostic, so this script only needed to change which axis (systems, not
quant levels or models) it sweeps over. The frontier remote API system is
deliberately not included here — see hypothesis.md.
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
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "10-slm-vs-llm"


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

    for system in config["local_systems"]:
        gguf_path = REPO_ROOT / system["file"]
        if not gguf_path.exists():
            print(f"skip {system['name']}: {gguf_path} not found")
            continue

        print(f"\n=== {system['name']} ({system['description']}) ===")
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
                        "system": system["name"],
                        "params_b": system["params_b"],
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
            print(f"  FAILED to start server for {system['name']}: {e}")
            continue

        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} rows -> {out_path}")

    meta = {
        "experiment": {
            "id": "10.1",
            "title": "SLM vs LLM — Local Systems Evaluation Pipeline",
            "date": time.strftime("%Y-%m-%d"),
        },
        "dataset": {"path": config["dataset"], "n_examples": len(dataset)},
        "generation": gen,
        "inference": {"threads": threads, "ctx_size": ctx_size, "systems": [s["name"] for s in config["local_systems"]]},
    }
    (RESULTS_RAW_DIR / "raw_outputs_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
