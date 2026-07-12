"""Experiment 3.2 — Context Scaling.

Question: how do Time to First Token, peak memory, and decode speed scale with
prompt (context) length on llama.cpp, run at this machine's throughput-optimal thread
count (2, per Week 2 Exp 2.2) — and does the pattern change at longer contexts than
Week 1 tested (up to 2048 tokens; this pushes to 4096)?

Follows up on Q4 (Week 1): does the roughly-linear latency-vs-length relationship
hold at longer prompt lengths, or does something change?
"""

import csv
import json
import sys
import time
from pathlib import Path

import yaml

WEEK1_SCRIPTS = Path(__file__).resolve().parents[2] / "01-inference-basics" / "scripts"
WEEK2_SCRIPTS = Path(__file__).resolve().parents[2] / "02-llama-cpp" / "scripts"
sys.path.insert(0, str(WEEK1_SCRIPTS))
sys.path.insert(0, str(WEEK2_SCRIPTS))
from common import make_prompt_of_length  # noqa: E402
from llama_cpp_runner import run_llama_cli  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "03-cpu-performance"

# Same filler text Week 1 used to build prompts of a controlled token length.
FILLER = (
    "Efficient AI systems research investigates how small language models behave under "
    "strict compute constraints. It considers tokenization, embeddings, transformer "
    "blocks, self-attention, feed-forward layers, key-value caching, and the difference "
    "between prefill and decode. It also considers CPU architecture, memory bandwidth, "
    "quantization, and the tradeoffs between model quality and inference performance."
)


def main() -> None:
    with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
        config = yaml.safe_load(f)

    gguf_path = str(REPO_ROOT / config["gguf_path"])
    llama_cli_bin = str(REPO_ROOT / config["llama_cli_bin"])
    cfg = config["experiment_3_2"]
    threads = cfg["threads"]
    max_new_tokens = cfg["max_new_tokens"]
    reps = cfg["repetitions"]

    # Use the same tokenizer as Week 1 purely to build prompts of a controlled *target*
    # token count via text truncation - llama.cpp's own tokenizer then re-tokenizes the
    # resulting text, so actual_prompt_tokens (recorded per run) may differ slightly,
    # exactly as in Week 1 Experiment 1.2.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    rows = []
    for target_len in cfg["prompt_lengths_tokens"]:
        prompt = make_prompt_of_length(tokenizer, target_len, FILLER)
        for i in range(reps):
            r = run_llama_cli(llama_cli_bin, gguf_path, prompt, max_new_tokens, threads)
            row = {
                "target_prompt_tokens": target_len,
                "actual_prompt_tokens": r.prompt_tokens,
                "run": i + 1,
                "load_time_s": round(r.load_time_s, 4),
                "ttft_s": round(r.ttft_s, 4),
                "decode_time_s": round(r.decode_time_s, 4),
                "total_time_s": round(r.total_time_s, 4),
                "decode_tokens_per_s": round(r.decode_tokens_per_s, 2),
                "peak_rss_mb": round(r.peak_rss_mb, 1),
            }
            rows.append(row)
            print(
                f"prompt~{target_len:>5} tok (actual {r.prompt_tokens:>5}) run {i + 1}/{reps}: "
                f"ttft={row['ttft_s']:.3f}s decode_tps={row['decode_tokens_per_s']:.1f} "
                f"rss={row['peak_rss_mb']:.0f}MB"
            )

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_RAW_DIR / "exp_3_2_context_scaling.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

    meta = {
        "experiment": {
            "id": "3.2",
            "title": "Context Scaling",
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": (
                "TTFT grows with prompt length (as in Week 1 Exp 1.2), and peak "
                "memory grows with context length due to KV cache growth; decode "
                "speed should stay roughly flat if it holds up to 4096 tokens, twice "
                "as far as Week 1 tested."
            ),
        },
        "inference": {"threads": threads, "max_new_tokens": max_new_tokens, "prompt_lengths_tokens": cfg["prompt_lengths_tokens"]},
        "measurement": {"repetitions": reps, "metrics": ["ttft_s", "decode_tokens_per_s", "peak_rss_mb"]},
    }
    (RESULTS_RAW_DIR / "exp_3_2_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
