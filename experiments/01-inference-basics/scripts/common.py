"""Shared inference utilities for Week 1 experiments (transformer inference fundamentals).

Provides model loading with timing/memory measurement, a manual token-by-token
generation loop that separates prefill (time to first token) from decode (steady-state
token generation), and environment metadata capture per the schema in
FULL-ROADMAP.md §14.

All experiments run on CPU only, in line with the program's CPU-first constraint —
Week 1 is about understanding raw inference, not about thread/hardware tuning (Week 3).
"""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw" / "01-inference-basics"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as _f:
    CONFIG: dict = yaml.safe_load(_f)

MODEL_ID: str = CONFIG["model_id"]
DEVICE = torch.device(CONFIG["device"])
DTYPE = getattr(torch, CONFIG["dtype"])
SEED: int = CONFIG["seed"]

# Week 1 is about understanding inference, not about thread scaling (that's Week 3) —
# but PyTorch's default intra-op thread count is conservative (often well below the
# physical core count), so we pin it to the physical core count for a sane baseline
# and record the actual value used in each experiment's metadata.
torch.set_num_threads(psutil.cpu_count(logical=False) or 1)


def process_memory_mb() -> float:
    """Resident set size of the current process, in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            return platform.processor() or "unknown"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def environment_metadata(
    experiment_id: str,
    title: str,
    hypothesis: str,
    model_id: str = MODEL_ID,
    inference: dict | None = None,
    measurement: dict | None = None,
) -> dict:
    """Build the experiment metadata record per FULL-ROADMAP.md §14."""
    return {
        "experiment": {
            "id": experiment_id,
            "title": title,
            "date": time.strftime("%Y-%m-%d"),
            "hypothesis": hypothesis,
        },
        "hardware": {
            "machine": platform.node(),
            "cpu": _cpu_brand(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        },
        "software": {
            "os": f"{platform.system()} {platform.release()}",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model": {
            "name": model_id,
            "parameters": "1.5B",
            "format": "safetensors (Hugging Face Transformers)",
            "quantization": "none (fp32)",
        },
        "inference": {
            "threads": torch.get_num_threads(),
            "context_size": None,
            "batch_size": 1,
            "temperature": 0.0,
            "seed": SEED,
            **(inference or {}),
        },
        "measurement": measurement or {},
    }


@dataclass
class LoadResult:
    model: object
    tokenizer: object
    load_time_s: float
    mem_before_mb: float
    mem_after_mb: float


def load_model(model_id: str = MODEL_ID) -> LoadResult:
    """Load model + tokenizer on CPU, timing the load and measuring RSS before/after."""
    torch.manual_seed(SEED)
    mem_before = process_memory_mb()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE)
    model.to(DEVICE)
    model.eval()
    load_time = time.perf_counter() - t0
    mem_after = process_memory_mb()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadResult(model, tokenizer, load_time, mem_before, mem_after)


@dataclass
class GenerationResult:
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    decode_time_s: float
    total_time_s: float
    decode_tokens_per_s: float
    overall_tokens_per_s: float
    generated_text: str = field(repr=False)


@torch.no_grad()
def generate_with_timing(model, tokenizer, prompt: str, max_new_tokens: int) -> GenerationResult:
    """Greedy, token-by-token generation with an explicit prefill/decode timing split.

    Prefill = the single forward pass over the full prompt that produces the first
    generated token. Decode = each subsequent forward pass, run one token at a time
    using the KV cache from the previous step. This mirrors what a production inference
    engine (e.g. llama.cpp, Week 2) actually does, and is why prefill and decode have
    different performance characteristics: prefill is one large, parallelizable matmul
    over the whole prompt, while decode is a sequence of small, latency-bound steps.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_tokens = input_ids.shape[1]

    t_start = time.perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    t_first_token = time.perf_counter()

    generated = [next_token]
    past_key_values = outputs.past_key_values
    cur_mask = attention_mask

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_token)], dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=cur_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    t_end = time.perf_counter()

    generated_tokens = len(generated)
    ttft = t_first_token - t_start
    decode_time = t_end - t_first_token
    total_time = t_end - t_start
    decode_tps = (generated_tokens - 1) / decode_time if generated_tokens > 1 and decode_time > 0 else 0.0
    overall_tps = generated_tokens / total_time if total_time > 0 else 0.0

    token_ids = torch.cat(generated, dim=1)
    text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    return GenerationResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_s=ttft,
        decode_time_s=decode_time,
        total_time_s=total_time,
        decode_tokens_per_s=decode_tps,
        overall_tokens_per_s=overall_tps,
        generated_text=text,
    )


def make_prompt_of_length(tokenizer, target_tokens: int, filler_text: str) -> str:
    """Build a prompt that tokenizes to approximately `target_tokens` tokens.

    Repeats `filler_text` and truncates at the token level so the reported prompt
    length in results matches what the model actually saw.
    """
    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(tokenizer(filler_text, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer("\n", add_special_tokens=False)["input_ids"][0])
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def write_json(data: dict, out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote metadata -> {out_path}")
