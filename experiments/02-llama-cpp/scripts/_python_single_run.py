"""Single Python-side generation run, invoked as its own subprocess.

Reuses Week 1's loading/generation code (experiments/01-inference-basics). Runs as a
fresh process per invocation — wrapped externally in `/usr/bin/time -l` by
exp_2_1_python_vs_llamacpp.py — so its peak RSS is measured by the OS the same way
llama-cli's is, rather than mixing measurement methodologies between engines.
"""

import json
import sys
from pathlib import Path

WEEK1_SCRIPTS = Path(__file__).resolve().parents[2] / "01-inference-basics" / "scripts"
sys.path.insert(0, str(WEEK1_SCRIPTS))

from common import MODEL_ID, generate_with_timing, load_model  # noqa: E402


def main() -> None:
    prompt = sys.argv[1]
    max_new_tokens = int(sys.argv[2])

    result = load_model(MODEL_ID)

    # llama-cli applies the model's chat template by default (Qwen2.5-Instruct ships
    # one in the GGUF). Apply the same template here so both engines process an
    # equivalent prompt — otherwise the token counts (and therefore TTFT) wouldn't be
    # comparable at all.
    chat_prompt = result.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    gen = generate_with_timing(result.model, result.tokenizer, chat_prompt, max_new_tokens)

    print(
        json.dumps(
            {
                "load_time_s": result.load_time_s,
                "prompt_tokens": gen.prompt_tokens,
                "generated_tokens": gen.generated_tokens,
                "ttft_s": gen.ttft_s,
                "decode_time_s": gen.decode_time_s,
                "total_time_s": result.load_time_s + gen.total_time_s,
                "decode_tokens_per_s": gen.decode_tokens_per_s,
            }
        )
    )


if __name__ == "__main__":
    main()
