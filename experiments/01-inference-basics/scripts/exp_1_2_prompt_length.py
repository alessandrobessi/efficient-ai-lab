"""Experiment 1.2 — Prompt Length.

Question: how does prompt (prefill) length affect Time to First Token, total latency,
and decode speed, when output length is held fixed?
"""

from common import (
    CONFIG,
    MODEL_ID,
    RESULTS_RAW_DIR,
    environment_metadata,
    generate_with_timing,
    load_model,
    make_prompt_of_length,
    write_csv,
    write_json,
)


def main() -> None:
    cfg = CONFIG["experiment_1_2"]
    prompt_lengths = cfg["prompt_lengths_tokens"]
    max_new_tokens = cfg["max_new_tokens"]
    reps = cfg["repetitions"]
    filler = CONFIG["prompt_length_filler"]

    result = load_model(MODEL_ID)
    model, tokenizer = result.model, result.tokenizer
    print(f"Model loaded in {result.load_time_s:.2f}s")

    rows = []
    for target_len in prompt_lengths:
        prompt = make_prompt_of_length(tokenizer, target_len, filler)
        for i in range(reps):
            gen = generate_with_timing(model, tokenizer, prompt, max_new_tokens)
            row = {
                "target_prompt_tokens": target_len,
                "actual_prompt_tokens": gen.prompt_tokens,
                "run": i + 1,
                "generated_tokens": gen.generated_tokens,
                "ttft_s": round(gen.ttft_s, 4),
                "decode_time_s": round(gen.decode_time_s, 4),
                "total_time_s": round(gen.total_time_s, 4),
                "decode_tokens_per_s": round(gen.decode_tokens_per_s, 2),
                "overall_tokens_per_s": round(gen.overall_tokens_per_s, 2),
            }
            rows.append(row)
            print(
                f"prompt~{target_len:>4} tok (actual {gen.prompt_tokens:>4}) run {i + 1}/{reps}: "
                f"ttft={row['ttft_s']:.3f}s decode_tps={row['decode_tokens_per_s']:.1f}"
            )

    write_csv(rows, RESULTS_RAW_DIR / "exp_1_2_prompt_length.csv")

    meta = environment_metadata(
        experiment_id="1.2",
        title="Prompt Length",
        hypothesis=(
            "Time to First Token grows with prompt length (prefill scales with prompt "
            "size), while steady-state decode speed (tokens/s) stays roughly constant "
            "regardless of prompt length, since decode only depends on the KV cache "
            "size grown during prefill, not on re-processing the prompt."
        ),
        inference={"max_new_tokens": max_new_tokens},
        measurement={
            "repetitions": reps,
            "warmup_runs": 0,
            "metrics": ["ttft_s", "decode_time_s", "total_time_s", "decode_tokens_per_s"],
            "prompt_lengths_tokens": prompt_lengths,
        },
    )
    write_json(meta, RESULTS_RAW_DIR / "exp_1_2_metadata.json")


if __name__ == "__main__":
    main()
