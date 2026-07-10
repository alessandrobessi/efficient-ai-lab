"""Experiment 1.3 — Output Length.

Question: with the prompt fixed, how does total latency and generation speed scale
with the number of tokens generated?
"""

from common import (
    CONFIG,
    MODEL_ID,
    RESULTS_RAW_DIR,
    environment_metadata,
    generate_with_timing,
    load_model,
    write_csv,
    write_json,
)


def main() -> None:
    cfg = CONFIG["experiment_1_3"]
    output_lengths = cfg["output_lengths_tokens"]
    reps = cfg["repetitions"]
    prompt = CONFIG["base_prompt"]

    result = load_model(MODEL_ID)
    model, tokenizer = result.model, result.tokenizer
    print(f"Model loaded in {result.load_time_s:.2f}s")

    rows = []
    for max_new_tokens in output_lengths:
        for i in range(reps):
            gen = generate_with_timing(model, tokenizer, prompt, max_new_tokens)
            row = {
                "requested_output_tokens": max_new_tokens,
                "run": i + 1,
                "prompt_tokens": gen.prompt_tokens,
                "generated_tokens": gen.generated_tokens,
                "ttft_s": round(gen.ttft_s, 4),
                "decode_time_s": round(gen.decode_time_s, 4),
                "total_time_s": round(gen.total_time_s, 4),
                "decode_tokens_per_s": round(gen.decode_tokens_per_s, 2),
                "overall_tokens_per_s": round(gen.overall_tokens_per_s, 2),
            }
            rows.append(row)
            print(
                f"output={max_new_tokens:>4} run {i + 1}/{reps}: "
                f"total={row['total_time_s']:.3f}s decode_tps={row['decode_tokens_per_s']:.1f}"
            )

    write_csv(rows, RESULTS_RAW_DIR / "exp_1_3_output_length.csv")

    meta = environment_metadata(
        experiment_id="1.3",
        title="Output Length",
        hypothesis=(
            "Total latency grows approximately linearly with the number of tokens "
            "generated, since each additional token costs one more constant-time "
            "decode step; TTFT (a one-time prefill cost) should stay roughly constant "
            "across output lengths since the prompt is fixed."
        ),
        inference={"fixed_prompt": True},
        measurement={
            "repetitions": reps,
            "warmup_runs": 0,
            "metrics": ["ttft_s", "decode_time_s", "total_time_s", "decode_tokens_per_s"],
            "output_lengths_tokens": output_lengths,
        },
    )
    write_json(meta, RESULTS_RAW_DIR / "exp_1_3_metadata.json")


if __name__ == "__main__":
    main()
