"""System prompts used to keep model output terse enough for automated scoring.

The dataset's `prompt` field (evaluation/datasets/v1.jsonl) already carries the
task-specific instruction (including any format constraint, for
instruction_following items) — these system prompts only add a category-level
"don't explain yourself" framing on top, identical across quant levels and
languages, so that any quality difference measured in Week 5 reflects the model's
own compliance/correctness rather than differences in how verbose it decided to be.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "classification": (
        "You are a strict classifier. Respond with only the single label, exactly "
        "as it should appear. No explanation, no punctuation, no extra words."
    ),
    "information_extraction": (
        "You extract structured data. Respond with only a single-line JSON object "
        "containing the requested fields. No explanation, no markdown code fences."
    ),
    "structured_output": (
        "You convert text into structured data. Respond with only a single-line "
        "JSON object with exactly the requested keys. No explanation, no markdown "
        "code fences."
    ),
    "summarization": (
        "You write concise summaries. Respond with only the one-sentence summary. "
        "No preamble, no explanation."
    ),
    "reasoning": (
        "Solve the problem step by step internally, but respond with only the "
        "final answer. No explanation, no working shown."
    ),
    "instruction_following": (
        "Follow the instruction exactly as stated, including any formatting "
        "constraint. Respond with only what is asked for. No explanation."
    ),
}


def build_messages(example: dict) -> list[dict]:
    system = SYSTEM_PROMPTS[example["category"]]
    user_content = example["prompt"]

    # classification prompts ("Classify the sentiment or topic...") don't spell out
    # the closed label set inline — without it, the model has to guess which of
    # several possible label sets applies, and (found during a first evaluation
    # pass) reliably answers with the meta-category name ("Topic", "Sentiment")
    # instead of a label. metadata.labels exists precisely to disambiguate this, so
    # surface it in the prompt, not just to the scorer.
    if example["category"] == "classification":
        labels = example["metadata"].get("labels", [])
        user_content = f"{user_content}\n\nRespond with exactly one of: {', '.join(labels)}."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
