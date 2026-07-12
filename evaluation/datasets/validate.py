"""Schema validator for evaluation dataset JSONL files.

Usage: uv run python evaluation/datasets/validate.py evaluation/datasets/v1.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"id", "category", "language", "prompt", "expected", "metadata"}
VALID_CATEGORIES = {
    "classification",
    "information_extraction",
    "structured_output",
    "summarization",
    "reasoning",
    "instruction_following",
}
VALID_LANGUAGES = {"en", "it"}


def validate(path: Path) -> None:
    seen_ids: set[str] = set()
    counts: Counter = Counter()
    errors: list[str] = []

    with path.open(encoding="utf-8") as f:
        lines = f.readlines()

    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e})")
            continue

        missing = REQUIRED_FIELDS - obj.keys()
        if missing:
            errors.append(f"line {lineno}: missing fields {missing}")
            continue

        if obj["category"] not in VALID_CATEGORIES:
            errors.append(f"line {lineno}: unknown category {obj['category']!r}")
        if obj["language"] not in VALID_LANGUAGES:
            errors.append(f"line {lineno}: unknown language {obj['language']!r}")
        if not str(obj["prompt"]).strip():
            errors.append(f"line {lineno}: empty prompt")
        if not str(obj["expected"]).strip():
            errors.append(f"line {lineno}: empty expected")
        if obj["id"] in seen_ids:
            errors.append(f"line {lineno}: duplicate id {obj['id']!r}")
        seen_ids.add(obj["id"])
        counts[(obj["category"], obj["language"])] += 1

    print(f"{path}: {len(lines)} lines, {len(seen_ids)} unique ids")
    for (category, language), n in sorted(counts.items()):
        print(f"  {category:<24} {language}: {n}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nOK — no schema errors.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    validate(Path(sys.argv[1]))
