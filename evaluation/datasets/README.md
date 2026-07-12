# Evaluation Dataset

## v1.jsonl

100 hand-authored examples, one JSON object per line, evenly split across 6 task
categories and 2 languages (8-9 examples per category per language):

| category | en | it |
|---|---|---|
| classification | 8 | 8 |
| information_extraction | 8 | 8 |
| structured_output | 8 | 8 |
| summarization | 8 | 8 |
| reasoning | 9 | 9 |
| instruction_following | 9 | 9 |

Schema (per FULL-ROADMAP.md §8):

```json
{
  "id": "classification-en-001",
  "category": "classification",
  "language": "en",
  "prompt": "...",
  "expected": "...",
  "metadata": {}
}
```

- **id** — `{category}-{language}-{3-digit sequence}`, unique within the file.
- **category** — one of `classification`, `information_extraction`,
  `structured_output`, `summarization`, `reasoning`, `instruction_following`.
- **language** — `en` or `it`.
- **prompt** — the instruction/question given to the model.
- **expected** — the gold answer. For `structured_output` and
  `information_extraction`, this is a JSON string (parse it, then compare structured
  fields — don't string-match). For `summarization`, it's a reference summary meant
  for semantic-similarity scoring, not exact match (see each example's
  `metadata.scoring` note). For everything else, it's close to an exact-match target.
- **metadata** — category-specific hints for an automated scorer: `labels` (the
  closed label set for classification items), `fields`/`schema_keys` (expected JSON
  keys for extraction/structured-output items), `answer_type` (reasoning items),
  `constraint` (the specific format rule for instruction-following items), or
  `scoring` (summarization items).

### Design notes

- **This is v1, scoped to ~100 of the roadmap's target 100-200 examples** (Week 4's
  brief is "begin constructing the evaluation dataset," not necessarily finish it).
  Extending toward 200 — more examples per category, harder/adversarial cases,
  additional categories — is open for Week 5, when the dataset actually gets used to
  score quantization levels' quality.
- **Unlike experiment results, this file is meant to be hand-edited directly** —
  it's an authored corpus, not measured data, so the "raw data is never edited by
  hand" rule (root `README.md` §7) doesn't apply here. Validate after editing:

  ```bash
  uv run python evaluation/datasets/validate.py evaluation/datasets/v1.jsonl
  ```
- **Italian examples are original Italian, not translations** of the English ones —
  different scenarios, names, and numbers throughout, so the two language splits
  aren't just paraphrases of each other.
