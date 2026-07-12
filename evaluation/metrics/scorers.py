"""Per-category scorers for evaluation dataset v1 (evaluation/datasets/v1.jsonl).

Every scorer returns a dict with at least:
    - "score": float in [0, 1] — quality score for this example (most categories
      are effectively 0/1; information_extraction/structured_output give partial
      credit per matched field, and summarization is a continuous lexical-overlap
      score).
    - "detail": short string identifying *why* the score came out as it did, used
      for the failure-category breakdown in analysis/analyze.py.

These are deliberately heuristic, regex/string-based checkers, not an LLM-as-judge
— per FULL-ROADMAP.md's Week 5 brief ("avoid relying entirely on LLM-as-judge
evaluation"). See each experiment README's limitations section for what this
trades away (mainly: no true semantic understanding, only lexical/structural
matching).
"""

from __future__ import annotations

import json
import re

_INT_RE = re.compile(r"-?\d+")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _normalize(s: str) -> str:
    return s.strip().strip(".,;:!?\"'`").strip().casefold()


def _first_int(s: str) -> int | None:
    m = _INT_RE.search(s.replace(",", ""))
    return int(m.group()) if m else None


def _last_int(s: str) -> int | None:
    matches = _INT_RE.findall(s.replace(",", ""))
    return int(matches[-1]) if matches else None


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of the first JSON object in free-form model output."""
    fenced = _CODE_FENCE_RE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fall back to brace-matching the first balanced {...} span, in case the model
    # added explanation text before/after the JSON despite instructions not to.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _extract_json_array(text: str) -> list | None:
    fenced = _CODE_FENCE_RE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, list):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("[", start + 1)
    return None


def _values_match(expected_value, actual_value) -> bool:
    if isinstance(expected_value, list):
        if not isinstance(actual_value, list):
            return False
        return {_normalize(str(v)) for v in expected_value} == {_normalize(str(v)) for v in actual_value}
    return _normalize(str(expected_value)) == _normalize(str(actual_value))


def score_classification(output: str, expected: str, metadata: dict) -> dict:
    labels = metadata.get("labels", [])
    norm_output = _normalize(output)
    norm_expected = _normalize(expected)

    if norm_output == norm_expected:
        return {"score": 1.0, "detail": "exact_match"}

    # First line/token only, in case the model added trailing text anyway.
    first_line = _normalize(output.splitlines()[0]) if output.strip() else ""
    if first_line == norm_expected:
        return {"score": 1.0, "detail": "exact_match_first_line"}

    # Word-boundary containment: exactly one label from the closed set appears.
    present = [lbl for lbl in labels if re.search(rf"\b{re.escape(_normalize(lbl))}\b", norm_output)]
    if present == [expected] or (len(present) == 1 and _normalize(present[0]) == norm_expected):
        return {"score": 1.0, "detail": "label_containment_match"}
    if not present:
        return {"score": 0.0, "detail": "no_known_label_found"}
    return {"score": 0.0, "detail": "wrong_or_ambiguous_label"}


def score_json_list(output: str, expected: str) -> dict:
    expected_list = json.loads(expected)
    actual_list = _extract_json_array(output)
    if actual_list is None:
        return {"score": 0.0, "detail": "invalid_json", "valid_json": False}
    ok = {_normalize(str(v)) for v in expected_list} == {_normalize(str(v)) for v in actual_list}
    return {"score": 1.0 if ok else 0.0, "detail": "list_match" if ok else "list_mismatch", "valid_json": True}


def score_json_fields(output: str, expected: str, metadata: dict, key_field: str) -> dict:
    expected_obj = json.loads(expected)
    keys = metadata.get(key_field, list(expected_obj.keys()))

    actual_obj = _extract_json_object(output)
    if actual_obj is None:
        return {"score": 0.0, "detail": "invalid_json", "valid_json": False}

    matched = [k for k in keys if k in actual_obj and _values_match(expected_obj[k], actual_obj[k])]
    missing_or_wrong = [k for k in keys if k not in matched]
    score = len(matched) / len(keys) if keys else 0.0
    detail = "all_fields_match" if not missing_or_wrong else f"fields_wrong:{','.join(missing_or_wrong)}"
    return {"score": score, "detail": detail, "valid_json": True}


def _flatten_values(obj) -> list[str]:
    values: list[str] = []

    def rec(v) -> None:
        if isinstance(v, list):
            for item in v:
                rec(item)
        elif isinstance(v, dict):
            for item in v.values():
                rec(item)
        else:
            values.append(str(v))

    rec(obj)
    return values


def _normalize_loose(s: str) -> str:
    s = re.sub(r"[$,]", "", s.strip().casefold())
    return re.sub(r"\s+", " ", s).strip()


def _value_present(expected_value, actual_values: list[str]) -> bool:
    if isinstance(expected_value, list):
        return all(_value_present(v, actual_values) for v in expected_value)
    exp_norm = _normalize_loose(str(expected_value))
    if not exp_norm:
        return False
    return any(exp_norm == _normalize_loose(av) or exp_norm in _normalize_loose(av) for av in actual_values)


def score_information_extraction(output: str, expected: str, metadata: dict) -> dict:
    """Unlike structured_output, information_extraction prompts never specify exact
    key names ("Extract the key fields from this text as JSON:") — so a model using
    a different but equally reasonable key (e.g. "customer_name" instead of
    "customer") is extracting correctly, just naming it differently. Score by
    whether each expected *value* appears anywhere in the output JSON, regardless
    of which key it's filed under, rather than requiring exact key alignment.
    """
    expected_obj = json.loads(expected)
    keys = metadata.get("fields", list(expected_obj.keys()))

    actual_obj = _extract_json_object(output)
    if actual_obj is None:
        return {"score": 0.0, "detail": "invalid_json", "valid_json": False}

    actual_values = _flatten_values(actual_obj)
    matched = [k for k in keys if _value_present(expected_obj[k], actual_values)]
    missing = [k for k in keys if k not in matched]
    score = len(matched) / len(keys) if keys else 0.0
    detail = "all_values_found" if not missing else f"values_missing:{','.join(missing)}"
    return {"score": score, "detail": detail, "valid_json": True}


def score_structured_output(output: str, expected: str, metadata: dict) -> dict:
    if metadata.get("schema_keys") == ["(list)"]:
        return score_json_list(output, expected)
    return score_json_fields(output, expected, metadata, "schema_keys")


def score_summarization(output: str, expected: str, metadata: dict) -> dict:
    """Bag-of-words token F1 between output and reference summary.

    A lexical-overlap proxy, not true semantic similarity (no embeddings/LLM-judge
    used — see module docstring). Documented as a limitation in the Week 5 README.
    """
    out_tokens = re.findall(r"\w+", output.casefold())
    exp_tokens = re.findall(r"\w+", expected.casefold())
    if not out_tokens or not exp_tokens:
        return {"score": 0.0, "detail": "empty_output_or_reference"}

    out_counts: dict[str, int] = {}
    for t in out_tokens:
        out_counts[t] = out_counts.get(t, 0) + 1
    exp_counts: dict[str, int] = {}
    for t in exp_tokens:
        exp_counts[t] = exp_counts.get(t, 0) + 1

    overlap = sum(min(out_counts.get(t, 0), c) for t, c in exp_counts.items())
    precision = overlap / len(out_tokens)
    recall = overlap / len(exp_tokens)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"score": f1, "detail": f"token_f1={f1:.2f}"}


def score_reasoning(output: str, expected: str, metadata: dict) -> dict:
    answer_type = metadata.get("answer_type")
    if answer_type == "day_of_week":
        exp_norm = _normalize(expected)
        ok = re.search(rf"\b{re.escape(exp_norm)}\b", output.casefold()) is not None
        return {"score": 1.0 if ok else 0.0, "detail": "day_match" if ok else "day_mismatch_or_missing"}

    # Despite the system prompt asking for "only the final answer," models
    # frequently show their work anyway (especially at higher precision) — take the
    # *last* number in the output, since that's where a worked solution's final
    # answer lands, not the first number that happens to appear (e.g. an input
    # value restated at the start of a derivation).
    exp_num = _first_int(expected)
    out_num = _last_int(output)
    if out_num is None:
        return {"score": 0.0, "detail": "no_number_in_output"}
    ok = exp_num is not None and out_num == exp_num
    return {"score": 1.0 if ok else 0.0, "detail": "numeric_match" if ok else "numeric_mismatch"}


def _instruction_format_ok(output: str, expected: str, constraint: str) -> bool:
    stripped = output.strip()
    if constraint in ("all_caps_yes_no", "all_caps"):
        return stripped == stripped.upper() and len(stripped.split()) == len(expected.split())
    if constraint == "one_word":
        return len(stripped.split()) == 1
    if constraint == "comma_list_3":
        return len([p for p in stripped.split(",") if p.strip()]) == 3
    if constraint == "comma_list_2":
        return len([p for p in stripped.split(",") if p.strip()]) == 2
    if constraint == "number_only":
        return bool(re.fullmatch(r"\d+", stripped))
    if constraint == "fixed_format":
        return bool(re.match(r"^[^\s:]+(\s[^\s:]+)?\s*:\s*\S", stripped))
    if constraint == "single_emoji":
        return len(stripped) <= 6 and not re.search(r"[A-Za-z0-9]", stripped)
    if constraint == "lowercase_bool":
        return stripped == stripped.lower() and len(stripped.split()) == 1
    return True


def score_instruction_following(output: str, expected: str, metadata: dict) -> dict:
    constraint = metadata.get("constraint", "")
    content_ok = _normalize(output) == _normalize(expected)
    format_ok = _instruction_format_ok(output, expected, constraint)

    if content_ok and format_ok:
        return {"score": 1.0, "detail": "correct"}
    if content_ok and not format_ok:
        return {"score": 0.5, "detail": "content_ok_format_violation"}
    if not content_ok and format_ok:
        return {"score": 0.0, "detail": "format_ok_content_wrong"}
    return {"score": 0.0, "detail": "content_and_format_wrong"}


SCORERS = {
    "classification": score_classification,
    "information_extraction": score_information_extraction,
    "structured_output": score_structured_output,
    "summarization": score_summarization,
    "reasoning": score_reasoning,
    "instruction_following": score_instruction_following,
}


def score_example(category: str, output: str, expected: str, metadata: dict) -> dict:
    return SCORERS[category](output, expected, metadata)
