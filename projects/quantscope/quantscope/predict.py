"""A heuristic, no-benchmarking format predictor — deliberately kept
separate from bench.py so a fast-but-approximate estimate is never confused
with a measured one.

This program's own findings (Weeks 4 and 6) are the reason this module
carries CONFIDENCE_CAVEAT everywhere its output is shown: quantization speed
does not correlate cleanly with bit width, because SIMD/kernel-fit effects
(the REPACK mechanism — see docs/methodology/glossary.md) can make a
nominally larger format faster than a nominally smaller one. predict exists
for a quick first guess when running a full `bench` sweep isn't worth it
yet, not as a substitute for one.
"""

from __future__ import annotations

from dataclasses import dataclass

CONFIDENCE_CAVEAT = (
    "heuristic only, not a measurement: this program's own benchmarking (Weeks 4 "
    "and 6) found quantization speed does not correlate cleanly with bit width -- "
    "SIMD/kernel-fit effects can make a nominally larger format faster than a "
    "smaller one. Run `quantscope bench` for a real, measured answer."
)

# Approximate bits-per-weight for common GGUF formats, used only to produce
# a rough ranking (smaller = naively presumed faster / smaller on disk).
# Deliberately approximate and not exhaustive: unknown formats are still
# accepted (see predict()), just ranked last with bpw=None.
_APPROX_BITS_PER_WEIGHT = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_1": 6.0,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q4_1": 5.0,
    "Q4_K_M": 4.8,
    "Q4_K_S": 4.6,
    "IQ4_XS": 4.3,
    "Q4_0": 4.5,
    "Q3_K_L": 4.1,
    "Q3_K_M": 3.9,
    "Q3_K_S": 3.5,
    "IQ3_XXS": 3.1,
    "IQ2_XS": 2.3,
    "Q2_K": 2.6,
    "IQ2_XXS": 2.1,
}


@dataclass
class Prediction:
    format: str
    approx_bits_per_weight: float | None
    predicted_rank: int  # 1 = naively predicted fastest / smallest


def predict(formats: list[str]) -> list[Prediction]:
    """Ranks formats by approximate bit width, ascending (smaller bpw first).
    Formats not in the known table sort after all known ones, in their
    original relative order among themselves, rather than being dropped —
    predict never silently excludes a format bench.sweep would have covered.
    """
    annotated = [(f, _APPROX_BITS_PER_WEIGHT.get(f.upper())) for f in formats]
    ranked = sorted(
        enumerate(annotated),
        key=lambda item: (item[1][1] is None, item[1][1] if item[1][1] is not None else 0.0, item[0]),
    )
    return [
        Prediction(format=fmt, approx_bits_per_weight=bpw, predicted_rank=i + 1)
        for i, (_, (fmt, bpw)) in enumerate(ranked)
    ]
