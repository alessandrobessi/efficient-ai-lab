"""Ranks formats by approximate storage size — deliberately not called a
speed predictor. quantscope's own central finding (Weeks 4 and 6 of the
research program it comes from) is that quantization speed does not track
bit width reliably, because SIMD/kernel-fit effects (the REPACK mechanism —
see docs/methodology/glossary.md) can make a nominally larger format faster
than a nominally smaller one. A command that ranked by bit width and called
that a "prediction" would directly contradict its own reason for existing.
This module ranks by size only — a real, exact property of a format,
independent of any performance claim — for a quick before-you-benchmark
sanity check ("is this even going to fit in memory"), not a substitute for
`bench`.
"""

from __future__ import annotations

from dataclasses import dataclass

NOTE = (
    "this ranks formats by approximate storage size only -- it is NOT a speed "
    "prediction. This program's own benchmarking (Weeks 4 and 6) found quantization "
    "speed does not correlate cleanly with bit width, so a smaller format here is not "
    "necessarily a faster one. Run `quantscope bench` to actually measure speed."
)

# Approximate bits-per-weight for common GGUF formats, used only to rank by
# storage footprint. Deliberately approximate and not exhaustive: unknown
# formats are still accepted (see estimate_size()), just ranked last with
# bpw=None.
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
class SizeEstimate:
    format: str
    approx_bits_per_weight: float | None
    size_rank: int  # 1 = smallest approximate storage footprint


def estimate_size(formats: list[str]) -> list[SizeEstimate]:
    """Ranks formats by approximate bit width, ascending (smaller bpw
    first) -- storage size only, see module docstring for why this is
    deliberately not framed as a speed estimate. Formats not in the known
    table sort after all known ones, in their original relative order among
    themselves, rather than being dropped -- this never silently excludes a
    format bench.sweep would have covered.
    """
    annotated = [(f, _APPROX_BITS_PER_WEIGHT.get(f.upper())) for f in formats]
    ranked = sorted(
        enumerate(annotated),
        key=lambda item: (item[1][1] is None, item[1][1] if item[1][1] is not None else 0.0, item[0]),
    )
    return [
        SizeEstimate(format=fmt, approx_bits_per_weight=bpw, size_rank=i + 1)
        for i, (_, (fmt, bpw)) in enumerate(ranked)
    ]
