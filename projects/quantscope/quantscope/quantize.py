"""Produces GGUF files for quantization formats missing from a model's
existing set, via llama-quantize, so bench.sweep can cover the full
candidate format space without requiring every format to be pre-generated
by hand.
"""

from __future__ import annotations

import os

from quantscope.llama_bin import run_llama_quantize


class ImatrixRequiredError(ValueError):
    """Raised when an IQ* format is requested without an imatrix and
    without an explicit override — IQ formats are known to quantize poorly
    without one, and producing a silently-low-quality file is worse than
    refusing.
    """


def missing_formats(existing: dict[str, str], desired: list[str]) -> list[str]:
    """Returns the desired formats not already present in existing (a
    format -> gguf_path mapping, as produced by discovering files on disk).
    """
    return [f for f in desired if f not in existing]


def _requires_imatrix(fmt: str) -> bool:
    return fmt.upper().startswith("IQ")


def produce_missing(
    llama_quantize_bin: str,
    input_gguf: str,
    output_dir: str,
    formats: list[str],
    imatrix_path: str | None = None,
    allow_iq_without_imatrix: bool = False,
) -> dict[str, str]:
    """Quantizes input_gguf into each of formats, writing
    <output_dir>/<model_stem>-<format>.gguf, and returns a format -> path
    mapping for what was produced.

    Validates every requested format before quantizing any of them: if an
    IQ* format is requested without imatrix_path and without
    allow_iq_without_imatrix=True, raises ImatrixRequiredError rather than
    silently producing a known-poor-quality file (or wasting time
    quantizing earlier formats before failing on a later one).
    """
    if not allow_iq_without_imatrix and not imatrix_path:
        needs_imatrix = [f for f in formats if _requires_imatrix(f)]
        if needs_imatrix:
            raise ImatrixRequiredError(
                f"{', '.join(needs_imatrix)} need an imatrix to quantize well. "
                f"Pass --imatrix PATH, or --allow-iq-without-imatrix to produce them "
                f"anyway and accept the quality risk."
            )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_gguf))[0]
    produced = {}
    for fmt in formats:
        out_path = os.path.join(output_dir, f"{stem}-{fmt}.gguf")
        run_llama_quantize(llama_quantize_bin, input_gguf, out_path, fmt, imatrix_path=imatrix_path)
        produced[fmt] = out_path
    return produced
