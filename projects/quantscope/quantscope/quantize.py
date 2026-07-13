"""Produces GGUF files for quantization formats missing from a model's
existing set, via llama-quantize, so bench.sweep can cover the full
candidate format space without requiring every format to be pre-generated
by hand.
"""

from __future__ import annotations

import os

from quantscope.llama_bin import run_llama_quantize


def missing_formats(existing: dict[str, str], desired: list[str]) -> list[str]:
    """Returns the desired formats not already present in existing (a
    format -> gguf_path mapping, as produced by discovering files on disk).
    """
    return [f for f in desired if f not in existing]


def produce_missing(
    llama_quantize_bin: str,
    input_gguf: str,
    output_dir: str,
    formats: list[str],
) -> dict[str, str]:
    """Quantizes input_gguf into each of formats, writing
    <output_dir>/<model_stem>-<format>.gguf, and returns a format -> path
    mapping for what was produced.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_gguf))[0]
    produced = {}
    for fmt in formats:
        out_path = os.path.join(output_dir, f"{stem}-{fmt}.gguf")
        run_llama_quantize(llama_quantize_bin, input_gguf, out_path, fmt)
        produced[fmt] = out_path
    return produced
