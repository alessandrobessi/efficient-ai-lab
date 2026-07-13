"""Discovers which GGUF quantization formats a specific `llama-quantize`
binary actually supports (rather than hardcoding a list, since llama.cpp
adds and renames formats across versions) and filters that list down to
formats plausibly applicable to a given model.
"""

from __future__ import annotations

import re

from quantscope.llama_bin import get_quantize_help

# Matches llama-quantize --help's "Allowed quantization types:" table, e.g.
#    2  or  Q4_0    :  3.56G, +0.2166 ppl @ Llama-3-8B
#   19  or  IQ2_XXS :  2.06 bpw quantization
_ALLOWED_TYPE_RE = re.compile(r"^\s*\d*\s*or\s+([A-Za-z0-9_]+)\s*:", re.MULTILINE)

# IQ* formats need an importance matrix (imatrix) to quantize well; without
# one, llama-quantize will still produce a file but quality is known to be
# poor. quantscope excludes them from the default candidate set rather than
# silently including a format that needs an input this tool doesn't have.
_IMATRIX_REQUIRED_PREFIX = "IQ"


def parse_allowed_types(help_text: str) -> list[str]:
    """Parses the format names out of llama-quantize --help text."""
    names = {m.upper() for m in _ALLOWED_TYPE_RE.findall(help_text)}
    return sorted(names)


def list_supported_formats(llama_quantize_bin: str) -> list[str]:
    """Queries the installed llama-quantize binary for its currently
    supported format list, rather than assuming a hardcoded one is current.
    """
    return parse_allowed_types(get_quantize_help(llama_quantize_bin))


def applicable_formats(all_formats: list[str], has_imatrix: bool = False) -> list[str]:
    """Filters a format list down to ones plausibly usable without further
    input. Excludes IQ* formats unless an imatrix is available, since
    quantizing to them without one is known to produce poor quality —
    a real gap this tool would otherwise paper over by including them
    unconditionally.
    """
    if has_imatrix:
        return list(all_formats)
    return [f for f in all_formats if not f.upper().startswith(_IMATRIX_REQUIRED_PREFIX)]
