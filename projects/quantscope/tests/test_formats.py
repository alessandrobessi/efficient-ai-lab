from quantscope.formats import applicable_formats, parse_allowed_types

HELP_TEXT = """
usage: ./llama-quantize [--help] [--allow-requantize] ... model type

Allowed quantization types:
   2  or  Q4_0    :  3.56G, +0.2166 ppl @ Llama-3-8B
   3  or  Q4_1    :  3.90G, +0.1585 ppl @ Llama-3-8B
  15  or  Q4_K_M  :  4.37G, +0.0632 ppl @ Llama-3-8B
  19  or  IQ2_XXS :  2.06 bpw quantization
  20  or  IQ2_XS  :  2.31 bpw quantization
   7  or  Q8_0    :  6.70G, +0.0004 ppl @ Llama-3-8B
   1  or  F16     : 14.00G, -0.0020 ppl @ Mistral-7B
          COPY    : only copy tensors, no quantizing
"""


def test_parse_allowed_types_extracts_names():
    formats = parse_allowed_types(HELP_TEXT)
    assert "Q4_0" in formats
    assert "Q4_K_M" in formats
    assert "IQ2_XXS" in formats
    assert "F16" in formats


def test_parse_allowed_types_excludes_copy_pseudo_format():
    formats = parse_allowed_types(HELP_TEXT)
    assert "COPY" not in formats


def test_applicable_formats_excludes_iq_without_imatrix():
    all_formats = ["Q4_0", "Q4_K_M", "IQ2_XXS", "IQ2_XS", "F16"]
    usable = applicable_formats(all_formats, has_imatrix=False)
    assert "IQ2_XXS" not in usable
    assert "IQ2_XS" not in usable
    assert "Q4_0" in usable
    assert "F16" in usable


def test_applicable_formats_includes_iq_with_imatrix():
    all_formats = ["Q4_0", "IQ2_XXS"]
    usable = applicable_formats(all_formats, has_imatrix=True)
    assert usable == all_formats
