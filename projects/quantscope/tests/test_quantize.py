from unittest.mock import patch

import pytest

from quantscope.quantize import ImatrixRequiredError, missing_formats, produce_missing


def test_missing_formats_excludes_existing():
    existing = {"Q4_K_M": "/models/m-Q4_K_M.gguf"}
    result = missing_formats(existing, ["Q4_K_M", "Q5_K_M", "Q8_0"])
    assert result == ["Q5_K_M", "Q8_0"]


def test_produce_missing_calls_llama_quantize_per_format(tmp_path):
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")
    output_dir = tmp_path / "out"

    with patch("quantscope.quantize.run_llama_quantize") as mock_quantize:
        produced = produce_missing("llama-quantize", str(input_gguf), str(output_dir), ["Q4_K_M", "Q5_K_M"])

    assert mock_quantize.call_count == 2
    assert set(produced.keys()) == {"Q4_K_M", "Q5_K_M"}
    for fmt, path in produced.items():
        assert path.endswith(f"model-f16-{fmt}.gguf")


def test_produce_missing_rejects_iq_format_without_imatrix(tmp_path):
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")

    with patch("quantscope.quantize.run_llama_quantize") as mock_quantize:
        with pytest.raises(ImatrixRequiredError, match="IQ2_XXS"):
            produce_missing("llama-quantize", str(input_gguf), str(tmp_path / "out"), ["Q4_K_M", "IQ2_XXS"])
    mock_quantize.assert_not_called()  # validated before quantizing anything, not partway through


def test_produce_missing_allows_iq_format_with_imatrix(tmp_path):
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")

    with patch("quantscope.quantize.run_llama_quantize") as mock_quantize:
        produce_missing(
            "llama-quantize", str(input_gguf), str(tmp_path / "out"), ["IQ2_XXS"], imatrix_path="calibration.dat"
        )
    mock_quantize.assert_called_once_with(
        "llama-quantize", str(input_gguf), str(tmp_path / "out" / "model-f16-IQ2_XXS.gguf"), "IQ2_XXS",
        imatrix_path="calibration.dat",
    )


def test_produce_missing_allows_iq_format_with_explicit_override(tmp_path):
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")

    with patch("quantscope.quantize.run_llama_quantize") as mock_quantize:
        produce_missing(
            "llama-quantize", str(input_gguf), str(tmp_path / "out"), ["IQ2_XXS"], allow_iq_without_imatrix=True
        )
    assert mock_quantize.call_count == 1
