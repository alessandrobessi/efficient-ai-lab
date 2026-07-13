from unittest.mock import patch

from quantscope.quantize import missing_formats, produce_missing


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
