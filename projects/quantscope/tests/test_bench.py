import os
from unittest.mock import patch

from quantscope.bench import sweep
from quantscope.llama_bin import BenchRow


def fake_rows_for(path):
    if "q4" in path:
        return [
            BenchRow(n_prompt=512, n_gen=0, avg_tokens_per_second=150.0, model_size_bytes=4_000_000, model_n_params=8_000_000_000),
            BenchRow(n_prompt=0, n_gen=128, avg_tokens_per_second=40.0, model_size_bytes=4_000_000, model_n_params=8_000_000_000),
        ]
    return [
        BenchRow(n_prompt=512, n_gen=0, avg_tokens_per_second=120.0, model_size_bytes=8_000_000, model_n_params=8_000_000_000),
        BenchRow(n_prompt=0, n_gen=128, avg_tokens_per_second=30.0, model_size_bytes=8_000_000, model_n_params=8_000_000_000),
    ]


def test_sweep_maps_prompt_and_gen_rows_per_format(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        results = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)})

    by_format = {r.format: r for r in results}
    assert by_format["Q4_K_M"].prompt_tokens_per_second == 150.0
    assert by_format["Q4_K_M"].gen_tokens_per_second == 40.0
    assert by_format["Q8_0"].gen_tokens_per_second == 30.0
    assert by_format["Q4_K_M"].file_size_mb == os.path.getsize(q4_path) / (1024 * 1024)
