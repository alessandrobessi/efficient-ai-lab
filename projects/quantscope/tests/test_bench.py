import os
from unittest.mock import patch

import pytest

from quantscope.bench import ModelIdentityError, sweep
from quantscope.llama_bin import BenchRow


def _row(n_prompt, n_gen, avg_ts, model_n_params=8_000_000_000, model_size_bytes=4_000_000, model_type="qwen2 0.5B"):
    return BenchRow(
        n_prompt=n_prompt,
        n_gen=n_gen,
        avg_tokens_per_second=avg_ts,
        stddev_tokens_per_second=0.5,
        model_size_bytes=model_size_bytes,
        model_n_params=model_n_params,
        model_type=model_type,
        build_commit="abc123",
        cpu_info="Apple M4",
        gpu_info="Metal",
        backends="CPU",
        n_threads=8,
        n_batch=2048,
        n_ubatch=512,
        n_gpu_layers=0,
        flash_attn="0",
        use_mmap="1",
    )


def fake_rows_for(path, model_n_params=8_000_000_000, model_type="qwen2 0.5B"):
    if "q4" in path:
        return [
            _row(512, 0, 150.0, model_n_params=model_n_params, model_size_bytes=4_000_000, model_type=model_type),
            _row(0, 128, 40.0, model_n_params=model_n_params, model_size_bytes=4_000_000, model_type=model_type),
        ]
    return [
        _row(512, 0, 120.0, model_n_params=model_n_params, model_size_bytes=8_000_000, model_type=model_type),
        _row(0, 128, 30.0, model_n_params=model_n_params, model_size_bytes=8_000_000, model_type=model_type),
    ]


def test_sweep_maps_prompt_and_gen_rows_per_format(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        results, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, compute_hashes=False)

    by_format = {r.format: r for r in results}
    assert by_format["Q4_K_M"].prompt_tokens_per_second == 150.0
    assert by_format["Q4_K_M"].gen_tokens_per_second == 40.0
    assert by_format["Q8_0"].gen_tokens_per_second == 30.0
    assert by_format["Q4_K_M"].file_size_mb == os.path.getsize(q4_path) / (1024 * 1024)
    assert by_format["Q4_K_M"].perplexity is None
    assert by_format["Q4_K_M"].model_n_params == 8_000_000_000


def test_sweep_builds_manifest_with_shared_environment(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        results, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, compute_hashes=False)

    assert manifest.cpu_info == "Apple M4"
    assert manifest.llama_cpp_build_commit == "abc123"
    assert manifest.n_gpu_layers == 0
    assert len(manifest.models) == 1
    assert manifest.models[0].format == "Q4_K_M"


def test_sweep_computes_sha256_when_requested(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"deterministic content")

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        _, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, compute_hashes=True)

    import hashlib
    expected = hashlib.sha256(b"deterministic content").hexdigest()
    assert manifest.models[0].sha256 == expected


def test_sweep_rejects_different_parameter_counts(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    def fake_bench(bin_, path, *a, **k):
        # q8 file reports a different parameter count -- not the same model.
        return fake_rows_for(path, model_n_params=3_000_000_000 if "q8" in path else 8_000_000_000)

    with patch("quantscope.bench.run_llama_bench", side_effect=fake_bench):
        with pytest.raises(ModelIdentityError, match="don't look like the same base model"):
            sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, compute_hashes=False)


def test_sweep_skips_identity_check_when_params_unreported(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1], model_n_params=0)):
        # Should not raise even though we can't confirm they match -- 0 means "unreported," not "mismatched."
        results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, compute_hashes=False)
    assert len(results) == 2


def test_sweep_without_quality_eval_args_leaves_perplexity_none(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)
    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity") as mock_ppl:
            results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, compute_hashes=False)
    assert results[0].perplexity is None
    mock_ppl.assert_not_called()


def test_sweep_with_quality_eval_args_calls_perplexity_per_format(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity", return_value=5.5) as mock_ppl:
            results, _ = sweep(
                "llama-bench",
                {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)},
                llama_perplexity_bin="llama-perplexity",
                perplexity_dataset="wiki.test.raw",
                compute_hashes=False,
            )
    assert mock_ppl.call_count == 2
    assert all(r.perplexity == 5.5 for r in results)
    assert all(r.ppl_delta is None for r in results)  # no baseline given


def test_sweep_computes_ppl_delta_and_ratio_against_baseline(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    ppl_by_path = {str(q4_path): 6.6, str(q8_path): 6.0}

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity", side_effect=lambda bin_, path, *a, **k: ppl_by_path[path]):
            results, _ = sweep(
                "llama-bench",
                {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)},
                llama_perplexity_bin="llama-perplexity",
                perplexity_dataset="wiki.test.raw",
                perplexity_baseline_format="Q8_0",
                compute_hashes=False,
            )
    by_format = {r.format: r for r in results}
    assert by_format["Q8_0"].ppl_delta == 0.0
    assert by_format["Q8_0"].ppl_ratio == 1.0
    assert by_format["Q4_K_M"].ppl_delta == pytest.approx(0.6)
    assert by_format["Q4_K_M"].ppl_ratio == pytest.approx(1.1)
