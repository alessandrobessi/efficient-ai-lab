import os
from unittest.mock import patch

import pytest

from quantscope.bench import ModelIdentityError, sweep
from quantscope.llama_bin import BenchRow, PerplexityResult


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
        results, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=2, compute_hashes=False)

    by_format = {r.format: r for r in results}
    assert by_format["Q4_K_M"].prompt_tokens_per_second == 150.0
    assert by_format["Q4_K_M"].gen_tokens_per_second == 40.0
    assert by_format["Q8_0"].gen_tokens_per_second == 30.0
    assert by_format["Q4_K_M"].file_size_mb == os.path.getsize(q4_path) / (1024 * 1024)
    assert by_format["Q4_K_M"].perplexity is None
    assert by_format["Q4_K_M"].model_n_params == 8_000_000_000
    # Deterministic mock -> every round returns the same value -> zero
    # cross-round spread, but the samples themselves should still be
    # collected (one per round).
    assert by_format["Q4_K_M"].gen_tokens_per_second_samples == [40.0, 40.0]
    assert by_format["Q4_K_M"].gen_tokens_per_second_stddev == 0.0


def test_sweep_aggregates_varying_samples_across_rounds(tmp_path):
    # Unlike the deterministic-mock tests above, this format's gen speed
    # actually differs round to round (e.g. thermal drift) -- proving sweep()
    # computes a real cross-round mean/stddev rather than just echoing
    # llama-bench's own single-launch numbers.
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)
    gen_values = iter([38.0, 40.0, 42.0])

    def fake_bench(bin_, path, *a, **k):
        return [
            _row(512, 0, 150.0),
            _row(0, 128, next(gen_values)),
        ]

    with patch("quantscope.bench.run_llama_bench", side_effect=fake_bench):
        results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, rounds=3, compute_hashes=False)

    assert results[0].gen_tokens_per_second_samples == [38.0, 40.0, 42.0]
    assert results[0].gen_tokens_per_second == pytest.approx(40.0)
    assert results[0].gen_tokens_per_second_stddev == pytest.approx(2.0)


def test_sweep_randomizes_order_per_round_and_records_it(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        _, manifest = sweep(
            "llama-bench",
            {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)},
            rounds=5,
            rng_seed=42,
            compute_hashes=False,
        )

    assert manifest.rounds == 5
    assert len(manifest.round_orders) == 5
    for order in manifest.round_orders:
        assert set(order) == {"Q4_K_M", "Q8_0"}
    # A seeded RNG with only 2 formats will sometimes produce the same order
    # twice in a row by chance, but across 5 rounds both orders should show
    # up -- proving it's actually shuffling, not just repeating input order.
    assert len({tuple(o) for o in manifest.round_orders}) == 2


def test_sweep_same_seed_is_reproducible(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        _, manifest_a = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=5, rng_seed=7, compute_hashes=False)
        _, manifest_b = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=5, rng_seed=7, compute_hashes=False)

    assert manifest_a.round_orders == manifest_b.round_orders


def test_sweep_builds_manifest_with_shared_environment(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        results, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, rounds=2, compute_hashes=False)

    assert manifest.cpu_info == "Apple M4"
    assert manifest.llama_cpp_build_commit == "abc123"
    assert manifest.n_gpu_layers == 0
    assert manifest.n_prompt == 512
    assert manifest.n_gen == 128
    assert len(manifest.models) == 1
    assert manifest.models[0].format == "Q4_K_M"


def test_sweep_manifest_stores_basename_not_absolute_path(tmp_path):
    # A quantscope v0.2.1 fix: the manifest used to store the full path a
    # sweep was invoked with, which for a real run is a machine-specific
    # temp directory -- not useful for reproducibility (the hash is) and
    # not something a committed benchmark artifact should publish.
    q4_path = tmp_path / "some-machine-specific-tmp-dir-model-q4.gguf"
    q4_path.write_bytes(b"content")

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        _, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, rounds=1, compute_hashes=False)

    assert manifest.models[0].filename == "some-machine-specific-tmp-dir-model-q4.gguf"
    assert str(tmp_path) not in manifest.models[0].filename


def test_sweep_computes_sha256_when_requested(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"deterministic content")

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        _, manifest = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, rounds=1, compute_hashes=True)

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
            sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=1, compute_hashes=False)


def test_sweep_rejects_mismatched_model_type_prefix_even_with_matching_params(tmp_path):
    # Same parameter count (a base model and, say, a different architecture
    # tuned to land on the same size can coincide), but a different
    # architecture/size prefix in model_type -- the soft secondary check
    # this fix adds specifically to catch this case.
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-other.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    def fake_bench(bin_, path, *a, **k):
        model_type = "llama3 1B" if "other" in path else "qwen2 1B"
        return fake_rows_for(path, model_n_params=8_000_000_000, model_type=model_type)

    with patch("quantscope.bench.run_llama_bench", side_effect=fake_bench):
        with pytest.raises(ModelIdentityError, match="don't look like the same base model"):
            sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=1, compute_hashes=False)


def test_sweep_accepts_same_model_type_prefix_with_quant_suffix_variation(tmp_path):
    # The prefix check must NOT reject legitimate same-model sweeps just
    # because model_type's trailing quant-format token differs per format
    # (this was the exact false-positive bug the original model_type
    # comparison had, before it was narrowed down to model_n_params only).
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    def fake_bench(bin_, path, *a, **k):
        model_type = "qwen2 1B Q4_K - Medium" if "q4" in path else "qwen2 1B Q8_0"
        return fake_rows_for(path, model_n_params=8_000_000_000, model_type=model_type)

    with patch("quantscope.bench.run_llama_bench", side_effect=fake_bench):
        results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=1, compute_hashes=False)
    assert len(results) == 2


def test_sweep_skips_identity_check_when_params_unreported(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1], model_n_params=0)):
        # Should not raise even though we can't confirm they match -- 0 means "unreported," not "mismatched."
        results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)}, rounds=1, compute_hashes=False)
    assert len(results) == 2


def test_sweep_without_quality_eval_args_leaves_perplexity_none(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)
    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity") as mock_ppl:
            results, _ = sweep("llama-bench", {"Q4_K_M": str(q4_path)}, rounds=1, compute_hashes=False)
    assert results[0].perplexity is None
    assert results[0].perplexity_error is None
    mock_ppl.assert_not_called()


def test_sweep_with_quality_eval_args_calls_perplexity_once_per_format_not_per_round(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity", return_value=PerplexityResult(value=5.5, error=0.02)) as mock_ppl:
            results, manifest = sweep(
                "llama-bench",
                {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)},
                rounds=10,  # perplexity must NOT be called 10x per format
                llama_perplexity_bin="llama-perplexity",
                perplexity_dataset="wiki.test.raw",
                compute_hashes=False,
            )
    assert mock_ppl.call_count == 2  # once per format, regardless of rounds
    assert all(r.perplexity == 5.5 for r in results)
    assert all(r.perplexity_error == 0.02 for r in results)
    assert all(r.ppl_delta is None for r in results)  # no baseline given
    assert manifest.perplexity_threads > 0
    assert manifest.perplexity_n_gpu_layers == 0


def test_sweep_computes_ppl_delta_and_ratio_against_baseline(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q8_path = tmp_path / "model-q8.gguf"
    q4_path.write_bytes(b"0" * 100)
    q8_path.write_bytes(b"0" * 200)

    ppl_by_path = {str(q4_path): 6.6, str(q8_path): 6.0}

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch(
            "quantscope.bench.run_llama_perplexity",
            side_effect=lambda bin_, path, *a, **k: PerplexityResult(value=ppl_by_path[path], error=0.01),
        ):
            results, manifest = sweep(
                "llama-bench",
                {"Q4_K_M": str(q4_path), "Q8_0": str(q8_path)},
                rounds=1,
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
    assert manifest.perplexity_baseline_format == "Q8_0"


def test_sweep_hashes_perplexity_dataset_when_given(tmp_path):
    q4_path = tmp_path / "model-q4.gguf"
    q4_path.write_bytes(b"0" * 100)
    dataset = tmp_path / "wiki.test.raw"
    dataset.write_bytes(b"some real prose")

    with patch("quantscope.bench.run_llama_bench", side_effect=lambda *a, **k: fake_rows_for(a[1])):
        with patch("quantscope.bench.run_llama_perplexity", return_value=PerplexityResult(value=5.5, error=0.02)):
            _, manifest = sweep(
                "llama-bench",
                {"Q4_K_M": str(q4_path)},
                rounds=1,
                llama_perplexity_bin="llama-perplexity",
                perplexity_dataset=str(dataset),
                compute_hashes=True,
            )
    import hashlib
    assert manifest.perplexity_dataset_sha256 == hashlib.sha256(b"some real prose").hexdigest()
