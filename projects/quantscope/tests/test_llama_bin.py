import subprocess
from unittest.mock import patch

import pytest

from quantscope.llama_bin import (
    LlamaBinError,
    run_llama_bench,
    run_llama_perplexity,
    run_llama_quantize,
)

# A realistic llama-bench -o csv header/row pair: one row for a
# prompt-processing test (n_gen=0), one for a token-generation test
# (n_prompt=0) -- llama-bench's actual behavior when both -p and -n are given.
# Includes the full provenance columns (build/cpu/gpu/backend/config) that
# a real llama-bench emits, not just the speed columns.
BENCH_CSV = (
    "build_commit,cpu_info,gpu_info,backends,model_type,model_size,model_n_params,"
    "n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,use_mmap,"
    "n_prompt,n_gen,avg_ts,stddev_ts\n"
    "abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,"
    "2048,512,8,0,0,1,"
    "512,0,145.32,1.1\n"
    "abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,"
    "2048,512,8,0,0,1,"
    "0,128,42.87,0.5\n"
)


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_llama_bench_parses_prompt_and_gen_rows():
    with patch("subprocess.run", return_value=_completed(stdout=BENCH_CSV)) as mock_run:
        rows = run_llama_bench("llama-bench", "model.gguf", n_prompt=512, n_gen=128, threads=4)
    assert mock_run.call_args.args[0][:2] == ["llama-bench", "-m"]
    assert len(rows) == 2
    prompt_row = next(r for r in rows if r.n_prompt == 512)
    gen_row = next(r for r in rows if r.n_gen == 128)
    assert prompt_row.avg_tokens_per_second == 145.32
    assert gen_row.avg_tokens_per_second == 42.87
    assert prompt_row.model_size_bytes == 4370000000
    assert prompt_row.model_n_params == 8000000000
    assert prompt_row.stddev_tokens_per_second == 1.1
    assert prompt_row.build_commit == "abc123"
    assert prompt_row.n_gpu_layers == 0


def test_run_llama_bench_always_forces_ngl_zero():
    with patch("subprocess.run", return_value=_completed(stdout=BENCH_CSV)) as mock_run:
        run_llama_bench("llama-bench", "model.gguf", n_prompt=512, n_gen=128, threads=4)
    cmd = mock_run.call_args.args[0]
    assert "-ngl" in cmd
    assert cmd[cmd.index("-ngl") + 1] == "0"


def test_run_llama_bench_raises_on_nonzero_exit():
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="model not found")):
        with pytest.raises(LlamaBinError, match="model not found"):
            run_llama_bench("llama-bench", "missing.gguf", 512, 128, 4)


def test_run_llama_bench_raises_on_unparseable_csv():
    with patch("subprocess.run", return_value=_completed(stdout="not,a,valid,bench,csv\n1,2,3,4,5\n")):
        with pytest.raises(LlamaBinError):
            run_llama_bench("llama-bench", "model.gguf", 512, 128, 4)


def test_run_llama_quantize_success():
    with patch("subprocess.run", return_value=_completed(returncode=0)) as mock_run:
        run_llama_quantize("llama-quantize", "in.gguf", "out.gguf", "Q4_K_M")
    assert mock_run.call_args.args[0] == ["llama-quantize", "in.gguf", "out.gguf", "Q4_K_M"]


def test_run_llama_quantize_passes_imatrix_when_given():
    with patch("subprocess.run", return_value=_completed(returncode=0)) as mock_run:
        run_llama_quantize("llama-quantize", "in.gguf", "out.gguf", "IQ2_XXS", imatrix_path="calibration.dat")
    assert mock_run.call_args.args[0] == [
        "llama-quantize", "--imatrix", "calibration.dat", "in.gguf", "out.gguf", "IQ2_XXS",
    ]


def test_run_llama_quantize_raises_on_failure():
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="quantize failed")):
        with pytest.raises(LlamaBinError, match="quantize failed"):
            run_llama_quantize("llama-quantize", "in.gguf", "out.gguf", "Q4_K_M")


PERPLEXITY_OUTPUT = """
[1]4.5000,[2]5.1000,[3]5.8000
Final estimate: PPL = 5.9070 +/- 0.03166
"""


def test_run_llama_perplexity_parses_final_estimate():
    with patch("subprocess.run", return_value=_completed(stdout=PERPLEXITY_OUTPUT)) as mock_run:
        result = run_llama_perplexity("llama-perplexity", "model.gguf", "wiki.test.raw")
    assert result.value == 5.9070
    assert result.error == 0.03166
    assert mock_run.call_args.args[0][:2] == ["llama-perplexity", "-m"]
    assert "-f" in mock_run.call_args.args[0]


def test_run_llama_perplexity_always_forces_ngl_zero():
    # A quantscope v0.2.0 bug: run_llama_bench forced -ngl 0 but
    # run_llama_perplexity didn't, so a Metal/CUDA-enabled build could
    # silently evaluate perplexity with GPU offload while quantscope's own
    # docs claimed CPU was always forced.
    with patch("subprocess.run", return_value=_completed(stdout=PERPLEXITY_OUTPUT)) as mock_run:
        run_llama_perplexity("llama-perplexity", "model.gguf", "wiki.test.raw")
    cmd = mock_run.call_args.args[0]
    assert "-ngl" in cmd
    assert cmd[cmd.index("-ngl") + 1] == "0"


def test_run_llama_perplexity_raises_on_nonzero_exit():
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="dataset not found")):
        with pytest.raises(LlamaBinError, match="dataset not found"):
            run_llama_perplexity("llama-perplexity", "model.gguf", "missing.raw")


def test_run_llama_perplexity_raises_when_final_estimate_missing():
    with patch("subprocess.run", return_value=_completed(stdout="no final estimate here")):
        with pytest.raises(LlamaBinError, match="Final estimate"):
            run_llama_perplexity("llama-perplexity", "model.gguf", "wiki.test.raw")
