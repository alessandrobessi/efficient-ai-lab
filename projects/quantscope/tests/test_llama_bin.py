import subprocess
from unittest.mock import patch

import pytest

from quantscope.llama_bin import (
    LlamaBinError,
    run_llama_bench,
    run_llama_quantize,
)

# A trimmed but realistic llama-bench -o csv header/row pair: one row for a
# prompt-processing test (n_gen=0), one for a token-generation test
# (n_prompt=0) -- llama-bench's actual behavior when both -p and -n are given.
BENCH_CSV = (
    "build_commit,model_size,model_n_params,n_prompt,n_gen,avg_ts,stddev_ts\n"
    "abc123,4370000000,8000000000,512,0,145.32,1.1\n"
    "abc123,4370000000,8000000000,0,128,42.87,0.5\n"
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


def test_run_llama_quantize_raises_on_failure():
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="quantize failed")):
        with pytest.raises(LlamaBinError, match="quantize failed"):
            run_llama_quantize("llama-quantize", "in.gguf", "out.gguf", "Q4_K_M")
