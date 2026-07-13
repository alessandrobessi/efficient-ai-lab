"""End-to-end CLI tests using stub shell scripts standing in for
llama-bench/llama-quantize, since no real llama.cpp build is assumed to be
available (matching the roadmap's own testing strategy: unit/CLI tests
never require a real llama.cpp build; only a manual pre-release step does).
"""

import os
import stat

from quantscope.cli import main

LLAMA_BENCH_STUB = """#!/bin/sh
if [ "$1" = "-m" ]; then
  cat <<'CSV'
build_commit,model_size,model_n_params,n_prompt,n_gen,avg_ts,stddev_ts
abc123,4370000000,8000000000,512,0,145.32,1.1
abc123,4370000000,8000000000,0,128,42.87,0.5
CSV
  exit 0
else
  echo "system_info: n_threads = 4 | CPU : AVX2 = 1 | AVX512 = 0 | FMA = 1 |" 1>&2
  exit 1
fi
"""

LLAMA_QUANTIZE_STUB = """#!/bin/sh
if [ "$1" = "--help" ]; then
  cat <<'HELP'
Allowed quantization types:
   2  or  Q4_0    :  3.56G, +0.2166 ppl @ Llama-3-8B
  15  or  Q4_K_M  :  4.37G, +0.0632 ppl @ Llama-3-8B
  19  or  IQ2_XXS :  2.06 bpw quantization
HELP
  exit 0
fi
touch "$2"
exit 0
"""


def _write_stub(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_cli_predict(capsys):
    rc = main(["predict", "Q8_0", "Q4_0", "F16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_0" in out
    assert "heuristic only, not a measurement" in out


def test_cli_formats(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    rc = main(["formats", "--llama-quantize-bin", stub])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out
    assert "IQ2_XXS" in out
    assert "excluded (need an imatrix): IQ2_XXS" in out


def test_cli_formats_with_imatrix(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    rc = main(["formats", "--llama-quantize-bin", stub, "--imatrix"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "excluded" not in out


def test_cli_cpu_info(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    rc = main(["cpu-info", "--llama-bench-bin", stub])
    assert rc == 0
    out = capsys.readouterr().out
    assert "AVX2: yes" in out
    assert "AVX512: no" in out


def test_cli_bench_end_to_end(tmp_path, capsys):
    bench_stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    gguf_path = tmp_path / "model-q4.gguf"
    gguf_path.write_bytes(b"0" * (10 * 1024 * 1024))
    out_csv = tmp_path / "results.csv"

    rc = main(
        [
            "bench",
            "--llama-bench-bin", bench_stub,
            "--gguf", f"Q4_K_M={gguf_path}",
            "--output", str(out_csv),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out
    assert out_csv.exists()
    assert "pareto_optimal" in out_csv.read_text()
