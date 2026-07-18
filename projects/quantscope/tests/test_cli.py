"""End-to-end CLI tests using stub shell scripts standing in for
llama-bench/llama-quantize, since no real llama.cpp build is assumed to be
available (matching the roadmap's own testing strategy: unit/CLI tests
never require a real llama.cpp build; only a manual pre-release step does).
"""

import json
import stat

from quantscope.cli import main

LLAMA_BENCH_STUB = """#!/bin/sh
if [ "$1" = "-m" ]; then
  cat <<'CSV'
build_commit,cpu_info,gpu_info,backends,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,use_mmap,n_prompt,n_gen,avg_ts,stddev_ts
abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,2048,512,8,0,0,1,512,0,145.32,1.1
abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,2048,512,8,0,0,1,0,128,42.87,0.5
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
if [ "$1" = "--imatrix" ]; then
  touch "$4"
else
  touch "$2"
fi
exit 0
"""

LLAMA_PERPLEXITY_STUB = """#!/bin/sh
echo "Final estimate: PPL = 6.1234 +/- 0.02" 1>&2
exit 0
"""


def _write_stub(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_cli_version(capsys):
    try:
        main(["--version"])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "quantscope" in out


def test_cli_estimate_size(capsys):
    rc = main(["estimate-size", "Q8_0", "Q4_0", "F16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_0" in out
    assert "NOT a speed" in out


def test_cli_formats(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    rc = main(["formats", "--llama-quantize-bin", stub])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out
    assert "IQ2_XXS" in out
    assert "excluded (need an imatrix): IQ2_XXS" in out


def test_cli_formats_with_imatrix_path(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    calibration = tmp_path / "calibration.dat"
    calibration.write_bytes(b"0")
    rc = main(["formats", "--llama-quantize-bin", stub, "--imatrix", str(calibration)])
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
            "--rounds", "2",
            "--output", str(out_csv),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out
    assert out_csv.exists()
    assert "pareto_optimal" in out_csv.read_text()
    assert "perplexity" not in out_csv.read_text()  # no quality eval given
    # Raw per-round sample lists are manifest/API-only, not CSV columns.
    assert "samples" not in out_csv.read_text()

    manifest_path = tmp_path / "results_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cpu_info"] == "Apple M4"
    assert manifest["n_gpu_layers"] == 0
    assert manifest["n_prompt"] == 512
    assert manifest["n_gen"] == 128
    assert manifest["rounds"] == 2
    assert len(manifest["round_orders"]) == 2
    assert manifest["round_orders"][0] == ["Q4_K_M"]
    assert manifest["pareto_minimize"] == ["file_size_mb"]
    assert manifest["pareto_maximize"] == ["gen_tokens_per_second"]
    assert len(manifest["models"]) == 1
    assert manifest["models"][0]["sha256"]  # hashing on by default when --output is given
    assert manifest["models"][0]["filename"] == "model-q4.gguf"
    assert "path" not in manifest["models"][0]


def test_cli_bench_skip_hash(tmp_path):
    bench_stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    gguf_path = tmp_path / "model-q4.gguf"
    gguf_path.write_bytes(b"0" * 1024)
    out_csv = tmp_path / "results.csv"

    main(["bench", "--llama-bench-bin", bench_stub, "--gguf", f"Q4_K_M={gguf_path}", "--rounds", "1", "--output", str(out_csv), "--skip-hash"])
    manifest = json.loads((tmp_path / "results_manifest.json").read_text())
    assert manifest["models"][0]["sha256"] == ""


def test_cli_bench_skips_hashing_when_no_output_requested(tmp_path):
    # A quantscope v0.2.0 bug: every GGUF got sha256-hashed even when
    # --output wasn't given, so there was nowhere for the hash to go --
    # wasted work on potentially multi-gigabyte files for no reason.
    from unittest.mock import patch

    bench_stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    gguf_path = tmp_path / "model-q4.gguf"
    gguf_path.write_bytes(b"0" * 1024)

    with patch("quantscope.bench.sha256_file") as mock_hash:
        rc = main(["bench", "--llama-bench-bin", bench_stub, "--gguf", f"Q4_K_M={gguf_path}", "--rounds", "1"])
    assert rc == 0
    mock_hash.assert_not_called()


def test_cli_bench_rejects_mismatched_models(tmp_path):
    # sweep()'s own mismatch-detection logic is covered thoroughly in
    # test_bench.py; this only confirms the CLI surfaces it as a clean
    # SystemExit rather than a raw traceback. Since --llama-bench-bin is one
    # binary shared across every --gguf entry in real usage, this stub
    # decides which model_n_params to report based on the GGUF path it's
    # given, to simulate two --gguf entries secretly being different models.
    stub_script = """#!/bin/sh
if [ "$1" = "-m" ]; then
  case "$2" in
    *other*)
      cat <<'CSV'
build_commit,cpu_info,gpu_info,backends,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,use_mmap,n_prompt,n_gen,avg_ts,stddev_ts
abc123,Apple M4,Metal,CPU,qwen2 3B,8000000000,3000000000,2048,512,8,0,0,1,512,0,90.0,1.0
abc123,Apple M4,Metal,CPU,qwen2 3B,8000000000,3000000000,2048,512,8,0,0,1,0,128,25.0,0.5
CSV
      ;;
    *)
      cat <<'CSV'
build_commit,cpu_info,gpu_info,backends,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,flash_attn,use_mmap,n_prompt,n_gen,avg_ts,stddev_ts
abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,2048,512,8,0,0,1,512,0,145.32,1.1
abc123,Apple M4,Metal,CPU,qwen2 0.5B,4370000000,8000000000,2048,512,8,0,0,1,0,128,42.87,0.5
CSV
      ;;
  esac
  exit 0
fi
exit 1
"""
    stub = _write_stub(tmp_path / "llama-bench-mixed", stub_script)
    q4_path = tmp_path / "model-q4.gguf"
    other_path = tmp_path / "model-other.gguf"
    q4_path.write_bytes(b"0" * 1024)
    other_path.write_bytes(b"0" * 1024)

    try:
        main(
            [
                "bench",
                "--llama-bench-bin", stub,
                "--gguf", f"Q4_K_M={q4_path}",
                "--gguf", f"Q3_K_M={other_path}",
                "--rounds", "1",
            ]
        )
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "same base model" in str(e)


def test_cli_bench_with_quality_eval(tmp_path, capsys):
    bench_stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    perplexity_stub = _write_stub(tmp_path / "llama-perplexity", LLAMA_PERPLEXITY_STUB)
    gguf_path = tmp_path / "model-q4.gguf"
    gguf_path.write_bytes(b"0" * (10 * 1024 * 1024))
    dataset = tmp_path / "wiki.test.raw"
    dataset.write_text("some text")
    out_csv = tmp_path / "results.csv"

    rc = main(
        [
            "bench",
            "--llama-bench-bin", bench_stub,
            "--gguf", f"Q4_K_M={gguf_path}",
            "--llama-perplexity-bin", perplexity_stub,
            "--perplexity-dataset", str(dataset),
            "--output", str(out_csv),
            "--rounds", "1",
            "--skip-hash",
        ]
    )
    assert rc == 0
    csv_text = out_csv.read_text()
    assert "perplexity" in csv_text
    assert "6.1234" in csv_text
    assert "perplexity_error" in csv_text


def test_cli_bench_quality_eval_requires_both_flags(tmp_path):
    bench_stub = _write_stub(tmp_path / "llama-bench", LLAMA_BENCH_STUB)
    gguf_path = tmp_path / "model-q4.gguf"
    gguf_path.write_bytes(b"0")

    try:
        main(
            [
                "bench",
                "--llama-bench-bin", bench_stub,
                "--gguf", f"Q4_K_M={gguf_path}",
                "--llama-perplexity-bin", "/some/path",
            ]
        )
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "quality evaluation" in str(e)


def test_cli_quantize_rejects_iq_without_imatrix(tmp_path):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")

    try:
        main(
            [
                "quantize",
                "--llama-quantize-bin", stub,
                "--input", str(input_gguf),
                "--output-dir", str(tmp_path / "out"),
                "IQ2_XXS",
            ]
        )
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "imatrix" in str(e)


def test_cli_quantize_with_imatrix_succeeds(tmp_path, capsys):
    stub = _write_stub(tmp_path / "llama-quantize", LLAMA_QUANTIZE_STUB)
    input_gguf = tmp_path / "model-f16.gguf"
    input_gguf.write_bytes(b"0")
    calibration = tmp_path / "calibration.dat"
    calibration.write_bytes(b"0")

    rc = main(
        [
            "quantize",
            "--llama-quantize-bin", stub,
            "--input", str(input_gguf),
            "--output-dir", str(tmp_path / "out"),
            "--imatrix", str(calibration),
            "IQ2_XXS",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "model-f16-IQ2_XXS.gguf").exists()


def test_cli_recommend_filters_csv(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "format,file_size_mb,gen_tokens_per_second\n"
        "Q4_K_M,4000,40\n"
        "Q8_0,8000,60\n"
        "Q2_K,2000,25\n"
    )
    rc = main(["recommend", "--csv", str(csv_path), "--max-size-gb", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Q8_0" not in out
    assert "Q4_K_M" in out
    assert "Q2_K" in out


def test_cli_recommend_no_match(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("format,file_size_mb,gen_tokens_per_second\nQ8_0,8000,60\n")
    rc = main(["recommend", "--csv", str(csv_path), "--max-size-gb", "1"])
    assert rc == 1
