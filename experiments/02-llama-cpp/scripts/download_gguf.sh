#!/usr/bin/env bash
# Download the GGUF equivalent of the Week 1 model.
#
# F16 (no quantization) is used deliberately, not one of the quantized formats
# (Q8_0, Q4_K_M, ...) also published in this repo — Week 2 isolates the engine
# variable (Python/PyTorch vs llama.cpp). Mixing in a quantization change here
# would confound that comparison; quantization is the explicit subject of
# Weeks 4-5.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST_DIR="${REPO_ROOT}/models/gguf"
mkdir -p "${DEST_DIR}"

cd "${REPO_ROOT}"
uv run hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-fp16.gguf \
  --local-dir "${DEST_DIR}"

echo
ls -lah "${DEST_DIR}"
