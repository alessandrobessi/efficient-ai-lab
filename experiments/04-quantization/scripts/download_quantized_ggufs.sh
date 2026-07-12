#!/usr/bin/env bash
# Download the quantization levels used in Experiments 4.1-4.4, in addition to the
# F16 GGUF already downloaded in Week 2 (which serves as the unquantized ceiling
# reference here). Levels chosen to match FULL-ROADMAP.md's suggested "Q8, Q6, Q5,
# Q4, Q3" using the _K variants (generally better quality per bit than the older
# plain Q4_0/Q5_0 formats also published in the same repo).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST_DIR="${REPO_ROOT}/models/gguf"
mkdir -p "${DEST_DIR}"

FILES=(
  qwen2.5-1.5b-instruct-q8_0.gguf
  qwen2.5-1.5b-instruct-q6_k.gguf
  qwen2.5-1.5b-instruct-q5_k_m.gguf
  qwen2.5-1.5b-instruct-q4_k_m.gguf
  qwen2.5-1.5b-instruct-q3_k_m.gguf
)

cd "${REPO_ROOT}"
for f in "${FILES[@]}"; do
  uv run hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF "${f}" --local-dir "${DEST_DIR}"
done

echo
ls -lah "${DEST_DIR}"
