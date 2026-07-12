#!/usr/bin/env bash
# Download the 5 models compared this week, all at Q4_K_M — the format Week 4/5
# identified as this project's speed-Pareto-optimal quantization level, held
# constant here so the only thing varying between models is the model itself
# (FULL-ROADMAP.md's "keep benchmark methodology constant" control).
#
# Qwen2.5-1.5B-Instruct-Q4_K_M is already on disk from Week 4/5 and is not
# re-downloaded here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST_DIR="${REPO_ROOT}/models/gguf"
mkdir -p "${DEST_DIR}"

cd "${REPO_ROOT}"

uv run hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir "${DEST_DIR}"
uv run hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir "${DEST_DIR}"
uv run hf download bartowski/gemma-2-2b-it-GGUF gemma-2-2b-it-Q4_K_M.gguf --local-dir "${DEST_DIR}"
uv run hf download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir "${DEST_DIR}"

echo
ls -lah "${DEST_DIR}"
