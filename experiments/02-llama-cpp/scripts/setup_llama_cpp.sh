#!/usr/bin/env bash
# Clone and build llama.cpp locally, CPU-only.
#
# Metal (GPU) is explicitly disabled so this stays a genuine CPU-only inference
# engine, comparable to Week 1's forced-CPU Python baseline. Apple's Accelerate
# framework (vDSP/BLAS) is kept enabled — it's still a CPU code path, just a
# faster one than naive loops, and llama.cpp would use it in any real CPU
# deployment on Apple hardware.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor/llama.cpp"

if [ ! -d "${VENDOR_DIR}" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${VENDOR_DIR}"
else
  echo "llama.cpp already cloned at ${VENDOR_DIR}, skipping clone."
fi

cmake -S "${VENDOR_DIR}" -B "${VENDOR_DIR}/build" \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=ON \
  -DGGML_ACCELERATE=ON \
  -DCMAKE_BUILD_TYPE=Release

NPROC="$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)"
cmake --build "${VENDOR_DIR}/build" --config Release -j "${NPROC}" \
  --target llama-cli llama-server llama-bench

echo
echo "Built binaries:"
ls -la "${VENDOR_DIR}/build/bin/llama-cli" "${VENDOR_DIR}/build/bin/llama-server" "${VENDOR_DIR}/build/bin/llama-bench"
"${VENDOR_DIR}/build/bin/llama-cli" --version
