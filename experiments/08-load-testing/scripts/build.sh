#!/usr/bin/env bash
# Build the inference-gateway and load-generator binaries used by this
# week's workload scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN_DIR="${REPO_ROOT}/experiments/08-load-testing/bin"
mkdir -p "${BIN_DIR}"

go -C "${REPO_ROOT}/services/inference-gateway" build -o "${BIN_DIR}/inference-gateway" .
go -C "${REPO_ROOT}/services/load-generator" build -o "${BIN_DIR}/load-generator" .

echo "Built -> ${BIN_DIR}/inference-gateway, ${BIN_DIR}/load-generator"
