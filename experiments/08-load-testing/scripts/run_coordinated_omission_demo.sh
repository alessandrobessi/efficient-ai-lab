#!/usr/bin/env bash
# A dedicated demonstration of coordinated omission (FULL-ROADMAP.md's
# "learn ... coordinated omission conceptually"), separate from the official
# Workloads A-D: open-loop mode at a target rate the backend can't sustain,
# with a deliberately small sender pool so dispatch itself backs up. Compare
# the resulting summary's latency_p* (naive) against corrected_latency_p*
# (includes queue delay) — see internal/worker/openloop.go's doc comment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/08-load-testing/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/08-load-testing"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"

mkdir -p "${RAW_DIR}"

"${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
  -mode open-loop -rps 3 -concurrency 2 -duration 45s \
  -output "${RAW_DIR}/coordinated_omission_demo.jsonl" -label coordinated_omission_demo

echo "Coordinated omission demo complete -> ${RAW_DIR}/coordinated_omission_demo.jsonl"
