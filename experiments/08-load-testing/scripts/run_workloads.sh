#!/usr/bin/env bash
# Runs FULL-ROADMAP.md's Week 8 Workloads A-D against an already-running
# gateway (POST /v1/generate on :8080) and llama-server (--metrics, -np 1,
# on :8799 — see this week's README for exactly how they were started).
# Requires scripts/build.sh to have been run first.
#
# Workload A — Single User: 1 sequential client.
# Workload B — Small Team: 5 concurrent clients.
# Workload C — Medium Load: 20 concurrent clients.
# Workload D — Saturation: a concurrency sweep (1/2/5/10/20/40/80) until
#   throughput plateaus and/or error rate rises — "collapse" is a result to
#   observe from the data, not a threshold assumed up front.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/08-load-testing/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/08-load-testing"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"
MAX_TOKENS=64
DURATION=60s

mkdir -p "${RAW_DIR}"

run() {
  local label="$1"; shift
  echo "=== ${label} ==="
  "${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens "${MAX_TOKENS}" \
    -output "${RAW_DIR}/${label}.jsonl" -label "${label}" "$@"
  echo
}

run workload_a -mode closed-loop -concurrency 1  -duration "${DURATION}"
run workload_b -mode closed-loop -concurrency 5  -duration "${DURATION}"
run workload_c -mode closed-loop -concurrency 20 -duration "${DURATION}"

for c in 1 2 5 10 20 40 80; do
  run "workload_d_c${c}" -mode closed-loop -concurrency "${c}" -duration 20s
done

echo "All workloads complete. Raw results -> ${RAW_DIR}/"
