#!/usr/bin/env bash
# Experiment 9.4 — Load Saturation: the same concurrency sweep as Week 8
# (1/2/5/10/20/40/80), now against the Kubernetes-deployed gateway (1
# replica, 2000m CPU / 2Gi memory), to see whether containerization itself
# changes the collapse curve — plus kubectl top snapshots per level for
# CPU/memory, since that's not something the load generator can see.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/09-kubernetes/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"
TOP_LOG="${RAW_DIR}/exp9_4_kubectl_top.jsonl"

mkdir -p "${RAW_DIR}"
: > "${TOP_LOG}"

for c in 1 2 5 10 20 40 80; do
  echo "=== concurrency ${c} ==="
  "${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
    -mode closed-loop -concurrency "${c}" -duration 20s \
    -output "${RAW_DIR}/exp9_4_c${c}.jsonl" -label "exp9_4_c${c}"

  pod=$(kubectl get pods -l app=slm-gateway -o jsonpath='{.items[0].metadata.name}')
  top_line=$(kubectl top pod "${pod}" --no-headers 2>/dev/null || echo "${pod} n/a n/a")
  cpu=$(echo "${top_line}" | awk '{print $2}')
  mem=$(echo "${top_line}" | awk '{print $3}')
  python3 -c "
import json
print(json.dumps({'concurrency': ${c}, 'pod': '${pod}', 'cpu_usage': '${cpu}', 'memory_usage': '${mem}'}))
" >> "${TOP_LOG}"
  echo
done

echo "Experiment 9.4 complete -> ${RAW_DIR}/exp9_4_c*.jsonl, ${TOP_LOG}"
