#!/usr/bin/env bash
# Experiment 9.5 — Horizontal Scaling: for replicas in 1/2/3, scale the
# Deployment, time how long until all replicas are Ready ("startup cost" —
# includes scheduling + image presence + model load), record each pod's
# memory usage (duplication), then run the same fixed-concurrency load test
# (concurrency=20, Week 8's saturation-inducing level) against the
# Service — which load-balances across every replica — to see whether
# throughput actually scales, unlike Week 8's single -np 1 instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/09-kubernetes/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"
STARTUP_LOG="${RAW_DIR}/exp9_5_startup_cost.jsonl"
MEMORY_LOG="${RAW_DIR}/exp9_5_memory_per_replica.jsonl"

mkdir -p "${RAW_DIR}"
: > "${STARTUP_LOG}"
: > "${MEMORY_LOG}"

now_iso() { python3 -c "import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())"; }

for replicas in 1 2 3; do
  echo "=== scaling to ${replicas} replica(s) ==="
  # Force a clean scale-from-zero for a genuine startup-cost measurement,
  # rather than reusing already-warm pods from a previous replica count.
  kubectl scale deployment/slm-gateway --replicas=0
  kubectl wait --for=delete pod -l app=slm-gateway --timeout=60s 2>/dev/null || true

  START_TS=$(now_iso)
  kubectl scale deployment/slm-gateway --replicas="${replicas}"
  kubectl rollout status deployment/slm-gateway --timeout=120s
  READY_TS=$(now_iso)

  python3 -c "
import json
print(json.dumps({'replicas': ${replicas}, 'scale_start': '${START_TS}', 'all_ready': '${READY_TS}'}))
" >> "${STARTUP_LOG}"

  echo "--- waiting for metrics-server to report this pod's usage ---"
  for _ in $(seq 1 20); do
    if kubectl top pod -l app=slm-gateway --no-headers > /tmp/exp9_5_top.txt 2>/dev/null; then
      break
    fi
    sleep 3
  done
  echo "--- per-pod memory (replicas=${replicas}) ---"
  cat /tmp/exp9_5_top.txt
  python3 -c "
import json
rows = []
with open('/tmp/exp9_5_top.txt') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 3:
            rows.append({'replicas': ${replicas}, 'pod': parts[0], 'cpu_usage': parts[1], 'memory_usage': parts[2]})
with open('${MEMORY_LOG}', 'a') as out:
    for r in rows:
        out.write(json.dumps(r) + '\n')
"

  echo "--- load test at concurrency=20 (replicas=${replicas}) ---"
  "${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
    -mode closed-loop -concurrency 20 -duration 45s \
    -output "${RAW_DIR}/exp9_5_replicas${replicas}.jsonl" -label "exp9_5_replicas${replicas}"
  echo
done

echo "Experiment 9.5 complete -> ${RAW_DIR}/exp9_5_replicas*.jsonl, ${STARTUP_LOG}, ${MEMORY_LOG}"
