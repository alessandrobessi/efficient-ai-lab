#!/usr/bin/env bash
# Experiment 9.5b — follow-up to 9.5: the fixed-concurrency=20 test showed
# throughput did NOT scale with replica count (see README §9), even though
# kubectl top showed low CPU utilization and llama-server's own metrics
# showed evenly-distributed load across pods — ruling out routing/CPU
# throttling as the cause. The likely culprit: Experiment 9.4 already showed
# a *single* replica collapses by concurrency ~20, so testing every replica
# count at the same fixed total concurrency keeps each individual replica in
# its own collapse zone rather than its comfortable operating range.
#
# This reruns the same replicas 1/2/3, but with concurrency SCALED by
# replica count (20/replica) so each replica sees a load comparable to its
# own solo-capacity test — the methodologically fairer way to ask "does
# adding replicas raise the achievable throughput ceiling."
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/09-kubernetes/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"

mkdir -p "${RAW_DIR}"

for replicas in 1 2 3; do
  concurrency=$((replicas * 20))
  echo "=== replicas=${replicas}, concurrency=${concurrency} (20/replica) ==="
  kubectl scale deployment/slm-gateway --replicas="${replicas}"
  kubectl rollout status deployment/slm-gateway --timeout=120s
  sleep 5

  "${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
    -mode closed-loop -concurrency "${concurrency}" -duration 45s \
    -output "${RAW_DIR}/exp9_5b_replicas${replicas}.jsonl" -label "exp9_5b_replicas${replicas}"
  echo
done

echo "Experiment 9.5b complete -> ${RAW_DIR}/exp9_5b_replicas*.jsonl"
