#!/usr/bin/env bash
# Experiment 9.1 — CPU Limits: sweep the llama-server container's CPU
# request/limit (set equal, for Guaranteed QoS / a strictly enforced cgroup
# CPU quota) and measure single-user latency/decode-speed degradation via
# the Week 8 load generator at concurrency=1 (isolates CPU throttling from
# any queueing effect).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/09-kubernetes/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"

mkdir -p "${RAW_DIR}"

for cpu in 250m 500m 1000m 2000m 4000m; do
  echo "=== CPU limit: ${cpu} ==="
  kubectl patch deployment slm-gateway --type=json -p="[
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/requests/cpu\", \"value\": \"${cpu}\"},
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/limits/cpu\", \"value\": \"${cpu}\"}
  ]"
  kubectl rollout status deployment/slm-gateway --timeout=90s
  sleep 3 # let the new pod's readiness probe stabilize past its first pass

  "${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
    -mode closed-loop -concurrency 1 -duration 20s \
    -output "${RAW_DIR}/exp9_1_cpu_${cpu}.jsonl" -label "exp9_1_cpu_${cpu}"
  echo
done

echo "Experiment 9.1 complete."
