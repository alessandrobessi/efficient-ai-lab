#!/usr/bin/env bash
# Experiment 9.2 — Memory Limits: sweep the llama-server container's memory
# request/limit (set equal, Guaranteed QoS) down toward and below the
# model's known RSS footprint (~2.1-2.8GB for Qwen2.5-1.5B-Instruct Q4_K_M,
# per Week 4/9.2's own kubectl top reading), and record what actually
# happens — OOMKilled, CrashLoopBackOff, or a healthy pod — rather than
# assuming a single failure mode.
#
# Unlike exp_9_1 this does NOT fail the script if rollout doesn't succeed —
# a failed rollout (OOMKilled pod stuck restarting) is the expected,
# interesting result at low memory limits, not a bug.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
OUT="${RAW_DIR}/exp9_2_memory_limits.jsonl"
mkdir -p "${RAW_DIR}"
: > "${OUT}"

for mem in 4Gi 3Gi 2.5Gi 2Gi 1.5Gi 1Gi; do
  echo "=== memory limit: ${mem} ==="
  kubectl patch deployment slm-gateway --type=json -p="[
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/requests/memory\", \"value\": \"${mem}\"},
    {\"op\": \"replace\", \"path\": \"/spec/template/spec/containers/0/resources/limits/memory\", \"value\": \"${mem}\"}
  ]" > /dev/null

  # Give the new pod up to 60s to become Ready; don't abort the sweep if it
  # never does (that's the point of this experiment at low memory limits).
  kubectl rollout status deployment/slm-gateway --timeout=60s > /dev/null 2>&1
  rollout_ok=$?
  sleep 5 # let a crash loop actually manifest (first OOM kill can take a few seconds post-"Running")

  pod=$(kubectl get pods -l app=slm-gateway -o jsonpath='{.items[0].metadata.name}')
  ready=$(kubectl get pod "${pod}" -o jsonpath='{.status.containerStatuses[?(@.name=="llama-server")].ready}')
  restarts=$(kubectl get pod "${pod}" -o jsonpath='{.status.containerStatuses[?(@.name=="llama-server")].restartCount}')
  last_reason=$(kubectl get pod "${pod}" -o jsonpath='{.status.containerStatuses[?(@.name=="llama-server")].lastState.terminated.reason}')
  health_status="unknown"
  if [ "${ready}" = "true" ]; then
    health_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/health || echo "unreachable")
  fi

  echo "  pod=${pod} rollout_ok=${rollout_ok} ready=${ready} restarts=${restarts} last_reason=${last_reason} health=${health_status}"
  python3 -c "
import json
print(json.dumps({
    'memory_limit': '${mem}',
    'rollout_ok': ${rollout_ok} == 0,
    'ready': '${ready}' == 'true',
    'restart_count': int('${restarts}' or 0),
    'last_terminated_reason': '${last_reason}',
    'health_status_code': '${health_status}',
}))
" >> "${OUT}"
  echo
done

echo "Experiment 9.2 complete -> ${OUT}"
