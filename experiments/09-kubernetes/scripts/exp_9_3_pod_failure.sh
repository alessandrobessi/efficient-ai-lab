#!/usr/bin/env bash
# Experiment 9.3 — Pod Failure: start a sustained load test, delete an
# inference pod mid-traffic, and measure the error spike / recovery time /
# request loss from the load generator's own per-request timestamps plus
# direct kubectl observation of when a replacement pod becomes Ready.
#
# Usage: exp_9_3_pod_failure.sh <replicas> <label>
#   e.g. exp_9_3_pod_failure.sh 1 exp9_3_1replica
#        exp_9_3_pod_failure.sh 2 exp9_3_2replica
set -euo pipefail

REPLICAS="${1:?usage: exp_9_3_pod_failure.sh <replicas> <label>}"
LABEL="${2:?usage: exp_9_3_pod_failure.sh <replicas> <label>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="${REPO_ROOT}/experiments/09-kubernetes/bin/load-generator"
RAW_DIR="${REPO_ROOT}/results/raw/09-kubernetes"
URL="http://127.0.0.1:8080/v1/generate"
PROMPTS="${REPO_ROOT}/evaluation/datasets/v1.jsonl"
mkdir -p "${RAW_DIR}"

now_iso() { python3 -c "import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())"; }

echo "=== scaling to ${REPLICAS} replica(s) ==="
kubectl scale deployment/slm-gateway --replicas="${REPLICAS}"
kubectl rollout status deployment/slm-gateway --timeout=120s
sleep 5

echo "=== starting 60s load test in background (concurrency=5) ==="
"${BIN}" -url "${URL}" -prompts "${PROMPTS}" -max-tokens 64 \
  -mode closed-loop -concurrency 5 -duration 60s \
  -output "${RAW_DIR}/${LABEL}.jsonl" -label "${LABEL}" &
LG_PID=$!

sleep 20
VICTIM=$(kubectl get pods -l app=slm-gateway -o jsonpath='{.items[0].metadata.name}')
DELETE_TS=$(now_iso)
echo "=== deleting pod ${VICTIM} at ${DELETE_TS} ==="
kubectl delete pod "${VICTIM}" --wait=false

echo "=== polling for a genuinely NEW (name != ${VICTIM}) Ready pod, at full replica capacity ==="
while true; do
  # Count only pods whose name differs from the deleted victim and whose
  # gateway container reports ready=true — the victim can briefly still
  # report ready=true while Terminating, which would otherwise false-positive
  # a near-zero "recovery time."
  new_ready_count=$(kubectl get pods -l app=slm-gateway -o json | python3 -c "
import json, sys
data = json.load(sys.stdin)
n = 0
for pod in data['items']:
    if pod['metadata']['name'] == '${VICTIM}':
        continue
    for cs in pod.get('status', {}).get('containerStatuses', []):
        if cs['name'] == 'inference-gateway' and cs.get('ready'):
            n += 1
print(n)
")
  if [ "${new_ready_count}" -ge "${REPLICAS}" ]; then
    break
  fi
  sleep 0.2
done
RECOVERED_TS=$(now_iso)
echo "=== full Ready capacity (${REPLICAS}/${REPLICAS}) restored at ${RECOVERED_TS}, via a new pod ==="

python3 -c "
import json
print(json.dumps({
    'label': '${LABEL}',
    'replicas': ${REPLICAS},
    'victim_pod': '${VICTIM}',
    'delete_timestamp': '${DELETE_TS}',
    'recovered_timestamp': '${RECOVERED_TS}',
}))
" > "${RAW_DIR}/${LABEL}_deletion_event.json"

wait "${LG_PID}"
echo "Experiment 9.3 (${LABEL}) complete."
