#!/usr/bin/env bash
# Creates the local kind (Kubernetes-in-Docker) cluster used for Week 9.
#
# Two things a static kind config can't express portably, so this script
# generates one instead:
#   - extraMounts needs an absolute host path to models/gguf/ (varies per
#     checkout) so pods can mount the already-downloaded GGUF model files
#     without baking multi-GB weights into a container image.
#   - extraPortMappings exposes the gateway's NodePort (30080) on the host at
#     :8080, so the existing Week 8 load generator can drive traffic at the
#     cluster exactly like it drove the native gateway, no code changes needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models/gguf"
CLUSTER_NAME="efficient-ai-lab"

if [ ! -d "${MODELS_DIR}" ]; then
  echo "error: ${MODELS_DIR} not found — download a GGUF model first (see models/README.md)" >&2
  exit 1
fi

cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config -
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: ${MODELS_DIR}
        containerPath: /models
        readOnly: true
    extraPortMappings:
      - containerPort: 30080
        hostPort: 8080
        protocol: TCP
EOF

echo
echo "Cluster ready. Context: kind-${CLUSTER_NAME}"
echo "Model directory mounted read-only at /models inside the node."
echo "Gateway will be reachable at http://127.0.0.1:8080 once deployed (Service NodePort 30080)."
