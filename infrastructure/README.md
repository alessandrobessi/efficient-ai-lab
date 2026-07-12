# Infrastructure

Deployment and observability configuration, introduced in Phase III (Weeks 7–9).

- [`docker/`](docker/) — **built in Week 7.** `inference-gateway.Dockerfile`
  (multi-stage, distroless final image, 17.6MB) and `docker-compose.yml` (wires the
  gateway to an `ghcr.io/ggml-org/llama.cpp:server` container for a one-command local
  stack). See `services/inference-gateway/README.md` for run instructions.
- `kubernetes/` — Deployment/Service/ConfigMap manifests, resource requests/limits,
  probes, used for the Week 9 orchestration and failure experiments. Not started —
  first used in Week 9.
- `prometheus/` — scrape configuration for the inference gateway and cluster. Not
  started — first used in Week 8/9.
- `grafana/` — dashboards for request rate, latency, errors, CPU, RAM, throughput. Not
  started — first used in Week 8/9.
