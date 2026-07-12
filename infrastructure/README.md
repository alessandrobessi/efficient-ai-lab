# Infrastructure

Deployment and observability configuration, introduced in Phase III (Weeks 7–9).

- [`docker/`](docker/) — **built in Week 7-8.** `inference-gateway.Dockerfile`
  (multi-stage, distroless final image, 17.6MB) and `docker-compose.yml` (wires the
  gateway to an `ghcr.io/ggml-org/llama.cpp:server` container for a one-command local
  stack) from Week 7; `observability-compose.yml` (Prometheus + Grafana) from Week 8.
  See `services/inference-gateway/README.md` and
  `experiments/08-load-testing/README.md` for run instructions.
- `kubernetes/` — Deployment/Service/ConfigMap manifests, resource requests/limits,
  probes, used for the Week 9 orchestration and failure experiments. Not started —
  first used in Week 9.
- [`prometheus/`](prometheus/) — **built in Week 8.** Scrape config for the gateway,
  llama-server's own `--metrics` endpoint, and `node_exporter` (host CPU/RAM).
- [`grafana/`](grafana/) — **built in Week 8.** Provisioned datasource + one dashboard
  covering request rate, latency percentiles, errors, active/processing/deferred
  requests, host CPU/RAM/load, and model throughput.
