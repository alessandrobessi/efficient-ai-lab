# Architecture Notes

Diagrams and write-ups of system architecture as it evolves: the Week 7 Go inference
gateway, the Week 9 Kubernetes topology, and how they connect to llama.cpp and the SLM
underneath.

- [Inference Gateway (Week 7)](inference-gateway.md) — request flow, health vs.
  readiness, error mapping, and what's deliberately not built yet.
- [Load Testing & Observability (Week 8)](load-testing.md) — topology, why three
  separate metrics sources, why `-np 1`, closed-loop vs. open-loop dispatch.
- [Kubernetes Topology (Week 9)](kubernetes.md) — pod design (gateway + llama-server
  sidecar), why a 2Gi memory limit, why horizontal scaling didn't show clean linear
  throughput.
