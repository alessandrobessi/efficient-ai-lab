# Services

Go services built in Phase III (Weeks 7–9).

- [`inference-gateway/`](inference-gateway/README.md) — **built in Week 7.** HTTP
  gateway in front of the llama.cpp server: request validation, timeouts, structured
  logging, Prometheus metrics, graceful shutdown. See its own README for the API,
  configuration, and running instructions, and
  [`docs/architecture/inference-gateway.md`](../docs/architecture/inference-gateway.md)
  for the design.
- `load-generator/` — hand-rolled concurrent load generator (goroutines/channels) used
  to drive the gateway and measure latency percentiles under increasing concurrency.
  Not started — first used in Week 8.
