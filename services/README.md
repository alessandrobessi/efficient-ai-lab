# Services

Go services built in Phase III (Weeks 7–9).

- [`inference-gateway/`](inference-gateway/README.md) — **built in Week 7.** HTTP
  gateway in front of the llama.cpp server: request validation, timeouts, structured
  logging, Prometheus metrics, graceful shutdown. See its own README for the API,
  configuration, and running instructions, and
  [`docs/architecture/inference-gateway.md`](../docs/architecture/inference-gateway.md)
  for the design.
- [`load-generator/`](load-generator/README.md) — **built in Week 8.** Hand-rolled
  concurrent load generator (goroutines/channels/tickers, no third-party load-testing
  library) — closed-loop and open-loop dispatch modes, exact percentile computation,
  a concrete coordinated-omission demonstration. See its own README for usage, and
  [`experiments/08-load-testing/README.md`](../experiments/08-load-testing/README.md)
  for the workload results.
