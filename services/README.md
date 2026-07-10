# Services

Go services built in Phase III (Weeks 7–9).

- `inference-gateway/` — HTTP gateway in front of the llama.cpp server: request
  validation, timeouts, structured logging, Prometheus metrics, graceful shutdown.
- `load-generator/` — hand-rolled concurrent load generator (goroutines/channels) used
  to drive the gateway and measure latency percentiles under increasing concurrency.

Not started — first used in Week 7.
