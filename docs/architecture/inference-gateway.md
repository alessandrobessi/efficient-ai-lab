# Inference Gateway Architecture (Week 7)

## Request flow

```text
CLIENT
   │  POST /v1/generate  {"prompt": ..., "max_tokens": ..., "temperature": ...}
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ GO INFERENCE GATEWAY  (services/inference-gateway/)              │
│                                                                    │
│  middleware chain (outermost → innermost):                        │
│    RequestID → Logging → Recover → Metrics → ServeMux             │
│                                                                    │
│  routes:                                                          │
│    GET  /health    → liveness (process up, no upstream check)     │
│    GET  /ready      → readiness (checks llama-server /health)     │
│    GET  /metrics    → Prometheus exposition                       │
│    POST /v1/generate → validate → bounded-timeout call → map errs │
└─────────────────────────────────────────────────────────────────┘
   │  POST /completion  {"prompt": ..., "n_predict": ..., "temperature": ...}
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLAMA.CPP SERVER (llama-server, vendor/llama.cpp)                 │
│   already driven the same way by Week 5-6's Python evaluation     │
│   pipeline (evaluation/runners/llama_server_runner.py) — this     │
│   gateway is a second, production-shaped client of the same       │
│   server, not a replacement for it.                                │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
SMALL LANGUAGE MODEL  (GGUF, loaded once at llama-server startup)
```

## Why a gateway in front of llama-server at all

llama-server already speaks HTTP — a client *could* call it directly, and Weeks 5-6
did exactly that from Python. The gateway exists for everything a raw model server
doesn't provide on its own and a production deployment needs:

- **A stable, small API surface** (`/v1/generate`) independent of llama-server's own
  API, which is free to change across llama.cpp versions.
- **Validation and bounded timeouts** at the edge, before a bad or slow request ever
  reaches the model.
- **Observability** (structured logs, Prometheus metrics, request IDs) that
  llama-server itself doesn't expose in the shape this program's Week 8 load testing
  and Week 9 Kubernetes deployment need.
- **A place to put cross-cutting concerns** (rate limiting, auth, request queuing)
  later, without touching llama-server or the model itself.

## Health vs. readiness

The gateway deliberately answers two different questions with two different
endpoints, because they call for two different reactions:

| endpoint | question | if it fails |
|---|---|---|
| `GET /health` | is the gateway process itself alive? | restart the container |
| `GET /ready` | can the gateway actually serve a generate request right now? | stop routing traffic, but don't restart |

A transient llama-server hiccup (e.g. mid-model-reload) should make `/ready` fail
without touching `/health` — restarting the gateway wouldn't fix an unavailable
upstream, it would just add churn. This maps directly onto Kubernetes'
`livenessProbe` / `readinessProbe` distinction, which Week 9 wires up.

## Error mapping

The `internal/llamacpp` client returns one of three sentinel errors, which
`internal/handler` maps to distinct HTTP statuses and `gateway_errors_total{type=...}`
metric labels — see `services/inference-gateway/README.md`'s API table for the full
mapping. The intent: a dashboard built on `/metrics` alone should be able to tell "the
model is slow" (`upstream_timeout`) apart from "the model is down"
(`upstream_unavailable`) apart from "we sent it something it choked on"
(`upstream_bad_response`), without reading a single log line.

## What's deliberately not here yet

- **No request queueing or concurrency limiting** — llama-server itself serializes
  generation (one model, one CPU budget); Week 8's load generator is what actually
  characterizes what happens under concurrent load, and any queueing strategy should
  be informed by those results, not guessed at now.
- **No auth** — out of scope per FULL-ROADMAP.md's scope-control rules (this isn't a
  product); the gateway is meant to sit behind whatever auth layer a real deployment
  already has.
- **No caching** — this program's model is a single deterministic small model, not
  a multi-tenant service where response caching would pay for itself; revisit only if
  Week 8's load testing shows it would matter.
