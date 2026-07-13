# llmpace — Development Roadmap

**Status: planned.** This document describes the architecture and milestones
for a tool that does not exist yet. No code, `go.mod`, or scaffolding has been
created — this is what a future implementation session should build against.

## Why this exists

[Week 8](../../experiments/08-load-testing/README.md) of the efficient-ai-lab
program built a load generator for the inference gateway
(`services/load-generator/`) and, in doing so, ran directly into two facts
that generic load testers don't handle:

1. LLM responses stream. A single "request latency" number conflates
   time-to-first-token (how long until the model starts responding — driven
   by prefill/queueing) with inter-token latency (how fast tokens arrive once
   generation starts — driven by decode speed). These have different causes
   and different user-facing meaning; collapsing them into one number throws
   away the more actionable half.
2. Measuring latency from **actual dispatch time** instead of **scheduled
   dispatch time** hides queueing delay — exactly the signal that says a
   server is overloaded. This is coordinated omission. Week 8 measured a
   20-36x gap between naive and coordinated-omission-corrected p99 latency
   under saturation using the same request trace.

Both are architectural properties of the load generator, not configuration
flags someone can be expected to discover. `llmpace` bakes in the correct
default (open-loop dispatch, streaming-aware measurement, both latency views
always shown) rather than requiring the operator to know to ask for it.

## Reuse from `services/load-generator/`

| Component | Path | Disposition |
|---|---|---|
| Percentile engine | `internal/stats/stats.go` | Reusable near-as-is |
| CO-correction dispatch | `internal/worker/openloop.go`, `closedloop.go` | Reusable behind a new interface — currently coupled to concrete `client.Client`/`prompts.Source` types |
| Prompt sourcing | `internal/prompts/prompts.go` | Reusable pattern |
| Report formatting | `internal/report/report.go` | Reusable pattern |
| Behavioral test model | `internal/worker/worker_test.go` (`TestRunOpenLoop_QueueDelayGrowsUnderSaturation`) | Port as-is — proves CO-correction via an artificially slow mock backend |
| HTTP client | `internal/client/client.go` | **Not reusable.** Hardcoded to one gateway's request schema; a single `io.ReadAll` + one JSON unmarshal, zero streaming support |
| Config | `internal/config/config.go` | Flags-only, no config-file support — extend, don't replace |

No streaming/SSE consumption code exists anywhere in efficient-ai-lab today.
That is the single biggest net-new build here — everything else is porting or
extending proven code.

## Architecture

Independent Go module (`go 1.23`, stdlib-first, following
[ADR 0001](../../docs/decisions/0001-go-for-inference-gateway.md)'s reasoning:
single static binary, goroutines as the natural concurrency idiom, no
dependency without a structural reason).

```
projects/llmpace/
├── go.mod                        module github.com/alessandrobessi/efficient-ai-lab/projects/llmpace
├── main.go
├── internal/
│   ├── adapter/                  backend-specific request/response handling
│   │   ├── adapter.go              Adapter interface: BuildRequest, Stream
│   │   ├── llamacpp.go             /completion, SSE
│   │   ├── openai.go               /v1/chat/completions, SSE
│   │   └── ollama.go               /api/generate, NDJSON
│   ├── dispatch/                 (evolves internal/worker/)
│   │   ├── openloop.go             default mode
│   │   └── closedloop.go           opt-in, documented CO-correction limitation
│   ├── stats/                    (ported from internal/stats/)
│   ├── prompts/                  (ported from internal/prompts/)
│   ├── config/                   flags (stdlib) + optional YAML test-plan file
│   └── report/                   human table, JSON, CSV, Prometheus exposition
└── ROADMAP.md, README.md, LICENSE
```

**`Adapter` interface** — the core new abstraction:

```go
type Adapter interface {
    BuildRequest(prompt string) (*http.Request, error)
    // Stream reads the response body and invokes onToken for each token as it
    // arrives, so no adapter buffers a full response. Returns TTFT and the
    // per-token arrival timestamps needed for inter-token latency.
    Stream(resp *http.Response, onToken func(t time.Time)) (ttft time.Duration, err error)
}
```

Each adapter owns its own wire format (SSE `data:` lines for llama.cpp/OpenAI,
newline-delimited JSON for Ollama) behind one `bufio.Scanner`-based line
reader shared across implementations.

**Dispatch** — open-loop is the default mode. Closed-loop (N clients, each
waiting for a response before re-issuing) is kept as an opt-in for parity with
the old tool, but is documented as CO-unsafe by construction: at concurrency N
with no queueing, naive and corrected latency are numerically identical, so
there is nothing for the correction to reveal. The `ScheduledAt`/`SentAt`/
`DoneAt` triad from `openloop.go` is preserved because it's exactly what makes
the correction possible.

**Measurement** — TTFT is time-of-first-streamed-token minus `SentAt`.
Inter-token latency is reported as a full distribution (p50/95/99), not a
mean, because tail ITL is exactly what coordinated omission hides. Every
report shows naive and corrected percentiles side by side, with a warning
when they diverge by more than 2x, rather than requiring a flag to opt in.

**Config** — CLI flags (stdlib `flag`) cover single-run usage. An optional
YAML test-plan file adds multi-stage/multi-backend runs — this is a
deliberate stdlib exception, justified the same way ADR 0001 justified
`prometheus/client_golang`: added only because the roadmap has an explicit
requirement (multi-stage config) that the standard library has no answer for.

```yaml
# example test-plan.yaml (M3+)
stages:
  - name: warmup
    backend: llamacpp
    url: http://localhost:8080
    duration: 30s
    rate: 5   # requests/sec, open-loop
  - name: soak
    backend: llamacpp
    url: http://localhost:8080
    duration: 10m
    rate: 20
```

**Reporting** — human-readable table (default), JSON (raw JSONL trace +
summary), CSV, and Prometheus exposition. Prometheus output targets two
consumers: a live `/metrics` endpoint that plugs directly into the existing
`infrastructure/prometheus/prometheus.yml` stack for real-time dashboards
during a run, and a static textfile-collector-format dump for CI/archival.

**Memory, for long or high-QPS runs** — the current tool holds the full result
set in memory and uses small buffered channels, which is fine for a 60-second
local benchmark and not fine for a soak test. `llmpace` streams results to the
JSONL writer incrementally instead of buffering them, and computes percentiles
over a bounded reservoir sample once total request count exceeds a
configurable threshold (falling back to exact percentiles below it) — trading
a small, controlled bias for bounded memory on multi-hour runs.

## Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0** | Walking skeleton: port stats/dispatch/prompts, non-streaming, llama.cpp only | Reproduces Week 8's own non-streaming numbers within noise against the same backend |
| **M1** | Streaming + TTFT/ITL for llama.cpp, open-loop-by-default | Mock SSE server with injected inter-chunk delays; measured TTFT/ITL land within tolerance of injected values |
| **M2** | Add OpenAI-compatible and Ollama adapters | Each adapter passes the same `httptest` conformance suite (correct TTFT, correct token count, correct stream termination handling) |
| **M3** | Config file, CSV/Prometheus output, bounded-memory soak validation | A multi-hour synthetic soak run holds bounded memory (measured, not assumed) and produces valid Prometheus exposition scraped by a local Prometheus instance |
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` with cross-compiled binaries | `go test ./...` green in CI on a fresh clone; release binaries produced for darwin/linux, amd64/arm64 |

## Testing strategy

- **Behavioral, not just compiling**: port
  `TestRunOpenLoop_QueueDelayGrowsUnderSaturation` as the model for proving
  the CO-correction mechanism against an artificially slow mock backend,
  rather than only asserting the code compiles and runs.
- **Adapter tests** via `httptest`, using both hand-written synthetic
  SSE/NDJSON fixtures and recorded-real traffic captured from an actual
  llama.cpp/Ollama server, so wire-format edge cases (partial chunks, keep-alive
  lines, trailing whitespace) are covered.
- **No live-server tier in CI.** A real end-to-end run against a live
  inference server is a documented manual pre-release step, not something CI
  depends on.

## Explicitly deferred (documented limitation, not silently omitted)

Gil Tene-style synthetic sample backfill for closed-loop coordinated-omission
correction is not implemented in v0.1.0 — open-loop-by-default already covers
the common case this tool is built for, and closed-loop is kept only for
parity with the old tool's mode, with its CO-unsafety documented rather than
worked around.

## License

MIT — see [`../../LICENSE`](../../LICENSE); each project carries its own copy
of the same license text.
