# llmpace — Development Roadmap

**Status: M0-M3 implemented.** The architecture below reflects what was
actually built (see `internal/`), not just a plan. M4 (CI, docs polish,
`CONTRIBUTING.md`, tagged release with cross-compiled binaries) remains.
Two deliberate simplifications versus the original plan: the YAML
multi-stage test-plan file (config is CLI-flags-only for now, single
stage per run) and a live `/metrics` Prometheus endpoint (a static
textfile-collector-format dump via `-prometheus-out` exists instead) —
both noted inline below where they diverge.

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

**`Adapter` interface** — the core new abstraction (as implemented, in
`internal/adapter/adapter.go`):

```go
type Adapter interface {
    Name() string
    BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error)
    // Stream reads resp.Body incrementally, invoking onToken with the
    // wall-clock time each token/chunk was observed — no adapter buffers a
    // full response. Returns the total number of tokens seen; TTFT and
    // inter-token gaps are derived by the caller (internal/dispatch.Sender)
    // from the onToken timestamps, not returned here.
    Stream(resp *http.Response, onToken func(t time.Time)) (tokens int, err error)
}
```

Each adapter owns its own wire format (SSE `data:` lines for llama.cpp/OpenAI,
newline-delimited JSON for Ollama) behind shared `bufio.Scanner`-based line
readers (`scanSSELines`/`scanNDJSONLines` in `internal/adapter/sse.go`).

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

**Config** — CLI flags (stdlib `flag`) cover single-run usage
(`internal/config/config.go`). **Deviation from the original plan:** the
optional YAML multi-stage test-plan file described below was not built —
single-run flags covered every M0-M3 exit criterion without it, so the extra
dependency wasn't yet justified per ADR 0001's own "only when structurally
necessary" standard. It remains a reasonable M4+ addition if multi-stage
sweeps (e.g. a concurrency ramp in one invocation) turn out to be worth it;
`-csv` already supports the common workaround (append one row per run,
scripted from the outside, into one comparison table).

```yaml
# example test-plan.yaml (not yet implemented — see deviation note above)
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

**Reporting** — human-readable table (default, `internal/report.PrintTable`),
JSON (raw JSONL trace + summary), CSV (`-csv`, appends one row per run for
comparison), and a Prometheus textfile-collector-format dump (`-prometheus-out`).
**Deviation from the original plan:** a live `/metrics` HTTP endpoint scraped
during the run was not built — only the static post-run dump. The static file
already integrates with `infrastructure/prometheus/prometheus.yml` via a
file-based scrape target; a live endpoint is only worth adding if watching a
dashboard *during* a long run (rather than after) becomes a real need.

**Memory, for long or high-QPS runs** — implemented as planned:
`internal/report.JSONLWriter` streams each result to disk as it arrives, and
`internal/stats.Accumulator` folds each result into a `Reservoir`
(`internal/stats/reservoir.go`, Algorithm R) instead of ever holding a full
result slice — bounded by `-max-samples` (default 100,000) regardless of how
long a run lasts. Below that many requests, the reservoir holds every value
(exact percentiles); above it, a uniform random sample (see
`TestReservoir_BoundedAboveCapacity`/`TestReservoir_ApproximatesMedian`).

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| **M0** | Walking skeleton: port stats/dispatch/prompts, non-streaming, llama.cpp only | **Done** — superseded by M1 (streaming shipped directly rather than staged) |
| **M1** | Streaming + TTFT/ITL for llama.cpp, open-loop-by-default | **Done** — `TestSender_Do_RecordsTTFTAndInterTokenGaps`, `TestLlamaCPP_StreamCountsTokensAndStopsAtStop`; verified end-to-end against a local mock SSE server |
| **M2** | Add OpenAI-compatible and Ollama adapters | **Done** — `TestOpenAI_StreamStopsAtDoneSentinel`, `TestOllama_StreamNDJSONStopsAtDone` |
| **M3** | CSV/Prometheus output, bounded-memory design | **Mostly done** — reservoir sampling, CSV, and Prometheus textfile output are implemented and unit-tested; a real multi-hour soak run against a live backend has not actually been executed, only the bounded-memory mechanism itself (unit tests up to 200k samples). YAML config file explicitly deferred (see note above) |
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` with cross-compiled binaries | **Not started** |

## Testing strategy

- **Behavioral, not just compiling**: ported as
  `TestRunOpenLoop_QueueDelayGrowsUnderSaturation`
  (`internal/dispatch/dispatch_test.go`) — proves the CO-correction mechanism
  against an artificially slow mock backend, including the assertion that
  corrected latency exceeds naive latency once queueing occurs.
  `internal/stats/stats_test.go`'s `TestAccumulator_FlagsCoordinatedOmission`
  and `TestAccumulator_NoWarningWhenLatenciesAgree` are the equivalent proof
  for the divergence-warning logic itself.
- **Adapter tests** via `httptest` (`internal/adapter/*_test.go`) using
  hand-written synthetic SSE/NDJSON fixtures, covering stream-termination
  correctness (must stop at `stop:true` / `[DONE]` / `done:true` and not count
  or hang on trailing data after it). Recorded-real traffic fixtures from an
  actual llama.cpp/Ollama server are not yet included — a reasonable M4
  addition once one is convenient to capture.
- **No live-server tier in CI** (there is no CI yet — M4). A real end-to-end
  run against a live inference server was done manually during development
  (see the Quickstart in `README.md`) rather than automated.

## Explicitly deferred (documented limitation, not silently omitted)

Gil Tene-style synthetic sample backfill for closed-loop coordinated-omission
correction is not implemented in v0.1.0 — open-loop-by-default already covers
the common case this tool is built for, and closed-loop is kept only for
parity with the old tool's mode, with its CO-unsafety documented rather than
worked around.

## License

MIT — see [`../../LICENSE`](../../LICENSE); each project carries its own copy
of the same license text.
