# llmpace — Development Roadmap

**Status: M0-M5 implemented — `llmpace/v0.1.1` tagged.** The
architecture below reflects what was actually built (see `internal/`), not
just a plan. Two deliberate simplifications remain from the original plan
and are not expected to change for v0.1.x: the YAML multi-stage test-plan
file (config is CLI-flags-only for now, single stage per run) and a live
`/metrics` Prometheus endpoint (a static textfile-collector-format dump via
`-prometheus-out` exists instead) — both noted inline below where they
diverge.

## v0.1.1: fixes from external review

An external review of v0.1.0 (recorded in full in project history) found
three real correctness/design gaps and asked for a real benchmark as
evidence. All four were addressed:

1. **TTFT was only ever measured from `SentAt`, never `ScheduledAt`** —
   inconsistent with how total latency was already handled, and able to
   look clean even when a request was queued a long time before dispatch.
   Fixed: `NaiveTTFT`/`CorrectedTTFT` now exist in parallel with
   `Naive`/`Corrected` latency, all the way through `dispatch.Result` →
   `stats.Summary` → every report format. The divergence warning now checks
   both metrics independently (`TestAccumulator_FlagsTTFTDivergenceEvenWhenLatencyLooksFine`
   is the behavioral proof that TTFT's warning does real, independent work,
   not just piggybacking on latency's).
2. **Open-loop dispatch spawned an unbounded number of goroutines under
   sustained overload** — a bounded sender-slot pool doesn't bound the load
   generator's *own* memory, and a self-inflicted client bottleneck (too few
   sender slots for the offered rate) could be mistaken for the backend
   being overloaded, since both look like growing queue delay. Fixed:
   `dispatch.QueueStats` + `-max-queue-depth` (drop-tail admission control,
   caps admitted-but-not-completed requests at `concurrency+N`), reported
   scheduled/dropped/peak-queue-depth counts in every output format, and an
   explicit configured-vs-actual-duration split in the table (backlog
   drain time is no longer invisible). See
   `TestRunOpenLoop_DropsWhenQueueDepthExceeded` for the behavioral proof
   the bound actually holds under `-race`.
3. **"Tokens" were actually stream chunks** — no adapter guarantees exactly
   one tokenizer token per streamed SSE/NDJSON event. Renamed throughout:
   `TokensGenerated`→`StreamChunks`, `tokens_per_second_mean`→
   `chunks_per_second_mean`, table row "tokens/sec"→"chunks/sec". Real
   tokenizer-based counting remains a possible future addition (see
   [Explicitly deferred](#explicitly-deferred-documented-limitation-not-silently-omitted)).
4. **No real benchmark evidence** — added
   [`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/):
   a real `llama-server` (Qwen2.5-0.5B-Instruct Q4_K_M) on an Apple M4 CPU,
   swept 1-5 req/s, with raw CSV, a plotting script, and 4 charts showing
   the saturation knee. Its own README documents exact hardware/model/
   commit/methodology and is honest about what's shortened relative to a
   rigorous benchmark (20s/point rather than 5-10 min, one concurrency
   setting, no repeated runs).

A real multi-hour soak run against a live backend still hasn't been
executed — the bounded-memory mechanism itself is unit-tested (Reservoir up
to 200k samples), but real-world validation at that scale remains open.

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
│   │   ├── sender.go               builds+sends+times one request end-to-end
│   │   ├── openloop.go             default mode; QueueStats admission control (v0.1.1)
│   │   └── closedloop.go           opt-in, documented CO-correction limitation
│   ├── stats/                    (ported from internal/stats/)
│   ├── prompts/                  (ported from internal/prompts/)
│   ├── config/                   flags (stdlib) + optional YAML test-plan file
│   └── report/                   human table, JSON, CSV, Prometheus exposition
├── benchmarks/                   real (not mock) saturation-curve runs, one dir per run
└── ROADMAP.md, README.md, CONTRIBUTING.md, LICENSE
```

**`Adapter` interface** — the core new abstraction (as implemented, in
`internal/adapter/adapter.go`):

```go
type Adapter interface {
    Name() string
    BuildRequest(ctx context.Context, baseURL string, req Request) (*http.Request, error)
    // Stream reads resp.Body incrementally, invoking onToken with the
    // wall-clock time each token/chunk was observed — no adapter buffers a
    // full response. Returns the total number of stream chunks seen (not
    // guaranteed to equal tokenizer token count — see v0.1.1 note above);
    // TTFT and inter-token gaps are derived by the caller
    // (internal/dispatch.Sender) from the onToken timestamps, not returned
    // here.
    Stream(resp *http.Response, onToken func(t time.Time)) (chunks int, err error)
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

**Admission control (`QueueStats`, v0.1.1)** — `openloop.go`'s dispatcher
loop tracks, via atomics safe for concurrent access from spawned goroutines:
`Scheduled` (every tick that fired), `Dropped` (ticks refused once the
queue hit its bound), and a high-water-mark `peak` of admitted-but-not-
completed requests. `-max-queue-depth N` (default 0 = unbounded, the
original behavior) caps admission at `concurrency+N`; a tick beyond that is
dropped rather than spawning another goroutine. This makes an otherwise
silent, unboundedly-growing client-side backlog into an explicit, reported
number — see `TestRunOpenLoop_DropsWhenQueueDepthExceeded`, which asserts
under `-race` that peak pending never exceeds the configured bound and that
dropped ticks are actually counted, not just silently absorbed.

**Measurement** — TTFT and total latency both get naive (`SentAt`-anchored)
and corrected (`ScheduledAt`-anchored) views, for the same reason: a request
queued a long time before ever being dispatched can show a perfectly clean
naive TTFT even though a real user waited far longer for their first token
(this was a real bug in v0.1.0, fixed in v0.1.1 — see above). Inter-token
latency is reported as a full distribution (p50/95/99), not a mean, because
tail ITL is exactly what coordinated omission hides. Every report shows
naive and corrected percentiles side by side for both metrics, with a
warning when either diverges by more than 2x, rather than requiring a flag
to opt in.

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
| **M4** | Docs, CI, `CONTRIBUTING.md`, tagged `v0.1.0` with cross-compiled binaries | **Mostly done** — `.github/workflows/llmpace-ci.yml` (build/vet/test/gofmt on every push/PR touching `projects/llmpace/**`), `CONTRIBUTING.md`, git tag `llmpace/v0.1.0`, and a `-version` flag baked in via `-ldflags`, verified against a real ldflags build. Cross-compilation for darwin/linux amd64/arm64 was built and smoke-tested locally, but **no GitHub Release with downloadable binaries has been published** — that step was deliberately held back pending the maintainer's own call on publishing publicly. The soak-test and YAML-config gaps noted under M3 are still open too — this milestone is about packaging/release infrastructure, not closing those |
| **M5** | v0.1.1: fixes from external review — scheduled/corrected TTFT, bounded client-side backlog, chunks-not-tokens naming, a real benchmark | **Done** — see [v0.1.1: fixes from external review](#v011-fixes-from-external-review) above for the full breakdown |

## Testing strategy

- **Behavioral, not just compiling**: ported as
  `TestRunOpenLoop_QueueDelayGrowsUnderSaturation`
  (`internal/dispatch/dispatch_test.go`) — proves the CO-correction mechanism
  against an artificially slow mock backend, including the assertion that
  corrected latency exceeds naive latency once queueing occurs.
  `internal/stats/stats_test.go`'s `TestAccumulator_FlagsCoordinatedOmission`
  and `TestAccumulator_NoWarningWhenLatenciesAgree` are the equivalent proof
  for the divergence-warning logic itself. v0.1.1 added
  `TestAccumulator_FlagsTTFTDivergenceEvenWhenLatencyLooksFine` (constructs a
  case where total latency's own divergence stays under threshold while
  TTFT's crosses it, proving the two warnings are independent, not one
  piggybacking on the other) and `TestRunOpenLoop_DropsWhenQueueDepthExceeded`
  (under `-race`: asserts peak pending never exceeds the configured
  `concurrency+maxQueueDepth` bound, and that dropped ticks are actually
  counted).
- **Adapter tests** via `httptest` (`internal/adapter/*_test.go`) using
  hand-written synthetic SSE/NDJSON fixtures, covering stream-termination
  correctness (must stop at `stop:true` / `[DONE]` / `done:true` and not count
  or hang on trailing data after it). Recorded-real traffic fixtures from an
  actual llama.cpp/Ollama server are not yet included — a reasonable future
  addition once one is convenient to capture.
- **No live-server tier in CI.** `.github/workflows/llmpace-ci.yml` runs
  `gofmt`/`go vet`/`go build`/`go test -race` on every push/PR touching
  `projects/llmpace/**` — all against mocks and stub servers, same as the
  local suite. A real end-to-end run against a live inference server *was*
  done manually — see
  [`benchmarks/2026-07-15-qwen2.5-0.5b-cpu/`](benchmarks/2026-07-15-qwen2.5-0.5b-cpu/),
  a real `llama-server` + real GGUF model swept across request rates,
  showing a genuine saturation knee — but this remains a manual step, not
  something CI depends on or automatically reruns.

## Explicitly deferred (documented limitation, not silently omitted)

- **Gil Tene-style synthetic sample backfill** for closed-loop
  coordinated-omission correction is not implemented — open-loop-by-default
  already covers the common case this tool is built for, and closed-loop is
  kept only for parity with the old tool's mode, with its CO-unsafety
  documented rather than worked around.
- **Tokenizer-based token counting.** v0.1.0 counted streamed SSE/NDJSON
  chunks and called them tokens; v0.1.1 renamed the metric
  (`StreamChunks`/`chunks_per_second_mean`) to describe what it actually
  measures rather than implying a precision it doesn't have. Adding a real
  tokenizer (matching the model in use) to count actual output tokens would
  be a legitimate improvement, but is a genuinely separate feature — it
  needs a tokenizer dependency and per-model vocabulary handling, not just a
  rename — and isn't planned for v0.1.x.

## License

MIT — see [`../../LICENSE`](../../LICENSE); each project carries its own copy
of the same license text.
