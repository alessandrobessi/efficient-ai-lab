# llmpace

[![llmpace CI](https://github.com/alessandrobessi/efficient-ai-lab/actions/workflows/llmpace-ci.yml/badge.svg)](https://github.com/alessandrobessi/efficient-ai-lab/actions/workflows/llmpace-ci.yml)

**Status: v0.1.0 tagged.** Builds, tests, and runs end-to-end against
llama.cpp, OpenAI-compatible, and Ollama servers. CI on every push,
cross-compiled release binaries for darwin/linux amd64/arm64 — see the
[releases page](https://github.com/alessandrobessi/efficient-ai-lab/releases)
or [`CONTRIBUTING.md`](CONTRIBUTING.md) to build from source. See
[`ROADMAP.md`](ROADMAP.md) for the full architecture rationale and what's
still open (a real multi-hour soak validation, a YAML multi-stage config
file, a live Prometheus endpoint).

A load testing tool built specifically for LLM inference servers, correct by
default about the one thing generic load testers routinely get wrong under
load: **coordinated omission**.

## Table of contents

- [The problem](#the-problem)
- [What llmpace does differently](#what-llmpace-does-differently)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [CLI reference](#cli-reference)
- [Backends](#backends)
- [Output formats](#output-formats)
- [Architecture](#architecture)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Relationship to this repo](#relationship-to-this-repo)
- [License](#license)

## The problem

Tools like wrk, k6, and locust measure HTTP request/response latency in
general. LLM inference servers aren't general HTTP services — a request has a
time-to-first-token (TTFT) and then a stream of tokens at some inter-token
latency (ITL), and the thing you actually care about (does this feel slow to a
user watching text stream in) lives in that per-token distribution, not in a
single round-trip time.

Worse: many load testers, including this program's own first attempt at one
(see [Week 8](../../experiments/08-load-testing/README.md)), measure latency
from when a request is *actually sent* rather than when it was *supposed* to be
sent. Under saturation, requests queue up before dispatch — measuring from
actual send time silently discards exactly the tail latency that overload
produces. This is **coordinated omission**, and it can make a server look
20-36x faster than it really is at high concurrency (the exact gap Week 8
measured).

## What llmpace does differently

- **Open-loop dispatch by default.** Requests are scheduled at a fixed nominal
  rate regardless of response time, so queueing under load shows up in the
  numbers instead of being hidden by a closed request/wait/re-request loop.
- **Native streaming support.** Consumes SSE and NDJSON response streams
  directly — TTFT and per-token ITL are first-class measurements, not
  something bolted onto a total-latency number.
- **Naive and corrected percentiles side by side, always.** Every report shows
  both, with a visible warning when they diverge — you don't have to know to
  ask for coordinated-omission correction to get it.
- **Bounded memory on long runs.** Percentiles are computed over a reservoir
  sample once a run exceeds `-max-samples` requests, instead of holding every
  result in memory — see [Concepts](#concepts).
- **Speaks the backends people actually run**: llama.cpp's `/completion`,
  OpenAI-compatible `/v1/chat/completions`, and Ollama's `/api/generate`.

## Installation

**Pre-built binary** — download the archive for your platform from the
[releases page](https://github.com/alessandrobessi/efficient-ai-lab/releases)
(darwin/linux, amd64/arm64), extract, and run:

```bash
tar -xzf llmpace-v0.1.0-<os>-<arch>.tar.gz
cd llmpace-v0.1.0-<os>-<arch>
./llmpace -version
```

**From source** — requires Go 1.23+. No third-party dependencies (stdlib
only, per [ADR 0001](../../docs/decisions/0001-go-for-inference-gateway.md)'s
reasoning) — `go build` needs no network access beyond fetching the Go
toolchain itself.

```bash
cd projects/llmpace
go build -o llmpace .
./llmpace -h
```

Or run directly without a separate build step:

```bash
go run . -backend llamacpp -url http://127.0.0.1:8080 -duration 10s
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for building, testing, and
extending llmpace (e.g. adding a new backend adapter).

## Quickstart

```bash
# against a local llama.cpp server (llama-server), open-loop, 20 req/s, 30s:
./llmpace -backend llamacpp -url http://127.0.0.1:8080 -rps 20 -duration 30s

# write raw per-request JSONL + a summary JSON alongside it:
./llmpace -backend llamacpp -url http://127.0.0.1:8080 -output results/run1.jsonl

# OpenAI-compatible or Ollama:
./llmpace -backend openai -url http://127.0.0.1:8000 -model my-model -rps 10 -duration 30s
./llmpace -backend ollama -url http://127.0.0.1:11434 -model llama3 -rps 10 -duration 30s

# use your own prompts instead of the built-in default set:
./llmpace -backend llamacpp -url http://127.0.0.1:8080 -prompts my-prompts.jsonl

# a concurrency/rate sweep, appending each run's summary as one CSV row:
for rps in 5 10 20 40; do
  ./llmpace -backend llamacpp -url http://127.0.0.1:8080 -rps "$rps" -duration 30s \
    -label "rps-$rps" -csv sweep.csv
done
```

Example table output (from a real run against a local mock server):

```
llmpace run: run
backend              llamacpp
mode                 open-loop
target               http://127.0.0.1:8099
duration             2.0s
requests             40 (0 errors, 0.0% error rate)
throughput           19.51 req/s
                         p50   p95   p99
latency (naive)      ms  31.7  32.6  33.8
latency (corrected)  ms  33.6  34.7  35.8
queue delay          ms  2.1   -     2.2
time to first token  ms  6.9   7.6   9.0
inter-token latency  ms  6.3   6.4   6.5
tokens/sec (mean)        158.42
```

If corrected p99 latency is more than 2x naive p99, a warning like this is
appended:

```
WARNING: corrected p99 latency (912.4ms) is more than 2x naive p99 (410.1ms) — the target is likely overloaded; trust corrected, not naive, numbers
```

## Concepts

**Open-loop vs. closed-loop dispatch.** Closed-loop mode runs N clients, each
waiting for its response before sending the next request — the model of "N
real users, each waiting for their answer." This makes `ScheduledAt ==
SentAt` for every request by construction, so there is never a queueing gap
to observe: naive and corrected latency come out numerically identical no
matter how overloaded the backend is. Open-loop mode instead dispatches
requests at a fixed nominal rate (`-rps`) regardless of how long previous
requests take; if the backend can't keep up, requests pile up waiting for a
free sender slot, and that wait shows up as queue delay. This is why
open-loop is llmpace's default and closed-loop is opt-in (`-mode
closed-loop`) — only open-loop mode can actually demonstrate coordinated
omission.

**Naive vs. corrected latency.** *Naive* latency is `DoneAt - SentAt` — the
number you get if you only time from when a request actually went out.
*Corrected* latency is `DoneAt - ScheduledAt` — what a real, constant-arrival
population of users would have experienced, including time spent queued
before dispatch even began. Reading only naive numbers under load is exactly
what coordinated omission looks like from the inside: the backend appears to
get *faster* as it saturates, because the requests that would have exposed
the backlog haven't been sent yet.

**TTFT and inter-token latency (ITL).** TTFT is the time from sending a
request to the first streamed token arriving — dominated by queueing and
prefill. ITL is the gap between each pair of consecutive token arrivals
within one request's response — dominated by decode speed. llmpace reports
ITL as a full p50/95/99 distribution across every token gap in the run, not
a mean, because a mean hides exactly the kind of tail stall a user would
notice mid-stream.

**Reservoir sampling.** Holding every result from a multi-hour, high-QPS run
in memory doesn't bound memory usage. Once a run exceeds `-max-samples`
(default 100,000) requests, each metric's percentiles are computed over a
uniform random sample of that size (Algorithm R) instead of the full set —
below that many requests, the "sample" is just every value, so short local
runs get exact percentiles for free.

## CLI reference

Run `./llmpace -h` for the authoritative list; summarized here:

| Flag | Default | Description |
|---|---|---|
| `-backend` | `llamacpp` | Backend to target: `llamacpp`, `openai`, or `ollama` |
| `-url` | `http://127.0.0.1:8080` | Base URL of the inference server (no path suffix — llmpace appends the backend-specific path) |
| `-model` | `""` | Model name, sent in the request body for `openai`/`ollama` backends (llama.cpp's `/completion` doesn't need one) |
| `-mode` | `open-loop` | `open-loop` (default) or `closed-loop` — see [Concepts](#concepts) |
| `-concurrency` | `10` | Concurrent sender slots (open-loop) or concurrent clients (closed-loop) |
| `-rps` | `10` | Target requests/sec, open-loop mode only |
| `-duration` | `30s` | How long to run (Go duration syntax, e.g. `10m`, `1h`) |
| `-prompts` | `""` | JSONL file with a `"prompt"` field per line, cycled round-robin. Empty uses a small built-in default set |
| `-output` | `""` | Path to write raw per-request JSONL results. Also writes `<path-without-ext>_summary.json` |
| `-csv` | `""` | Path to append a one-row CSV summary — write header once, append thereafter, so a sweep script can build a comparison table |
| `-prometheus-out` | `""` | Path to write a Prometheus textfile-collector-format summary (static dump, not a live endpoint — see [Known limitations](#known-limitations)) |
| `-max-tokens` | `128` | Max tokens requested per generation |
| `-temperature` | `0.0` | Sampling temperature (0 = deterministic) |
| `-request-timeout` | `60s` | Per-request HTTP timeout |
| `-label` | `run` | Label recorded in output metadata and in the table header |
| `-max-samples` | `100000` | Per-metric reservoir capacity — see [Concepts](#concepts) |

## Backends

| Backend | Endpoint | Wire format | Notes |
|---|---|---|---|
| `llamacpp` | `POST {url}/completion` | SSE, one `content`/`stop` JSON object per token | Matches `llama-server`'s native API. No `model` field needed |
| `openai` | `POST {url}/v1/chat/completions` | SSE, OpenAI chunk format, terminated by `data: [DONE]` | Works against OpenAI itself, vLLM, TGI, or llama.cpp's own OpenAI-compatible route. Sends the prompt as a single user message |
| `ollama` | `POST {url}/api/generate` | NDJSON, one `response`/`done` JSON object per line | `-model` is required — Ollama's API always needs a model name |

All three adapters stream the response as it arrives — none of them buffer a
full response body — since TTFT and inter-token latency only exist if tokens
are observed as they arrive.

## Output formats

**Table** (stdout, always) — see the example under [Quickstart](#quickstart).

**JSONL** (`-output`) — one line per request, e.g.:

```json
{"scheduled_at":"2026-07-13T14:48:02.305Z","sent_at":"2026-07-13T14:48:02.307Z","done_at":"2026-07-13T14:48:02.341Z","latency_ns":34209000,"corrected_latency_ns":36157501,"queue_delay_ns":1948501,"ttft_ns":9978833,"inter_token_gaps_ms":[5.67,6.47,5.63,6.46],"tokens_generated":5,"status_code":200}
```

**Summary JSON** (written alongside `-output` as `<name>_summary.json`) —
run configuration plus every metric in the table, as JSON (field names like
`naive_latency_p99_ms`, `corrected_latency_p99_ms`, `ttft_p50_ms`,
`itl_p99_ms`, `coordinated_omission_warning`).

**CSV** (`-csv`) — one row per run, header written once: `timestamp, label,
backend, mode, target_url, concurrency, requests_per_second, duration_s, n,
errors, error_rate, throughput_rps, naive_p50_ms, naive_p95_ms,
naive_p99_ms, corrected_p50_ms, corrected_p95_ms, corrected_p99_ms,
queue_delay_p50_ms, queue_delay_p99_ms, ttft_p50_ms, ttft_p95_ms,
ttft_p99_ms, itl_p50_ms, itl_p95_ms, itl_p99_ms, tokens_per_second_mean,
coordinated_omission_warning`. Meant for sweep scripts (see the
[Quickstart](#quickstart) rps-sweep example) that need every run's summary
in one comparable table.

**Prometheus textfile** (`-prometheus-out`) — a static dump in
`node_exporter` textfile-collector format (`llmpace_requests_total`,
`llmpace_latency_ms{quantile="0.99",view="corrected"}`, `llmpace_ttft_ms`,
`llmpace_itl_ms`, `llmpace_tokens_per_second_mean`, all labeled with
`label`/`backend`/`mode`), for archival or scraping via a file-based target
against `../../infrastructure/prometheus/prometheus.yml`.

## Architecture

```
projects/llmpace/
├── main.go                    CLI entry point, wires everything together
└── internal/
    ├── adapter/                backend-specific request building + response streaming
    │   ├── adapter.go            Adapter interface
    │   ├── llamacpp.go, openai.go, ollama.go
    │   └── sse.go                shared SSE/NDJSON line-scanning helpers
    ├── dispatch/                request scheduling
    │   ├── sender.go              builds+sends+times one request end-to-end
    │   ├── openloop.go            default mode
    │   └── closedloop.go          opt-in mode
    ├── stats/                   percentile engine
    │   ├── reservoir.go            bounded-memory sampling (Algorithm R)
    │   └── stats.go                Accumulator, Summary, divergence warning
    ├── prompts/                 round-robin prompt source
    ├── config/                  CLI flag parsing
    └── report/                  table/JSON/CSV/Prometheus output
```

See [`ROADMAP.md`](ROADMAP.md) for the design rationale behind each package
and what's planned but not yet built.

## Testing

```bash
go test ./... -race    # all packages
go vet ./...
gofmt -l .              # should print nothing
```

`.github/workflows/llmpace-ci.yml` runs exactly these checks on every push
or PR touching `projects/llmpace/**`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow.

Notable tests: `internal/dispatch/dispatch_test.go`'s
`TestRunOpenLoop_QueueDelayGrowsUnderSaturation` is the concrete behavioral
proof of the coordinated-omission mechanism (an artificially slow mock
backend, asserting corrected latency exceeds naive latency once queueing
occurs); `internal/stats/stats_test.go`'s
`TestAccumulator_FlagsCoordinatedOmission` proves the divergence-warning
logic the same way. Each adapter has `httptest`-based stream-termination
tests (must stop at `stop:true` / `[DONE]` / `done:true`, not hang or
over-count on trailing data after it).

## Known limitations

- **No YAML multi-stage config file.** Every run is single-stage,
  single-backend, driven entirely by CLI flags. A sweep across
  stages/backends needs an outer shell loop (see the
  [Quickstart](#quickstart) example) rather than one `llmpace` invocation.
- **No live Prometheus `/metrics` endpoint.** `-prometheus-out` writes a
  static file after the run completes, not something scrapeable *during* a
  long run.
- **Closed-loop mode cannot demonstrate coordinated omission**, by
  construction (see [Concepts](#concepts)) — it exists only for parity with
  workloads that genuinely are "N fixed concurrent clients."
- **No Gil Tene-style synthetic sample backfill** for closed-loop
  coordinated-omission correction — open-loop-by-default already covers the
  case this tool is built for.
- **Not validated at real multi-hour/high-QPS scale.** The bounded-memory
  reservoir mechanism is unit-tested up to 200k samples; a real soak run
  against a live backend hasn't been executed yet.

These are the genuinely open items — see [`ROADMAP.md`](ROADMAP.md) for the
full rationale behind each.

## Relationship to this repo

Evolves `../../services/load-generator/`, this program's own Week 8 load
tester. That tool's percentile engine and coordinated-omission dispatch
mechanism are sound and were carried forward; its HTTP client had no
streaming support and was hardcoded to one gateway's request schema, which
is the main reason this needed to be its own project rather than a patch.
See `ROADMAP.md` for the exact reuse plan and what's cited from where.

## License

MIT — see [`LICENSE`](LICENSE) (this project's own copy) or the
[repo root](../../LICENSE).
