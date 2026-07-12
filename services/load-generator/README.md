# Load Generator

**Week 8 — Phase III.** A hand-rolled concurrent load tester for the Week 7
inference gateway, built from scratch (goroutines, channels, a ticker for rate
limiting) rather than reaching for an existing load-testing tool — per
FULL-ROADMAP.md's explicit Week 8 brief: "the objective is educational."

See [`experiments/08-load-testing/README.md`](../../experiments/08-load-testing/README.md)
for the actual workload results and analysis; this README covers the tool itself.

## Usage

```bash
go build -o load-generator .
./load-generator \
  -url http://127.0.0.1:8080/v1/generate \
  -mode closed-loop \
  -concurrency 20 \
  -duration 60s \
  -prompts ../../evaluation/datasets/v1.jsonl \
  -output results.jsonl \
  -label my_run
```

| flag | default | meaning |
|---|---|---|
| `-url` | `http://127.0.0.1:8080/v1/generate` | gateway endpoint under test |
| `-mode` | `closed-loop` | `closed-loop` or `open-loop` (see below) |
| `-concurrency` | `1` | concurrent clients (closed-loop) or sender-pool size (open-loop) |
| `-rps` | `10` | target requests/sec (open-loop only) |
| `-duration` | `30s` | how long to run |
| `-prompts` | `evaluation/datasets/v1.jsonl` | JSONL file with a `prompt` field per line, cycled round-robin |
| `-output` | *(required)* | path for raw per-request JSONL output |
| `-max-tokens` | `128` | `max_tokens` per request |
| `-temperature` | `0.0` | `temperature` per request |
| `-request-timeout` | `30s` | per-request HTTP client timeout |
| `-label` | `run` | recorded in the output metadata |

Output: `<output>` (one JSON `client.Result` per line) and
`<output-without-extension>_summary.json` (run config + computed `stats.Summary` —
n, errors, error rate, throughput, latency/TTFT percentiles, tokens/sec).

## Closed-loop vs. open-loop

- **Closed-loop** (`-mode closed-loop`): N goroutines, each looping — send, wait for
  the response, send again — for the run duration. This is the literal model of "N
  real users, each waiting for their answer before asking the next question," and is
  what Workloads A-D (FULL-ROADMAP.md's Week 8 spec) are defined in terms of.
- **Open-loop** (`-mode open-loop`): dispatches at a fixed nominal rate (`-rps`)
  through a bounded pool of `-concurrency` senders, regardless of how long previous
  requests take. Exists specifically to make **coordinated omission** visible: if the
  backend can't keep up, requests queue for a sender slot, and actual dispatch time
  drifts later than the nominal scheduled time. Every result records both — see
  `internal/worker/openloop.go`'s doc comment for the full reasoning, and
  `experiments/08-load-testing/README.md` §9 for a real measured example (naive vs.
  corrected latency differing by 20-36x).

## Testing

```bash
go vet ./...
go test ./... -v
```

All tests use `httptest` — no real gateway or llama-server is required. The
`worker` package's tests include a specific regression test
(`TestRunOpenLoop_QueueDelayGrowsUnderSaturation`) that artificially slows a mock
backend to confirm queue delay actually grows under saturation, not just that the
code compiles.

## Design notes

- **Exact percentiles, not HDR histograms.** A single workload run comfortably fits
  in memory (thousands of samples, not millions) — sort-and-index is simpler and
  fully accurate at this scale, so there's no reason to trade accuracy for a
  technique built for a problem this program doesn't have.
- **Prompts are reused from Weeks 4-6's evaluation dataset**, not a new one — varied,
  realistic prompt lengths for free, no new dataset to build or validate.
- **This tool always calls the gateway, never llama-server directly** — its
  measurements include the gateway's own overhead (validation, middleware,
  proxying), which is the actually-relevant number for "what does a real caller
  experience," not just "how fast is the model."
