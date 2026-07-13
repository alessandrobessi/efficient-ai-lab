# llmpace

**Status: core implemented (M0-M3), pre-release.** Builds, tests, and runs
end-to-end against llama.cpp, OpenAI-compatible, and Ollama servers. Not yet
packaged as a tagged release (no CI, no cross-compiled binaries,
no `CONTRIBUTING.md`) — see [`ROADMAP.md`](ROADMAP.md) for what's left (M4).

A load testing tool built specifically for LLM inference servers, correct by
default about the one thing generic load testers routinely get wrong under
load: **coordinated omission**.

## Quickstart

```bash
cd projects/llmpace
go build -o llmpace .

# against a local llama.cpp server (llama-server), open-loop, 20 req/s, 30s:
./llmpace -backend llamacpp -url http://127.0.0.1:8080 -rps 20 -duration 30s

# write raw per-request JSONL + a summary JSON alongside it:
./llmpace -backend llamacpp -url http://127.0.0.1:8080 -output results/run1.jsonl

# OpenAI-compatible or Ollama:
./llmpace -backend openai -url http://127.0.0.1:8000 -model my-model -rps 10 -duration 30s
./llmpace -backend ollama -url http://127.0.0.1:11434 -model llama3 -rps 10 -duration 30s
```

Run `./llmpace -h` for every flag (mode, concurrency, max-tokens, temperature,
CSV/Prometheus output, reservoir cap for long runs, etc).

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
- **Speaks the backends people actually run**: llama.cpp's `/completion`,
  OpenAI-compatible `/v1/chat/completions`, and Ollama's `/api/generate`.

## Relationship to this repo

Evolves `../../services/load-generator/`, this program's own Week 8 load
tester. That tool's percentile engine and coordinated-omission dispatch
mechanism are sound and are being carried forward; its HTTP client has no
streaming support and is hardcoded to one gateway's request schema, which is
the main reason this needs to be its own project rather than a patch. See
`ROADMAP.md` for the exact reuse plan.

## License

MIT — see [`../../LICENSE`](../../LICENSE).
