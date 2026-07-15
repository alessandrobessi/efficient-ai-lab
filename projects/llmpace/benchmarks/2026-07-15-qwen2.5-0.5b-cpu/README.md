# Benchmark: Qwen2.5-0.5B-Instruct on Apple M4 CPU

A real rate sweep against a real `llama-server`, not a mock — showing the
saturation knee llmpace exists to make visible.

![Saturation curves](saturation_curves.png)

## Setup

| | |
|---|---|
| Hardware | Apple M4, 10 cores (4P+6E), 16GB RAM, macOS 15.7.3 |
| llama.cpp | version 9960 (`a935fbffe`), installed via `brew install llama.cpp` |
| Model | [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M, ~491MB) |
| Server launch | `llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8090 --threads 8 --parallel 4 -c 4096` |
| llmpace | this branch (v0.1.1, pre-tag), built from source |
| Prompts | llmpace's built-in default set (`internal/prompts.Default`), round-robin |
| Per-request config | `-max-tokens 64`, temperature 0 |
| Client concurrency | `-concurrency 30` (generously above every tested rate, so the client's own sender-slot pool is never the bottleneck at these rates — see [Known limitations](#known-limitations)) |
| Rates tested | 1, 2, 3, 4, 5 req/s (open-loop) |
| Duration per point | 20s configured (see [Known limitations](#known-limitations) — shorter than a rigorous run) |

Reproduce with:

```bash
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8090 --threads 8 --parallel 4 -c 4096 &
for rate in 1 2 3 4 5; do
  ./llmpace -backend llamacpp -url http://127.0.0.1:8090 \
    -mode open-loop -rps "$rate" -concurrency 30 -max-tokens 64 -duration 20s \
    -label "rate_${rate}" -csv results.csv
done
uv run --project ../../../.. python plot.py   # regenerates saturation_curves.png from results.csv
```

## What the data shows

| Rate (req/s) | Throughput (req/s) | Naive p99 latency | Corrected p99 latency | Peak queue depth |
|---|---|---|---|---|
| 1 | 0.95 | 624ms | 626ms | 1 |
| 2 | 1.94 | 892ms | 893ms | 2 |
| 3 | 2.89 | 1258ms | 1259ms | 4 |
| 4 | 3.34 | 4000ms | 4002ms | 15 |
| 5 | 3.33 | 9028ms | 10128ms | 36 |

The saturation knee is unambiguous: throughput tracks the offered rate up
through 3 req/s, then flattens at ~3.3 req/s from 4 req/s onward — this
server, on this hardware, with 4-way parallel batching, cannot sustain more
than roughly 3.3 requests/sec at 64 output tokens each. Past that point,
p99 latency and TTFT both grow by more than 7x from their 3 req/s values
while throughput stays flat — the textbook signature of a system past its
knee.

**Where the naive/corrected divergence actually shows up, and where it
doesn't.** At rates 1-3, naive and corrected latency are nearly identical —
client-side queueing (the gap coordinated omission hides) never becomes
significant because `-concurrency 30` gives every request a free sender
slot immediately; almost all of the latency growth in this range is
server-side (the model doing more concurrent work). Only at rate 5 —
exactly where peak queue depth (36) exceeds `-concurrency` (30) — does a
real client-side queueing gap open up (queue delay p99 reaches 1.4s, and
corrected p99 latency pulls ahead of naive by ~1.1s). This is the expected,
honest behavior: coordinated omission is about client-side dispatch delay
specifically, and it only bites once the client itself starts falling
behind, not before. A tighter `-concurrency` (e.g. 4, matching `--parallel`)
would surface the same effect at much lower request rates — left as a
natural follow-up sweep.

## Known limitations of this specific run

- **Duration per point (20s) is shorter than a rigorous benchmark would use**
  (the original ask was 5-10 minutes per point). This was a deliberate
  tradeoff for turnaround time in one working session, not a claim that 20s
  is methodologically sufficient — treat the numbers above as indicative of
  the real saturation behavior, not as precise, low-variance measurements.
  No repeated runs were taken per rate, so there's no variance/error bar on
  any of these numbers.
- **Only one concurrency setting (30) was tested.** The interaction between
  `-concurrency` and `--parallel` (the server's own batching width) is real
  and only partially explored here — see the note above.
- **One model, one quantization, one machine.** This demonstrates llmpace's
  measurement correctness and the saturation-knee concept, not a
  quantization or hardware comparison (that's `quantscope`'s job, not
  llmpace's).
- **No error-rate stress test.** All 5 points completed with 0 errors;
  behavior under actual request failures/timeouts wasn't exercised here.
