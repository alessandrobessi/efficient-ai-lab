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
| Server launch | `llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --n-gpu-layers 0 --host 127.0.0.1 --port 8090 --threads 8 --parallel 4 -c 4096` |
| llmpace | this branch (v0.1.2, pre-tag), built from source |
| Prompts | llmpace's built-in default set (`internal/prompts.Default`), round-robin |
| Per-request config | `-max-tokens 64`, temperature 0 |
| Client concurrency | `-concurrency 30` (generously above every tested rate, so the client's own sender-slot pool is never the bottleneck at these rates — see [Known limitations](#known-limitations)) |
| Rates tested | 1, 2, 3, 4, 5 req/s (open-loop) |
| Duration per point | 20s configured (see [Known limitations](#known-limitations) — shorter than a rigorous run) |

**On the `--n-gpu-layers 0` flag:** an earlier version of this benchmark
omitted it. `llama-server`'s own default is `-ngl auto`, and this
Homebrew build has Metal compiled in (`backends: BLAS,MTL`, confirmed via
a separate tool's manifest against the same binary) — so that earlier run
likely benchmarked GPU-accelerated inference while labeled "CPU." This
version passes `--n-gpu-layers 0` explicitly (visible in the process
arguments, not just a config file) so the CPU claim is actually verifiable.
Interestingly, the true CPU-only numbers below are comparable to, and at
the lower rates slightly *better than*, the earlier (likely
GPU-influenced) numbers — this benchmark doesn't have an explanation for
that beyond "a 0.5B model may not benefit much from Metal offload given
dispatch overhead," which is speculation, not something this run
demonstrates. Take the corrected numbers below as authoritative for what
they show (a real CPU saturation curve); treat the comparison to the
retracted GPU-labeled run as an open question, not a finding.

Reproduce with:

```bash
llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --n-gpu-layers 0 --host 127.0.0.1 --port 8090 --threads 8 --parallel 4 -c 4096 &
for rate in 1 2 3 4 5; do
  ./llmpace -backend llamacpp -url http://127.0.0.1:8090 \
    -mode open-loop -rps "$rate" -concurrency 30 -max-tokens 64 -duration 20s \
    -label "rate_${rate}" -csv results.csv
done
uv run --project ../../../.. python plot.py   # regenerates saturation_curves.png from results.csv
```

## What the data shows

| Rate (req/s) | Throughput (req/s) | Naive p99 latency | Corrected p99 latency | Peak in-flight | Peak waiting |
|---|---|---|---|---|---|
| 1 | 0.95 | 466ms | 467ms | 1 | 1 |
| 2 | 1.95 | 549ms | 550ms | 2 | 1 |
| 3 | 2.88 | 1236ms | 1237ms | 4 | 1 |
| 4 | 3.26 | 4628ms | 4629ms | 17 | 1 |
| 5 | 3.19 | 9522ms | 11233ms | 30 | 8 |

The saturation knee is unambiguous: throughput tracks the offered rate
through 3 req/s, then flattens at ~3.2-3.3 req/s from 4 req/s onward —
this server, on this hardware, cannot sustain more than roughly 3.3
requests/sec at 64 output tokens each. Past that point, p99 latency and
TTFT both grow by roughly 7-9x from their 3 req/s values while throughput
stays flat — the textbook signature of a system past its knee.

**A more precise reading of the coordinated-omission story than the
previous version of this benchmark gave.** Naive (service) latency alone
already reveals the saturation clearly: it jumps from 1236ms at 3 req/s to
4628ms at 4 req/s, with **peak in-flight requests** (bounded by
`-concurrency`) climbing from 4 to 17 — none of that needs the
naive-vs-corrected comparison to see. The naive/corrected divergence stays
essentially zero through 4 req/s (peak waiting is pinned at 1 the whole
time) and only opens up at 5 req/s, exactly when **peak in-flight hits the
`-concurrency` ceiling of 30** and **peak waiting jumps to 8** — the point
where the client's own sender-slot pool starts falling behind its nominal
schedule, on top of whatever the backend itself is doing.

The corrected takeaway: **service metrics (naive latency, naive TTFT,
throughput flattening) are what reveal backend saturation, and they reveal
it earlier.** The naive-vs-corrected divergence is a narrower, specific
signal — it tells you the *load generator's own dispatch* has started
falling behind its schedule, which usually only happens once `-concurrency`
itself becomes the limiting factor. Watch peak in-flight against
`-concurrency` and peak waiting against zero, not just the naive/corrected
gap, to get the full picture.

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
  and only partially explored here.
- **One model, one quantization, one machine.** This demonstrates llmpace's
  measurement correctness and the saturation-knee concept, not a
  quantization or hardware comparison (that's `quantscope`'s job, not
  llmpace's).
- **No error-rate stress test.** All 5 points completed with 0 errors;
  llmpace's failed-request tracking (added in v0.1.2 — see the main
  [`README.md`](../../README.md)) wasn't exercised by this particular sweep.
- **The GPU-vs-CPU comparison noted above is an open observation, not a
  conclusion** — this run wasn't designed to isolate why the corrected
  numbers came out comparable to the earlier, retracted GPU-labeled run.
