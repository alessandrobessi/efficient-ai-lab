# Hypotheses — Week 8: Load Testing and Observability

## Overall (naive)

Throughput should scale roughly linearly with concurrency until some resource
(CPU, memory, or the model's own serialization) saturates, after which latency
should grow and throughput should plateau — the standard textbook queueing story.

## Central hypothesis

> Throughput plateaus almost immediately past concurrency=1, because llama-server
> (run with `-np 1`, one processing slot) serializes all requests regardless of how
> many clients are waiting — concurrency beyond 1 buys queueing, not parallelism.
> Latency should then scale roughly linearly with concurrency (each extra client
> waits behind N-1 others), and error rate should stay near zero until queued wait
> time starts exceeding the request timeout, at which point it should rise sharply.

Falsifiable by: throughput continuing to climb meaningfully as concurrency
increases, latency growing sub-linearly (suggesting real parallelism), or errors
appearing for a reason other than timeouts.

## Why `-np 1` is a deliberate, disclosed choice, not an oversight

llama-server supports multiple processing slots (`--parallel N`) that can interleave
several requests' batches on the same model — a form of real concurrency. This week
deliberately uses `-np 1` (one slot) to isolate the specific question the Field Note
is named after — "What Happens When Multiple Users *Share* a CPU Language Model?" —
meaning one model instance, not an auto-scaled fleet of instances or interleaved
slots. Whether `-np >1` changes this week's collapse curve is a direct, obvious
follow-up (see §11).

## Sub-hypotheses (the roadmap's specific analysis questions)

- **When does throughput stop increasing?** Expected: almost immediately (by
  concurrency ≈ 2-5), since a single serialized processing slot has a fixed
  ceiling — sending it more concurrent work doesn't increase how fast it works, only
  how long the queue gets.
- **When does latency become unacceptable?** No principled threshold assumed in
  advance — this is what the concurrency sweep is for. Expected to track queueing
  theory: roughly `latency(N) ≈ N × per-request-service-time` for a single-server
  queue.
- **What happens to p99 latency?** Expected to track p50 closely at low
  concurrency (little variance) and diverge from it as the queue grows (a few unlucky
  requests land behind an especially long queue) — until requests start timing out,
  at which point p99 should approach the timeout ceiling itself rather than keep
  growing, since anything slower than that gets recorded as an error, not a very
  slow success.
- **Does the system queue requests?** Expected: yes, directly observable via
  llama-server's own `--metrics` endpoint (`requests_deferred`), not just inferred
  from latency growth.
- **Does CPU usage reach 100%?** Expected: yes, but only on the 1-2 cores this
  machine's `-t 2` setting (Week 2/3's throughput-optimal thread count) actually
  assigns to llama-server — not system-wide, since this model, by design, doesn't use
  the other 8 cores on this 10-core machine.
- **What resource becomes the bottleneck?** Expected: the single processing slot
  itself (a software/configuration constraint — `-np 1`), not raw CPU capacity —
  i.e. the ceiling should show up as a hard cap on `requests_processing` (≤1) well
  before the assigned CPU cores show sustained 100% saturation across the full
  request lifecycle.

## Coordinated omission (separate, dedicated demonstration)

> A load generator that only measures latency from actual dispatch time (not the
> nominal scheduled time) will under-report tail latency whenever its own sender
> pool can't keep up with the target rate — because the requests that would reveal
> the backlog haven't been sent yet when the measurement window closes.

Falsifiable by: naive and corrected latency percentiles staying close together even
when the sender pool is deliberately undersized relative to the target rate.
