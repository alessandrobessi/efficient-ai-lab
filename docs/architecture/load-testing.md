# Load Testing & Observability Architecture (Week 8)

## Topology

```text
LOAD GENERATOR (Go, hand-rolled)          OBSERVABILITY (Docker)
  closed-loop  ──┐                          ┌─ Prometheus (scrapes every 2s)
  open-loop    ──┤                          │    ├─ gateway:8080/metrics
                 │                          │    ├─ llama-server:8799/metrics
                 ▼                          │    └─ node_exporter:9100/metrics
   POST /v1/generate                        │
                 │                          └─ Grafana (provisioned dashboard)
                 ▼
   GO INFERENCE GATEWAY (:8080, native)
                 │
                 ▼
   LLAMA-SERVER (:8799, native, -np 1, --metrics)
                 │
                 ▼
   SMALL LANGUAGE MODEL
```

The load generator, gateway, and llama-server all run **natively on the host**, not
containerized — matching every prior week's convention of measuring the real
machine, not a Docker-Desktop-VM-emulated one (macOS containers don't give accurate
CPU/memory numbers for a CPU-bound workload like this). Only Prometheus and Grafana
are containerized, reaching the native processes via `host.docker.internal`.

## Why three separate metrics sources, not one

- **`gateway_*` metrics** (Week 7's Prometheus instrumentation) see the system from
  the client's perspective: full request duration including gateway overhead,
  validation errors, active in-flight count.
- **`llamacpp:*` metrics** (llama-server's own `--metrics` endpoint, enabled for this
  week only) see the system from the model's perspective: `requests_processing` vs.
  `requests_deferred` is a direct, unambiguous answer to "is this request running or
  queued right now" that the gateway alone can't provide — the gateway doesn't know
  *why* a call to llama-server is slow, only that it is.
- **`node_*` metrics** (node_exporter) see the whole machine: which of this 10-core
  Apple M4's cores are actually busy. Neither the gateway nor llama-server's own
  metrics can show this — both only report their own process, not the host.

Together, these three answer "is it queued, is it computing, or is the machine out
of a resource neither service can see" — three different questions a single
metrics source can't disambiguate.

## Why `-np 1`

llama-server supports multiple processing slots (`--parallel N`) for interleaving
several requests' batches. This week's workloads deliberately use one slot, to
isolate the specific question the Field Note is named after: what happens when
multiple users share *one* model instance, not an auto-scaled fleet or a
multi-slot batching setup. See
[`experiments/08-load-testing/hypothesis.md`](../../experiments/08-load-testing/hypothesis.md)
and its README §11 for why `--parallel N > 1` is flagged as a direct follow-up, not
tested here.

## Closed-loop vs. open-loop load generation

FULL-ROADMAP.md's Workloads A-D are specified by concurrency (1/5/20/sweep), which
maps directly onto **closed-loop** dispatch: N goroutines, each waiting for its own
response before sending the next request — the literal model of N real users. This
is what the official workload results are built from.

**Open-loop** dispatch (fixed nominal rate, independent of response time) is used
separately, specifically to make **coordinated omission** concrete: naive load
generators that only measure latency from when a request was actually sent
under-report tail latency whenever their own sender pool falls behind the target
rate, because the requests that would reveal the backlog haven't been sent yet when
the measurement window closes. See `services/load-generator/internal/worker/openloop.go`'s
doc comment and this week's README §9 for a real measured example (naive vs.
corrected latency differing by 20-36x).
