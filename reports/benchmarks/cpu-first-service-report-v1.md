# Building and Breaking a CPU-First AI Service on Kubernetes

### Phase III Final Deliverable — Weeks 7–9, Efficient AI Systems

**Central question:** How do we turn a locally running model into a reliable and
observable AI service?

This report synthesizes three weeks building and stress-testing a production-shaped
system on top of Phase I-II's model/quantization findings: a Go inference gateway
(Week 7), a hand-rolled load generator plus Prometheus/Grafana observability (Week
8), and Kubernetes deployment plus deliberate failure injection (Week 9). All three
weeks front the same model established in Phase II (`Qwen2.5-1.5B-Instruct`,
Q4_K_M) on the same Apple M4 machine.

- [Week 7 — Go Inference Gateway](../../experiments/07-inference-service/README.md)
- [Week 8 — Load Testing and Observability](../../experiments/08-load-testing/README.md)
- [Week 9 — Kubernetes and Failure Engineering](../../experiments/09-kubernetes/README.md)
- [Open questions tracker](../../docs/methodology/open-questions.md)
- [Phase I report](cpu-inference-performance-report-v1.md) · [Phase II report](cpu-slm-benchmark-report-v1.md)

---

## 1. The Service: A Thin, Typed Gateway in Front of llama-server

Week 7 built `services/inference-gateway` — a single static Go binary (17.6MB
distroless Docker image, no third-party web framework) implementing the roadmap's
required surface: `GET /health`, `GET /ready`, `GET /metrics`, `POST /v1/generate`.
Its one substantive design decision, validated by every later week: **`/health` and
`/ready` answer different questions** (is the process alive vs. can it actually
serve, i.e. is llama-server reachable) — the same distinction Kubernetes'
liveness/readiness probes are built around, and directly reused in Week 9's manifest
without modification. Verified end-to-end (not just unit-tested): a real graceful
shutdown correctly drained a live 3-second in-flight request before exiting on
SIGTERM, and the built Docker image ran correctly against a real llama-server via
`host.docker.internal`.

## 2. Load Generation: Built, Not Borrowed

Per FULL-ROADMAP.md's explicit brief, Week 8 built a concurrent load generator in Go
from scratch — goroutines, channels, a ticker for rate limiting — rather than reaching
for an existing tool. Two dispatch modes matter for different reasons: **closed-loop**
(N clients, each waiting for its response before asking again) is what "5 concurrent
users" or "20 concurrent users" actually means, and is what every workload in this
report uses; **open-loop** (fixed nominal rate, independent of response time) exists
specifically to demonstrate **coordinated omission** — a load generator that only
measures latency from actual dispatch time, not the time a request was supposed to go
out, systematically hides tail latency when its own dispatch falls behind. The
demonstration was stark: naive p50 latency looked unremarkable (3.4s) while corrected
latency — what a real constant-rate user population would experience — was **77.5
seconds** at p50, a 20x+ gap, from one real, reproducible run.

## 3. The Core Finding, Established in Week 8 and Confirmed Unchanged in Week 9

llama-server run with one processing slot (`-np 1`) serializes all requests
regardless of client concurrency — confirmed directly via its own `requests_processing`
metric never exceeding 1, and `requests_deferred` (an explicit queue depth) peaking
at 78 during a concurrency=80 test. Three independent measurements across two weeks
agree on the shape of what happens next:

| | Week 8 (native) | Week 9 (Kubernetes) |
|---|---|---|
| throughput at concurrency 1 | 0.83 req/s | 0.66 req/s |
| collapse point (error rate >20%) | concurrency ≈ 20 | concurrency ≈ 20 |
| error rate at concurrency 80 | 72.5% | 84.9% |
| CPU during saturation | 1-2 of 10 cores at 96-98%/54-64% | (not independently re-measured; same `-t 2` config) |

**Containerizing the exact same bottleneck doesn't change it** — Week 9's
Kubernetes collapse curve tracks Week 8's native curve within noise (see Week 9
README §9's figure). The bottleneck throughout is a configuration decision
(`-np 1`, one processing slot per model instance) sitting on top of a real compute
ceiling (this machine's 2 cores assigned via `-t 2`), not something either the
gateway or Kubernetes introduces or fixes.

## 4. Reliability: Redundancy Works, Concretely

Week 9's clearest, most unambiguous result: deleting the only replica mid-traffic
caused a **30% error rate** and a full **15-second** outage while Kubernetes
rescheduled the pod and reloaded the model from scratch. With 2 replicas, the same
deletion caused only a **5.5% error rate** — the Service kept routing to the
survivor throughout, and both replicas were back to full capacity in 6.8 seconds.
Memory limits behave as a hard cliff, not a gradual slope: every tested limit from
4Gi down to 1.5Gi ran perfectly healthy, while 1Gi immediately crash-looped
(`OOMKilled`) — with the caveat that actual memory usage is load-dependent (~1.06GB
idle, up to ~2.1GB observed under concurrent traffic in an earlier test), so a "safe"
limit has to account for expected load, not just the model's static footprint.

## 5. Horizontal Scaling: This Phase's Most Important Negative Result

The naive expectation — that N independent model replicas should let aggregate
throughput scale roughly N-fold, unlike Week 8's single shared instance — did not
hold. Under a fixed total concurrency (20) or concurrency scaled with replica count
(20/replica), throughput across 1/2/3 replicas stayed flat or got *worse*
(0.52→0.40→0.47 and 0.62→0.36→0.31 req/s respectively), nowhere near linear scaling.
Two pieces of direct evidence rule out the obvious first guesses: `kubectl top pod`
showed every replica using well under its CPU limit throughout (not throttled), and
querying each pod's own `llamacpp:n_decode_total` after a 3-replica run found a
roughly **10x spread** in work done per pod — load was genuinely uneven, not just
measured coarsely. The most likely mechanism is a real, well-documented Kubernetes
characteristic: **Services load-balance HTTP at the TCP connection level** (iptables
DNAT at connection-establishment time), while any normal HTTP client's persistent
keep-alive connections, once established, stick to one backend for their lifetime —
so a handful of long-lived connections can distribute unevenly across a small number
of pods purely by chance. A second, harder-to-isolate contributor: this is a
single-node kind cluster, so "replicas" here are processes time-sharing one Apple
M4's actual CPU/RAM (observed under genuine memory pressure during the heaviest
tests), not the independent hardware a real multi-node cluster would provide. This
result is reported as a real, disclosed limitation of the test setup, not resolved —
the natural next step (a genuine multi-node cluster) is exactly what would
distinguish "horizontal scaling doesn't help this workload" from "this single-node
setup couldn't demonstrate it."

## 6. Statistical and Engineering Methodology, Carried Forward

Every quantitative claim in Weeks 7-9 traces to a real, executed run against a real
service — end-to-end gateway verification (not just unit tests), a live 224-second
coordinated-omission run, a real kind cluster with real OOM kills and real pod
deletions. Week 8/9's percentile computation (exact, sort-based — not an HDR
histogram approximation) and Week 5's bootstrap-CI/paired-comparison toolkit remain
the statistical backbone wherever this phase's data warranted it (e.g. comparing
Week 8 vs. Week 9's collapse curves).

## 7. Observed Anomalies This Phase

1. **Coordinated omission's naive-vs-corrected latency gap (20-36x)** — Week 8's
   single clearest illustration that a load generator's own measurement methodology
   can hide the exact failure mode it's supposed to reveal.
2. **250m CPU collapses completely (100% errors), not just slowly** (Week 9.1) — a
   qualitative cliff, not an extrapolation of the 500m-4000m trend.
3. **Memory limits are a hard OOM cliff, with a load-dependent safe boundary**
   (Week 9.2) — 1.5Gi fully healthy, 1Gi immediately crash-looping, and the model's
   own footprint varying 2x between idle and loaded conditions.
4. **Horizontal scaling not delivering proportional throughput** (Week 9.5) — this
   phase's most significant negative result, with direct evidence (uneven per-pod
   decode counts) pointing at connection-level Service load balancing as a
   contributing mechanism, compounded by single-node resource sharing.

## 8. Unanswered Questions Going Into Phase IV

The full, itemized register is
[`docs/methodology/open-questions.md`](../../docs/methodology/open-questions.md).
Most directly relevant to Phase IV's "when is an SLM the right choice" framing:

- Does a genuine multi-node cluster resolve Week 9.5's horizontal-scaling result, or
  is connection-level load balancing a persistent tax on this architecture regardless
  of node topology?
- Does `--parallel N > 1` (llama-server's own continuous batching, flagged since
  Week 8) raise the single-instance throughput ceiling this phase found repeatedly,
  and how would it interact with horizontal scaling?
- How do llama-server's HTTP-level latencies compare under Kubernetes' own ingress
  patterns (not just NodePort) at genuine multi-node scale?

## 9. Reproducibility

Every number traces back to raw JSONL/JSON under
`results/raw/{07-inference-service,08-load-testing,09-kubernetes}/` (Week 7 has no
raw experiment data — it's a build week, see its README), generated by the scripts
in each week's `experiments/*/scripts/` or `services/*/` and processed by each week's
`analysis/` scripts. Week 9's cluster is fully reproducible from
`infrastructure/kubernetes/create_cluster.sh` plus the checked-in manifests; see each
week's README §4 for exact commands.
