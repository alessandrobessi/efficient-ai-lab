# Hypotheses — Week 9: Kubernetes and Failure Engineering

## Overall (naive)

Constraining resources should degrade the service predictably and gracefully;
redundancy (extra replicas) should make failures less disruptive and should let
throughput scale with replica count, since each replica runs its own independent
model instance (unlike Week 8's single `-np 1` bottleneck).

## Sub-hypotheses (one per experiment)

### 9.1 — CPU Limits

> Latency degrades smoothly as the CPU limit shrinks below what `-t 2` naturally
> needs (~2 cores), and plateaus at or above 2000m with no further benefit.

Falsifiable by: latency not tracking CPU limit, or continued improvement well past
2000m (suggesting the model can profitably use more than 2 cores' worth of quota).

### 9.2 — Memory Limits

> There exists a memory limit below which the pod cannot run at all (OOMKilled),
> close to but somewhat above the model's known RSS footprint (~1-2.8GB across prior
> weeks' native measurements) — not a gradual slowdown.

Falsifiable by: graceful degradation (e.g. swapping, slower responses) instead of a
hard OOM cliff, or a cliff far from the model's known memory footprint.

### 9.3 — Pod Failure

> Deleting the only replica mid-traffic causes a full outage (100% error rate) for
> the duration of rescheduling + model reload; with 2 replicas, the Service
> continues routing to the survivor, so only a small fraction of in-flight requests
> are lost.

Falsifiable by: comparable error rates regardless of replica count (suggesting the
Service isn't actually routing around the failure), or a 1-replica outage shorter
than the model's own load time (suggesting something other than a full cold start
is happening).

### 9.4 — Load Saturation

> The same concurrency sweep run against Kubernetes collapses at roughly the same
> concurrency level as Week 8's native test, since the bottleneck (llama-server's
> single processing slot) is unchanged by containerization.

Falsifiable by: a meaningfully different collapse point, suggesting Kubernetes
networking/scheduling overhead itself becomes a bottleneck before the model does.

### 9.5 — Horizontal Scaling

> Aggregate throughput scales with replica count, because each replica is an
> independent model instance with its own `-np 1` slot — unlike Week 8, where all
> requests funneled through one instance, this should let 2-3 replicas do 2-3x the
> work of one.

Falsifiable by: throughput not scaling proportionally with replica count despite
each replica having spare, unthrottled CPU capacity (which would point to something
other than compute — e.g. how the Service actually distributes load — as the real
constraint).

## Why this year's "obvious" horizontal-scaling hypothesis deserves skepticism

Every prior systems week in this program (4, 7, 8) found at least one naive
"more resources/replicas → proportionally more of the good thing" assumption
break down for a specific, mechanistic reason (non-monotonic quantization formats,
CPU-core-boundary thread collapse, single-processing-slot queueing). There's no
strong reason to expect Experiment 9.5 to be the first exception — and Kubernetes
Services load-balance HTTP at the TCP **connection** level (via iptables/kube-proxy
DNAT at connection-establishment time), not the HTTP **request** level. A load
generator using persistent keep-alive connections (the normal, sensible behavior for
any real client) could plausibly see uneven backend utilization purely from how few
long-lived connections get distributed across a small number of backend pods — a
mechanism worth checking directly (e.g. via each pod's own request count) rather than
assuming a clean throughput result either way.
