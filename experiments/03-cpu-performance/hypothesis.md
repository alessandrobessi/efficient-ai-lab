# Hypotheses — Week 3: CPU Performance Engineering

## Overall

CPU inference performance is shaped by hardware topology (core types, thread
scheduling), workload shape (context length), and environment (contention, thermals) —
not just raw FLOPs. This week tests four specific mechanisms, several of them direct
follow-ups on open questions from Weeks 1–2 (tracked in
[`docs/methodology/open-questions.md`](../../docs/methodology/open-questions.md)).

## Experiment 3.1 — Thread Scaling

> Throughput increases with thread count up to roughly the performance-core count (4
> on this machine: Apple M4, 4P+6E), then flattens or declines as efficiency cores and
> cross-core-type scheduling overhead start to dominate.

Follows up on Q3 (Week 1) and Q5 (Week 2). Falsifiable by: the collapse point not
aligning with the P-core count, or throughput continuing to climb past 4 threads.

## Experiment 3.2 — Context Scaling

> TTFT continues to grow with prompt length beyond Week 1's tested range (up to 4096
> tokens here, vs 2048 there), and — because decode attends over the full KV cache —
> decode speed measurably decreases as context grows, contrary to Week 1's "roughly
> constant" read of noisy fp32/Python data.

Follows up on Q4 (Week 1). Falsifiable by: TTFT growth staying linear all the way to
4096 tokens, or decode speed staying flat across context lengths.

## Experiment 3.3 — Background Load

> Inference throughput degrades monotonically as concurrent, unrelated CPU load
> increases, since background processes compete for the same physical cores the
> inference threads need.

Falsifiable by: throughput not degrading, or degrading non-monotonically with no
plateau.

## Experiment 3.4 — Thermal Effects

> If Week 2 Experiment 2.3's strong throughput decline was mainly a
> thread-oversubscription artifact (10 threads on a 4P+6E machine) rather than a
> generic thermal effect, then repeating the same style of sweep at the
> throughput-optimal thread count (2) should show much less decline.

Follows up on Q7/Q8 (Week 2). Falsifiable by: the same strong decline-then-plateau
shape appearing again at 2 threads, which would instead support a genuine,
thread-count-independent thermal effect.
