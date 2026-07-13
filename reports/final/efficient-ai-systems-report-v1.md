# Efficient AI Systems

## What I Learned from 12 Weeks of Building, Measuring, and Breaking CPU-First Small Language Model Systems

### Final Flagship Report — Efficient AI Systems

- [Phase I report](../benchmarks/cpu-inference-performance-report-v1.md) (Weeks 1-3) ·
  [Phase II report](../benchmarks/cpu-slm-benchmark-report-v1.md) (Weeks 4-6) ·
  [Phase III report](../benchmarks/cpu-first-service-report-v1.md) (Weeks 7-9) ·
  [Decision framework](../decision-framework/ai-architecture-decision-framework-v1.md) (Week 11)
- [Open questions tracker](../../docs/methodology/open-questions.md) (39 entries across 12 weeks)
- Every number below traces to a specific week's README and its `results/raw/<experiment-id>/`
  data — see each linked report's own reproducibility section for exact commands.

---

## 1. Introduction

The central question this program set out to answer was simple to state and hard to
answer honestly: **how much useful AI can we build under strict compute
constraints?** Not "can a small model technically run on a CPU" — that's a solved
problem — but whether a CPU-only, self-hosted small language model can be pushed
through an entire real system lifecycle (inference, quantization, evaluation,
service architecture, load testing, orchestration, failure recovery) and still come
out the other end as something worth choosing over the default alternative: a large
frontier model behind a paid API.

Twelve weeks of experiments don't produce a single yes-or-no answer to that
question — they produce something more useful: a map of exactly *where* the CPU-first
approach holds up, where it breaks, and what it costs to keep it working. That map is
this report.

## 2. Research Questions

The program's central research question, stated in `FULL-ROADMAP.md` from the start:

> How much useful AI can we build under strict compute constraints?

Each phase asked a narrower version of it:

- **Phase I (Weeks 1-3):** What actually happens when an SLM runs on a CPU?
- **Phase II (Weeks 4-6):** What tradeoffs emerge when SLMs are compressed and compared?
- **Phase III (Weeks 7-9):** How do we turn a local model into a reliable, observable service?
- **Phase IV (Weeks 10-12):** When is an SLM actually the right architectural choice?

And, running through every week regardless of phase, the framing this report keeps
coming back to: the research question was never "which model is best" — it was
**"under which constraints does each architecture become preferable."**

## 3. Experimental Environment

Every experiment in this program ran on the same machine: an **Apple M4** (4
performance + 6 efficiency cores, 10 total; 16GB unified memory; macOS 15.7 arm64).
This single-machine constraint is deliberate — the program studies constrained
compute, and one consistent, well-characterized machine is what let Week 3 pin the
thread-scaling cliff to an exact core count, and what lets every later week's
absolute numbers be compared against each other without a hardware confound.

**Primary model:** `Qwen/Qwen2.5-1.5B-Instruct`, used as the CPU-SLM baseline from
Week 1 through Week 10. **Primary engine:** llama.cpp/GGUF (CPU-only, Metal
disabled) from Week 2 onward, after Week 2 found it strictly faster and leaner than
a Python/Transformers baseline on every measured axis. **Stack by phase:** Python
(NumPy, pandas, matplotlib, SciPy) for all experimentation and analysis; Go (stdlib
`net/http`, `log/slog`, plus `prometheus/client_golang`) for the Week 7-8 production
services; Docker and `kind` (Kubernetes-in-Docker) for Week 9's orchestration
experiments; Prometheus and Grafana for observability.

**Methodology:** every week followed the same loop —
`QUESTION → HYPOTHESIS → EXPERIMENT → MEASUREMENT → ANALYSIS → INTERPRETATION → NEXT QUESTION`
— with a naive, falsifiable hypothesis stated *before* running each experiment (see
each week's `hypothesis.md`), raw data preserved untouched under
`results/raw/<experiment-id>/`, and quality comparisons (Weeks 5-6, 10) using
bootstrap 95% confidence intervals and paired effect sizes rather than point
estimates alone. Where an evaluation was needed, this program deliberately used
heuristic, non-LLM-judge scorers (`evaluation/metrics/scorers.py`) — exact/fuzzy
match, JSON field matching, lexical-overlap for summarization — rather than an
LLM-as-judge, per the roadmap's explicit brief.

## 4. Understanding CPU Inference

Week 1's from-scratch Python generation loop established the structure every later
week's numbers are built on: **prefill** (one large parallelizable pass over the
whole prompt, producing the first token) and **decode** (one small forward pass per
subsequent token, reusing the KV cache). This split — measured as TTFT and decode
tok/s — holds up under every stress test this program ever applied to it, across
three phases and two languages.

Week 2 found llama.cpp beats a Python/Transformers baseline on every axis under
matched conditions: 3.8x faster loading, 3.2x faster TTFT, 1.7x faster decode, 2.3x
lower peak memory (17.98 vs. 10.48 tok/s decode; 3769 vs. 8739 MB peak RSS). That gap
— not fully isolated from the accompanying F16-vs-fp32 precision difference, tracked
as an open question — was wide enough that "use a dedicated CPU inference engine"
became this program's first unambiguous, load-bearing conclusion, adopted for every
subsequent week.

## 5. Performance

What determines CPU inference performance, ranked by how much each factor actually
moved the numbers (Phase I):

1. **Thread oversubscription past the performance-core count** — the single biggest
   lever (2-4x swings). Week 2's coarse thread sweep found throughput peaking at 2
   threads and *collapsing*, not plateauing, past 4; Week 3's finer sweep confirmed
   the collapse point lines up almost exactly with this machine's 4 performance
   cores (`sysctl hw.perflevel0.physicalcpu` = 4). The practical rule adopted for
   the rest of the program: **use 2-4 threads, never more**, on a heterogeneous-core
   CPU.
2. **Engine choice** (§4) — 1.7-3.8x across every metric.
3. **Context length**, via two mechanisms: TTFT grows slightly super-linearly (self-
   attention's quadratic term becoming non-negligible at longer contexts), and decode
   speed *decreases* with context (21.8→13.6 tok/s from 156 to 4068 tokens) since every
   decode step attends over the entire KV cache — Week 1's "decode is roughly
   context-independent" read did not hold up under Week 3's tighter measurement.
4. **Concurrent system load** — real and monotonic, but with diminishing marginal
   effect past moderate contention.
5. **Numeric precision** (fp32 vs. F16) — present but not cleanly isolated from
   engine choice until Phase II's quantization work.

Week 3 also resolved a real methodological scare: Week 2's 25-run repeatability test
showed a strong throughput *decline* (r=-0.89), consistent with thermal throttling —
but repeating it at the actually-optimal thread count found the decline **disappeared
entirely** (and mildly *improved* instead), strong evidence the original decline was
a bad-thread-count artifact, not a generic thermal effect.

## 6. Quantization

Week 4 measured disk size, memory, load time, and decode speed across six
quantization levels (F16 down to Q3_K_M) of the same model, and falsified the naive
"fewer bits, less of everything except speed" intuition on three of four axes.
Disk size dropped cleanly and monotonically; nothing else did. **Q8_0 used more
peak memory than unquantized F16** (4108.5 vs. 3771.5 MB). Load time was *slowest in
the middle* of the precision range. Most strikingly, **Q4_K_M — not the least-
quantized format — was the fastest decoder tested** (57.15 tok/s, 36% faster than its
immediate neighbors), plausibly because llama.cpp's CPU backend can *repack* some
quantized block layouts for faster SIMD access and not others (`REPACK`, a supported
feature on this build) — a mechanism tied to specific block/group structure, not
raw bit-width.

## 7. Quality

Week 5 scored all six quantization levels against a purpose-built 100-example, 6-
category, bilingual (English/Italian) evaluation dataset (600 generations) and found
quality tracks quantization *even less* than the naive hypothesis predicted — in the
good direction, mostly. No level differed significantly from unquantized F16 down
through Q4_K_M (Q6_K nominally scored *highest* of all six, 0.731 vs. F16's 0.702);
only Q3_K_M showed a real, borderline decline, and was the only level to ever emit
invalid JSON (96.9% vs. 100% elsewhere). Joining this with Week 4's speed data
produced a clean two-point Pareto frontier — **Q6_K** (best quality) and **Q4_K_M**
(fastest, quality indistinguishable from F16) — with every other level, including F16
itself, dominated on both axes by one of these two.

The single most interesting quality failure wasn't a smooth decline: **7 of 8 Italian
summarization examples got answered in English at Q4_K_M specifically** (vs. 0-2 at
every other level) — a spike, not a trend, and a concrete reminder that multilingual
capability is both weaker to start and more fragile under quantization than
English-only numbers suggest. This evaluation pipeline itself needed a real
methodological lesson mid-week: a first pass caught three prompt/scorer bugs via a
self-scoring sanity check (every example scored against its own answer should
return 1.0) — fixed, and the entire 600-generation run repeated from scratch rather
than patched, a discipline carried into every later evaluation week.

## 8. Comparing Small Models

Week 6 repeated Weeks 4-5's full methodology across 5 models spanning 4 families
(Qwen2.5-0.5B/1.5B, Llama-3.2-1B, Gemma-2-2B-it, Phi-3.5-mini — 0.5B to 3.82B
params, all Q4_K_M) and found parameter count correlates strongly with both quality
(r=0.91) and speed (r=-0.94) — but with a clean, informative counterexample:
**Llama-3.2-1B-Instruct was strictly Pareto-dominated by Qwen2.5-0.5B-Instruct**, a
model under half its size that was both faster and higher-quality. Family reputation
didn't predict multilingual robustness either: **Qwen — the family most associated
with multilingual training — had the largest English/Italian quality gap of any
model tested (0.185), while Phi-3.5-mini had almost none (0.772 EN vs. 0.777 IT)**.
By tokens/sec per billion parameters, Qwen was the most CPU-friendly family tested
and Phi-3.5-mini the least — consistent with Phi-3.5-mini also being a 6-16x load-
time outlier. The throughline across both this week and Week 4: "more parameters" and
"fewer bits" are each, on their own, weak predictors of anything except disk size.

## 9. Concurrency

Week 8 built a hand-rolled Go load generator (goroutines, channels, a ticker for
rate limiting — deliberately not an off-the-shelf load-testing tool) and found the
Week 7 gateway's throughput plateaus almost immediately past concurrency 1, because
llama-server run with one processing slot (`-np 1`) serializes every request
regardless of client concurrency — confirmed directly via `requests_processing`
never exceeding 1. Latency scaled roughly linearly with concurrency up to the point
where a 30-second timeout started firing, and the error rate then collapsed sharply
(0% through concurrency 10, 72.5% by concurrency 80). The week's most dramatic single
result, though, was methodological: a deliberately undersized open-loop load test
showed **coordinated omission** in action — naive latency looked unremarkable (p50
3.4s) while latency measured from the *nominal* scheduled dispatch time (what a real
constant-rate user population would experience) was **77.5 seconds** at p50, a 20x+
gap that a naive load generator would have completely missed.

## 10. Production Infrastructure

Week 7 built `services/inference-gateway` — a single static Go binary (17.6MB
distroless Docker image, zero third-party web framework dependencies) implementing
`GET /health`, `GET /ready`, `GET /metrics`, and `POST /v1/generate`. Its one
substantive, later-validated design decision: **`/health` and `/ready` answer
different questions** — is the process alive, versus can it actually serve (i.e. is
llama-server reachable) — directly reused by Week 9's Kubernetes liveness/readiness
probes without modification. Verified end-to-end, not just unit-tested: a real
graceful shutdown correctly drained a live 3-second in-flight request before exiting
on SIGTERM.

## 11. Kubernetes

**What orchestration solved:** pod failure recovery. Deleting the only replica
mid-traffic caused a 30% error rate and a full 15-second outage while Kubernetes
rescheduled and reloaded the model; with 2 replicas, the same deletion caused only a
5.5% error rate, since the Service kept routing to the survivor the whole time.
Resource limits also behaved exactly as a production operator would want: CPU
limits produced a smooth, predictable slowdown (with a real qualitative collapse —
100% errors — at a quarter of a core), and memory limits behaved as a hard,
legible cliff (1.5Gi fully healthy, 1Gi immediately `OOMKilled`), not a fuzzy
gray zone.

**What it did not solve:** throughput scaling. Week 8's single-instance bottleneck
was expected to relax with more replicas — it did not. Under both fixed and
proportionally-scaled concurrency, aggregate throughput across 1-3 replicas stayed
flat or got *worse*. Direct evidence — `kubectl top pod` showing no CPU throttling,
but a ~10x spread in each pod's own request count — pointed at Kubernetes Services
load-balancing HTTP at the TCP-connection level (not per request) as a contributing
mechanism, compounded by all replicas sharing one physical machine's CPU/RAM on this
single-node test cluster. This is disclosed as this program's most significant
unresolved negative result, not smoothed over: a genuine multi-node cluster remains
the natural next experiment.

## 12. Failure Engineering

Week 9's five stress experiments, taken together, answer "how did the system break"
directly rather than by inference: **CPU starvation** collapses completely (not
gradually) below a quarter of the threads the model actually uses; **memory
pressure** kills the process outright, with a load-dependent safe boundary (idle
usage ~1.06GB, up to ~2.1GB observed under concurrent traffic); **pod deletion**
causes a full, measured outage window (15s) that redundancy directly shrinks (to a
5.5% error rate with one spare replica); and **load saturation** on Kubernetes
reproduces Week 8's native collapse curve almost exactly, confirming containerizing
the bottleneck doesn't change it. The one failure mode this phase could not fully
explain — throughput not scaling with replicas — is carried forward as this
program's most important open question, not resolved by assumption.

## 13. SLM vs LLM

Week 10 compared the CPU-SLM baseline against a larger self-hosted model
(Qwen2.5-7B-Instruct, same family, ~4.7x the parameters) on the same evaluation
pipeline, and found a real, paired-bootstrap-significant quality gain (+0.115,
Cohen's dz 0.37, 95% CI [0.057, 0.178] — excludes zero) concentrated in
summarization (+0.211) and reasoning (+0.167), the two categories most plausibly
helped by more capacity rather than pattern-matching a closed answer set. That gain
cost roughly 6x the decode speed, 12x the load time, and 2.8x the peak memory — a
real, quantified trade-off, not a free upgrade. A frontier remote API was
deliberately **not** called live this week (no live API spending or credential
sourcing) — its cost was estimated from a documented formula with illustrative,
not verified, price constants, and its other dimensions (privacy, operational
complexity, governance) were reasoned about qualitatively rather than measured. The
answer to "under what conditions is each approach preferable": CPU-SLM wins on
latency/throughput/memory/cost for closed-form tasks; the larger local model wins
on quality for summarization/reasoning-heavy or multilingual work when the hardware
budget allows; a frontier API's real, if unmeasured-here, advantages are near-zero
deployment complexity and someone else absorbing Weeks 7-9's entire operational
surface, at the cost of data leaving the premises and per-token billing.

## 14. Decision Framework

Week 11 didn't run new experiments — it reorganized Weeks 1-10's results around
FULL-ROADMAP.md's original decision flowchart (is the task constrained → test SLM /
consider a larger model → is quality sufficient → check system constraints or add
retrieval/fine-tuning/a larger model → evaluate privacy, latency, throughput, cost,
operations, governance), annotating every branch with the specific experiments,
results, limitations, and exceptions behind it. The clearest cross-cutting finding
from that exercise: **"is quality sufficient" turned out to depend far more on task
category than on which model or quantization level was used** — classification and
structured-output tasks reached ceiling-level quality at every size and
quantization level tested (0.5B through 7B, F16 through Q4_K_M); summarization was
consistently the hardest category for every system this program ever measured. Just
as important, the framework states plainly where this program has **no evidence at
all** — retrieval-augmented generation, fine-tuning, and governance/compliance were
never tested in any of the 12 weeks — rather than filling those branches with
generic, unverified advice.

## 15. Limitations

Aggregated across all 12 weeks, the honest boundaries of what this program's
evidence supports:

- **Single machine throughout.** Every absolute number (tok/s, memory, latency) is
  specific to this Apple M4; the *shapes* of the findings (thread cliffs, non-
  monotonic quantization, single-slot bottlenecks) are the more portable claims.
- **One model family predominant.** `Qwen2.5-1.5B-Instruct` is the through-line from
  Week 1 to Week 10; Week 6's 4-family, 5-model comparison and Week 10's same-family
  size comparison are each real but narrower slices, not a fully crossed
  model-family × size × quantization design.
- **Heuristic, non-LLM-judge scorers throughout** (Weeks 5, 6, 10) — real, known
  precision costs (e.g. summarization's lexical-overlap metric can't credit a
  correct paraphrase with low word overlap).
- **No live frontier API call anywhere in this program** (Week 10) — the frontier
  system's cost is a formula with illustrative constants; its quality/latency were
  never measured, only reasoned about.
- **Single-node Kubernetes cluster** (Week 9) — this program's most consequential
  limitation, since it directly confounds Week 9.5's headline negative result
  (horizontal scaling not improving throughput) with "this test setup couldn't show
  it," not necessarily "it doesn't work."
- **No retrieval-augmented generation or fine-tuning experiments anywhere** — an
  entire class of quality intervention this program never tested, disclosed
  explicitly in Week 11 rather than assumed away.
- **No root-level thermal instrumentation** (`powermetrics` needs `sudo`, never
  used) — several plausible thermal explanations (Weeks 2-3, 10) remain
  circumstantial, not confirmed via direct clock-frequency data.

## 16. Future Research

The full, itemized register — 39 open questions accumulated across 12 weeks, most
still open — is
[`docs/methodology/open-questions.md`](../../docs/methodology/open-questions.md).
The handful most consequential for anyone extending this work:

- **Does a genuine multi-node Kubernetes cluster resolve Week 9.5's horizontal-
  scaling result?** This program's single most important unresolved question — it
  determines whether "add replicas" is actually a viable throughput lever for this
  architecture at all.
- **Would a live frontier-API run, on this program's own dataset and scorers, put
  frontier quality above, at, or only modestly above a 7B open model's?** Directly
  extends Week 10/11 from "documented, not measured" to measured.
- **Would retrieval-augmented generation or fine-tuning close the summarization/
  reasoning quality gap more cost-effectively than a larger model?** A genuine
  evidence gap this program never touched.
- **Does `--parallel N > 1` (llama-server's continuous batching, flagged since
  Week 8) raise the single-instance throughput ceiling found repeatedly in Weeks
  8-9?** Untested, and directly relevant to whether Week 9.5's result is really
  about Kubernetes or about llama-server's own configuration.
- **Would a harder, larger evaluation dataset** (beyond this program's 100-example
  v1) **tighten the several "not statistically significant" or "n=5, descriptive
  only" findings** (Weeks 5, 6) enough to resolve them either way?

## 17. Conclusion

Twelve weeks, four phases, and one recurring meta-lesson: **almost every naive
"more X means more Y, proportionally" intuition this program tested turned out to be
false, and the specific way it was false was usually more informative than the
intuition would have been if it had simply held.** Fewer bits didn't mean uniformly
faster or leaner (Week 4); more parameters didn't mean uniformly better or more
efficient (Week 6); more threads didn't mean more throughput past a hard core-count
boundary (Week 2-3); more replicas didn't mean more aggregate throughput on this
program's own test infrastructure (Week 9). Each of these required an actual
measurement to discover, not a plausible guess — which is the entire methodological
point of running an experiment instead of reasoning from first principles about
systems this complex.

Within that, a genuinely useful, evidence-backed picture of CPU-first small language
models did emerge. They are unambiguously viable for closed-form, classification-
like tasks, at very low cost and very high throughput ceilings per instance,
regardless of quantization level down to Q4_K_M or model size down to 0.5B. They are
a real, quantified trade-off — not a clear win or loss — for summarization- and
reasoning-heavy or multilingual work, where a larger self-hosted model or a
frontier API each buy something specific (quality, or zero operational overhead)
that the smallest CPU-SLM configuration doesn't provide. And the operational
reality of running any of this reliably — health/readiness distinctions, load
testing, observability, orchestration, failure recovery — turned out to be at least
as large an engineering surface as the model itself, three full weeks of this
twelve-week program on its own.

The question this program set out to answer was never "which model is best." It was
"under which constraints does each architecture become preferable" — and the honest
answer, after twelve weeks of measurement, is: it depends on exactly the dimensions
this report walked through, each with real numbers behind it, and on being willing
to actually measure rather than assume which side of each trade-off a given task
falls on.
