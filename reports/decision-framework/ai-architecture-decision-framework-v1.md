# AI Architecture Decision Framework

### Week 11 Deliverable — Efficient AI Systems

**Objective:** transform 10 weeks of empirical results into an engineering decision
framework — not a leaderboard.

> The research question is not "which model is best?" The question is "under which
> constraints does each architecture become preferable?"

Every branch below is annotated with **relevant experiments**, **observed results**,
**limitations**, and **exceptions** — per FULL-ROADMAP.md's explicit requirement that
this framework be based on experimental evidence, not general AI industry advice.
Where this program has no evidence for a branch, that's stated plainly rather than
filled in with a plausible-sounding guess.

- [Phase I report](../benchmarks/cpu-inference-performance-report-v1.md) (Weeks 1-3) ·
  [Phase II report](../benchmarks/cpu-slm-benchmark-report-v1.md) (Weeks 4-6) ·
  [Phase III report](../benchmarks/cpu-first-service-report-v1.md) (Weeks 7-9) ·
  [Week 10 — SLM vs LLM](../../experiments/10-slm-vs-llm/README.md)
- [Open questions tracker](../../docs/methodology/open-questions.md)

---

## The Framework

```text
                              NEW AI TASK
                                   │
                                   ▼
                    IS THE TASK CONSTRAINED?
              (compute, latency, concurrency, cost,
               privacy/data-residency — see §1)
                                   │
                  ┌────────────────┴────────────────┐
                 YES                                 NO
                  │                                   │
                  ▼                                   ▼
             TEST SLM (§2)                   CONSIDER LARGER MODEL (§3)
                  │
                  ▼
        IS QUALITY SUFFICIENT? (§4)
                  │
      ┌───────────┴───────────┐
     YES                       NO
      │                         │
      ▼                         ▼
CHECK SYSTEM              ADD RETRIEVAL, FINE-TUNING,
CONSTRAINTS                  OR LARGER MODEL (§5)
      │
      ▼
   EVALUATE (§6)
   ├── PRIVACY
   ├── LATENCY
   ├── THROUGHPUT
   ├── COST
   ├── OPERATIONS
   └── GOVERNANCE
```

This is FULL-ROADMAP.md's original flowchart, unchanged in shape — ten weeks of
evidence refined *what each box means on this program's data*, not the shape of the
decision tree itself. Where evidence pushed back on the naive reading of a box (e.g.
"is quality sufficient" turning out to be more about *task category* than model
choice), that's called out explicitly in the relevant section.

---

## §1 — Is the task constrained?

**What "constrained" means, made concrete by this program's data:** not just "no
GPU available" — the sharper, evidenced version is "what's the expected concurrent
request volume, latency tolerance, and data-residency requirement?"

- **Relevant experiments:** Weeks 1-3 (CPU inference viability), Week 8 (load
  saturation), Week 9 (Kubernetes scaling).
- **Observed results:** CPU inference is entirely viable for a single-user or
  low-concurrency workload (Week 1-3: a 1.5B model decodes at 20-60+ tok/s
  single-threaded-optimal on a 2024 Apple Silicon CPU) — but a single self-hosted
  instance serializes requests (Week 8: `requests_processing` never exceeded 1),
  producing a hard throughput ceiling around 0.6-0.8 req/s regardless of client
  concurrency, with error rates climbing past 20% once concurrency exceeds ~10-20
  (Week 8/9, consistent within noise on both native and Kubernetes deployments).
- **Limitations:** all measured on one machine (Apple M4, 10 cores, 2 used), one
  model family predominant (Qwen2.5). Absolute throughput numbers are
  machine-specific; the *shape* of the constraint (single instance = hard ceiling)
  is the more portable finding.
- **Exceptions:** Week 9.5 found horizontal scaling (adding replicas) did **not**
  cleanly relax this constraint on a single-node test cluster — throughput stayed
  flat or worsened across 1-3 replicas, with evidence pointing at Kubernetes
  Service connection-level load balancing and single-node resource sharing as
  contributing causes. **"Add more replicas" is not a verified fix for the
  concurrency constraint on this program's own evidence** — a genuine multi-node
  cluster remains untested (tracked as an open question).

**Reading this box:** if the task genuinely needs high concurrent throughput on
constrained/self-hosted hardware, that itself is evidence *against* naively trusting
horizontal scaling to solve it — plan capacity per-instance, not per-fleet, until a
multi-node test says otherwise.

## §2 — Test SLM (if constrained)

- **Relevant experiments:** Weeks 4-6 (quantization/model comparison), Week 10
  (SLM vs. larger local model).
- **Observed results:** a 1.5B model at Q4_K_M scores 0.694 (95% CI [0.614, 0.772])
  on this program's 100-example, 6-category evaluation dataset (Week 5/10) — not
  uniformly across categories: **classification and structured-output tasks reach
  ceiling-level quality even at 0.5-1.5B** (Week 6/10: classification saturates at
  1.000 for every model tested, 0.5B through 7B), while **summarization and
  reasoning show the largest model-size sensitivity** (Week 10: +0.211 and +0.167
  respectively moving to a 7B model in the same family). Quantization down to Q4_K_M
  costs no statistically significant quality versus unquantized (Week 5); Q3_K_M is
  the first level to show a real, if borderline, decline.
- **Limitations:** one evaluation dataset (100 examples), heuristic non-LLM-judge
  scorers (exact/fuzzy match, JSON field matching, lexical-overlap for
  summarization) — not a judge of true semantic quality, and specifically weak at
  crediting correct paraphrases with low word overlap.
- **Exceptions:** multilingual (Italian) quality is both weaker to start and more
  fragile under quantization than English-only numbers suggest (Week 5); this
  fragility does **not** track family reputation (Week 6: Qwen, the family most
  associated with multilingual training, had the *largest* EN-IT gap of any model
  tested; Phi-3.5-mini had almost none). **If the task is closed-form/classification-
  like, an SLM is very likely sufficient regardless of language. If it's
  summarization/reasoning-heavy and multilingual, test explicitly — don't assume.**

## §3 — Consider larger model (if not constrained)

- **Relevant experiments:** Week 6 (model comparison up to 3.82B), Week 10
  (1.5B vs. 7B, same family).
- **Observed results:** moving from 1.5B to 7B (same family) produced a real,
  paired-bootstrap-significant quality gain (+0.115, Cohen's dz 0.37, 95% CI
  [0.057, 0.178]) at a real cost (~6x slower decode, ~12x slower load, ~2.8x peak
  memory — Week 10). Parameter count correlated strongly with quality (r=0.91) and
  speed (r=-0.94) across 5 *different* models spanning 4 families (Week 6) — but
  n=5 is descriptive, not confirmatory, and had a clean counterexample
  (Llama-3.2-1B-Instruct, Pareto-dominated by a model under half its size).
- **Limitations:** Week 10's clean size comparison was same-family only (Qwen2.5);
  it isolates parameter count but says nothing about whether a different
  architecture at 7B shows the same gain or the same category pattern. No genuinely
  larger-than-7B model was tested in this program.
- **Exceptions:** bigger is not automatically better or more efficient —
  Llama-3.2-1B was both slower *and* lower-quality than a model under half its size
  (Week 6); Phi-3.5-mini's tokens/sec-per-billion-parameters was the *worst* of 5
  models tested despite being mid-pack in raw size (Week 6). **"Consider a larger
  model" should mean "test specific candidates," not "assume more parameters helps
  monotonically" — this program falsified that assumption at least twice.**

## §4 — Is quality sufficient?

This is the box where evidence most reshaped the naive reading: "sufficient" turned
out to depend far more on **task category** than on which model or quantization level
is used.

- **Relevant experiments:** Week 5 (quantization vs. quality), Week 6 (model
  comparison quality), Week 10 (SLM vs. larger local, quality by category).
- **Observed results:** every quality comparison this program ran (across 6
  quantization levels, 5 models, and 2 model sizes) found the **same** category
  ordering: classification and structured-output near-ceiling regardless of model
  size/quantization; summarization consistently the *hardest* category for every
  system tested (0.275-0.486 across all of Weeks 5/6/10's summarization scores,
  never approaching the ~0.7-1.0 range other categories reach); reasoning and
  information-extraction in between, with the most size/quantization sensitivity.
- **Limitations:** this ordering is specific to this program's one dataset and its
  specific scorers (e.g. summarization's lexical-overlap metric may itself
  understate quality for good paraphrases — see §2's limitations).
- **Exceptions:** JSON validity (a hard, binary form of "quality" for structured
  tasks) held at 100% for every quantization level down to Q4_K_M and every model
  tested in Week 6/10 — it only broke down at the most aggressive quantization
  tested (Q3_K_M, 96.9% — Week 5). **If "quality sufficient" specifically means
  "produces syntactically valid structured output," that's a much lower bar to
  clear than general task quality, and this program's evidence says nearly every
  configuration tested clears it.**

## §5 — Add retrieval, fine-tuning, or larger model (if quality insufficient)

**This branch is the framework's most significant evidence gap.** This program never
ran a retrieval-augmented-generation or fine-tuning experiment — every quality
intervention tested was "change the model or its quantization" (§2, §3). This is
disclosed here rather than papered over with generic advice:

- **Relevant experiments:** none directly. Week 10's "larger model" comparison is the
  only *quality-improving intervention* actually measured.
- **Observed results:** larger model, specifically, produces a real, quantified
  quality gain in this program's own data (§3) — the only one of the three options
  in this box that this program can speak to with evidence.
- **Limitations:** retrieval and fine-tuning are entirely untested here — this
  framework cannot honestly rank them against "use a larger model" or against each
  other. Anyone using this framework to make a real decision should treat this box
  as "known gap, not resolved by this program" rather than infer an answer.
- **Exceptions:** none available — there's no evidence to state an exception to.

## §6 — Check system constraints → Evaluate

### Privacy

- **Relevant experiments:** Week 10 (system comparison framing).
- **Observed results:** both self-hosted systems (CPU-SLM, Larger-Local) keep data
  entirely on-premises by construction; a remote frontier API sends prompts to a
  third party — a categorical, not incremental, difference that doesn't require a
  live benchmark to state confidently.
- **Limitations:** this program never operated under an actual regulatory or
  contractual privacy requirement — the categorical claim is architectural, not
  validated against a specific compliance framework.
- **Exceptions:** none identified.

### Latency

- **Relevant experiments:** Week 1 (TTFT/decode split), Week 3 (thread count,
  context length), Week 8 (concurrent load), Week 10 (SLM vs. larger model).
- **Observed results:** single-request latency is low and predictable for a small
  model (CPU-SLM: ~50.7 tok/s decode, Week 10) but degrades sharply under
  concurrency (Week 8: p50 latency scales roughly linearly with concurrent clients,
  from ~1.3s at concurrency 1 to ~20-26s by concurrency 10-20) and drops
  substantially for a larger model even at concurrency 1 (Larger-Local: 8.5 tok/s,
  ~6x slower — Week 10).
- **Limitations:** all single-machine; a dedicated multi-core server or GPU-backed
  deployment would shift these numbers considerably.
- **Exceptions:** context length also degrades decode speed independent of
  concurrency (Week 3: 21.8→13.6 tok/s from 156 to 4068 prompt tokens) — a
  long-context task can look latency-constrained even at concurrency 1.

### Throughput

- **Relevant experiments:** Week 8 (saturation sweep), Week 9 (Kubernetes,
  horizontal scaling).
- **Observed results:** a single self-hosted instance has a hard throughput
  ceiling (~0.6-0.8 req/s in this program's tests) that does not improve by adding
  client concurrency, and — on this program's own single-node evidence — does not
  cleanly improve by adding replicas either (Week 9.5).
- **Limitations:** Week 9.5's negative result is specifically confounded by a
  single-node test cluster (see §1) — a genuine multi-node deployment remains
  untested.
- **Exceptions:** `--parallel N > 1` (llama-server's own continuous batching) was
  flagged in Week 8 as a plausible lever never actually tested in this program —
  an open question, not a validated exception.

### Cost

- **Relevant experiments:** Week 10 (cost model).
- **Observed results:** a formula-based comparison (measured local generation time
  × an hourly compute-cost estimate; measured token counts × per-token API pricing)
  found self-hosted cost scales with model size (CPU-SLM cheaper than Larger-Local,
  roughly matching their ~4.3x generation-time ratio) — but the frontier-API
  comparison used **illustrative, not verified, price constants**.
- **Limitations:** this is this framework's most explicitly caveated dimension —
  Week 10 built a real formula with real measured inputs, but the price constants
  must be replaced with actual current provider pricing before the dollar
  comparison is decision-grade for any specific real choice.
- **Exceptions:** cost-per-request isn't the only relevant cost lens — self-hosting
  also carries the fixed cost of the entire operational stack (see below), which a
  per-request formula doesn't capture.

### Operations

- **Relevant experiments:** Weeks 7-9 (gateway, load testing/observability,
  Kubernetes).
- **Observed results:** running a self-hosted model reliably required building an
  entire stack this program spent three full weeks on: a request-handling gateway
  with health/readiness distinctions and graceful shutdown (Week 7), load testing
  and a Prometheus/Grafana observability stack (Week 8), and container
  orchestration with resource limits, failure recovery, and horizontal scaling
  attempts (Week 9). A frontier API requires essentially none of this — an HTTP
  client and a key.
- **Limitations:** this stack was built once and is reused, not rebuilt per
  deployment — the *marginal* operational cost of an additional self-hosted
  deployment on already-built infrastructure is much lower than these three weeks
  suggest in isolation.
- **Exceptions:** none identified — this is the dimension where the self-hosted vs.
  API trade-off is least ambiguous in this program's own experience.

### Governance

- **Relevant experiments:** none directly tested in this program.
- **Observed results:** none measured.
- **Limitations:** this program never operated under a real governance/compliance
  regime (model versioning audits, data retention policy, vendor risk assessment) —
  this dimension is stated qualitatively only: self-hosting gives full control over
  exact model version and update timing; a remote API makes you dependent on the
  provider's own deprecation schedule and policy changes.
- **Exceptions:** none available — flagged as an evidence gap, matching §5's
  honesty about untested branches.

---

## Summary: Where Each Architecture Wins, By This Program's Evidence

| architecture | wins when | evidence |
|---|---|---|
| **CPU-SLM** | task is closed-form (classification, structured output), low-to-moderate concurrency, cost-sensitive, hardware-constrained | Weeks 4-6, 8, 10 |
| **Larger self-hosted model** | task is summarization/reasoning-heavy, multilingual, self-hosting is required (privacy/no vendor dependency), and the ~6x latency cost is acceptable | Week 10 |
| **Frontier API** | quality ceiling matters more than cost-per-request, zero operational/deployment complexity is wanted, and no privacy/self-hosting requirement exists | Reasoned qualitatively (§6) — **not measured live this program** |

## Field Note #4

This framework is the basis for Field Note #4, "When Is a Small Language Model
Enough?" (published externally per `reports/field-notes/README.md`'s schedule) — the
field note is the accessible narrative version of this document's evidence-annotated
framework.

## Reproducibility

Every claim above traces to a specific week's README and its underlying
`results/raw/<experiment-id>/` data — see the phase reports linked at the top for the
full evidence trail. This document adds no new experiments; it only re-reads and
re-organizes Weeks 1-10's existing, already-reproducible results around decision
points rather than around weeks.
