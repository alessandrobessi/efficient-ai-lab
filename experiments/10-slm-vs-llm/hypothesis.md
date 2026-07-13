# Hypotheses — Week 10: SLM vs LLM

## The central principle (from FULL-ROADMAP.md, not a hypothesis to test)

> The research question is not "which model is best?" The question is "under which
> constraints does each architecture become preferable?"

Every hypothesis below is written to be falsifiable in that spirit — the goal is
identifying the conditions under which each system wins, not crowning one system
"best" outright.

## Systems compared

1. **CPU-SLM** — `Qwen2.5-1.5B-Instruct`, Q4_K_M — this program's baseline since
   Week 2, self-hosted on this machine's CPU throughout.
2. **Larger-Local** — `Qwen2.5-7B-Instruct`, Q4_K_M — same family, ~4.7x the
   parameters, still self-hosted and CPU-only (feasible on this machine per Week 6's
   established methodology).
3. **Frontier-API** — a remote frontier model, accessed via API. **Deliberately not
   called live this week** — see "Why the frontier system isn't measured live" below.
   Its cost, privacy, operational/deployment complexity, observability, and failure
   modes are assessed from documented characteristics rather than a live benchmark
   run; its quality/latency/throughput are explicitly not claimed.

## Central hypothesis

> The Larger-Local system scores higher on quality than the CPU-SLM (more parameters,
> same family, same training approach), at a real, measurable cost in latency,
> throughput, and memory footprint — quantifying the actual size of that trade-off on
> this specific machine and dataset, rather than assuming a generic "bigger is
> better" story applies uniformly across task categories.

Falsifiable by: the Larger-Local system not outscoring the CPU-SLM (echoing Week 6's
finding that parameter count doesn't always predict quality even within a
comparison set), or the latency/memory cost being small enough that the trade-off
isn't actually meaningful in practice.

## Sub-hypotheses (roadmap's measurement dimensions)

- **Quality:** Larger-Local ≥ CPU-SLM overall, but not necessarily in every category —
  Week 6 already found category-level quality doesn't move in lockstep with
  parameter count.
- **Latency / throughput:** CPU-SLM should be substantially faster (lower TTFT,
  higher decode tok/s) — 7B has ~4.7x the compute per token of 1.5B, and this
  machine's fixed thread budget (2, held constant per prior weeks) doesn't scale to
  match.
- **Cost:** self-hosted compute cost scales with generation time — so Larger-Local
  should cost meaningfully more per request than CPU-SLM on the same hardware.
  Whether either self-hosted system is cheaper *or* more expensive than a frontier
  API depends entirely on the (currently only illustrative, not verified) price
  constants — this is explicitly a "check the real numbers for your situation"
  question, not one this week's data alone can answer.
- **Privacy:** both self-hosted systems keep data entirely on-premises; a frontier
  API sends prompts to a third party — a categorical difference, not a matter of
  degree.
- **Operational / deployment complexity:** both self-hosted systems require the full
  stack Weeks 2-9 built (quantized model management, an inference server, a gateway,
  observability, orchestration); a frontier API requires none of that — an HTTP
  client and a key.
- **Observability / failure modes:** self-hosted systems get the full metrics access
  Weeks 8-9 built (Prometheus, per-request tracing, resource limits); a frontier API
  gives only what the provider chooses to expose, with different failure modes
  (rate limits, provider outages, content filtering) than Week 9's (OOM, CPU
  throttling, pod eviction).

## Why the frontier system isn't measured live this week

Calling a real frontier API costs real money and requires real credentials this
session does not (and should not) source for itself. Rather than skip the
comparison entirely or fabricate plausible-looking numbers, this week builds the
full comparison framework — dataset, scorers, cost model — so a frontier system
could be added by pointing `evaluation/runners/` at a new backend and re-running
`analysis/`, while being explicit in every artifact (config comments, cost model
docstring, README) about exactly which numbers are measured versus documented from
public information versus templated for future use.
