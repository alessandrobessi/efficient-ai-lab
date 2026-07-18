# Follow-On Projects

The [12-week efficient-ai-lab program](../README.md) was research: controlled
experiments, measured and written up. Two of its findings kept pointing at the
same conclusion — the *tool* needed to act on the finding doesn't exist yet, and
building it would be useful independently of this program. That's a different
kind of work than an experiment, so it lives here rather than under
`experiments/`.

Each directory below has a `README.md` (the pitch: what problem it solves, for
whom, and how to run it) and a `ROADMAP.md` (the full phased development plan:
architecture, milestones, testing strategy).

## [`llmpace/`](llmpace/) — coordinated-omission-safe LLM load testing

**[`llmpace/v0.1.2`](https://github.com/alessandrobessi/efficient-ai-lab/tree/llmpace/v0.1.2) tagged** (scheduled/corrected TTFT
for both latency and TTFT, a bounded and correctly-defaulted client-side
backlog split into peak-in-flight/peak-waiting, distinct offered/admitted/
completed request rates, per-category failure tracking, chunks-not-tokens
naming, and a genuinely CPU-only saturation benchmark with charts — see its
[ROADMAP.md](llmpace/ROADMAP.md#v012-fixes-from-a-second-external-review)).
CI on every push; cross-compiled darwin/linux amd64/arm64 binaries are
published on the
[releases page](https://github.com/alessandrobessi/efficient-ai-lab/releases).

Generic HTTP load testers (wrk, k6, locust) don't model LLM serving semantics —
no native time-to-first-token or inter-token-latency reporting, no streaming
awareness — and, as
[Week 8](../experiments/08-load-testing/README.md) found firsthand, are easy to
get *coordinated omission* wrong in ways that specifically hide the tail latency
that matters most under load.

## [`quantscope/`](quantscope/) — GGUF quantization / CPU-kernel-fit profiler

**[`quantscope/v0.2.1`](https://github.com/alessandrobessi/efficient-ai-lab/tree/quantscope/v0.2.1) tagged**
— CI on every push, installable, CPU inference always forced (`-ngl 0`) on
every llama.cpp invocation, all subcommands (`bench`, `estimate-size`,
`quantize`, `report`, `recommend`, `formats`, `cpu-info`) work and are
tested end-to-end. A real
[8-format benchmark](quantscope/benchmarks/2026-07-15-qwen2.5-0.5b-cpu/),
rerun with a randomized multi-round architecture (instead of a confounded
sequential one), found round-to-round variance large enough on a small
model that no single format's speed edge is well-supported by the data —
an honest result the previous, order-confounded run's clean-looking
numbers had been masking.

[Week 4](../experiments/04-quantization/README.md) and
[Week 6](../experiments/06-model-comparison/README.md) repeatedly found that
quantization format and model choice interact with SIMD/`REPACK` kernel fit in
ways that don't correlate with bit-width or parameter count — discovered only
through many hours of manual benchmarking, with the mechanism never actually
profiled.

## Relationship to the research program

Neither tool is required reading to understand the 12-week program's findings —
the program's own conclusions stand on their own in `experiments/` and
`reports/`. These are what came *after*: a judgment call that the underlying
problem is general enough, and the fix concrete enough, to be worth building as
standalone open-source software rather than leaving as a documented research
finding.
