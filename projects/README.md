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

**v0.1.0 tagged** — CI on every push, cross-compiled release binaries
(darwin/linux, amd64/arm64), builds/tests/runs end-to-end against
llama.cpp, OpenAI-compatible, and Ollama servers.

Generic HTTP load testers (wrk, k6, locust) don't model LLM serving semantics —
no native time-to-first-token or inter-token-latency reporting, no streaming
awareness — and, as
[Week 8](../experiments/08-load-testing/README.md) found firsthand, are easy to
get *coordinated omission* wrong in ways that specifically hide the tail latency
that matters most under load.

## [`quantscope/`](quantscope/) — GGUF quantization / CPU-kernel-fit profiler

**Core implemented (M0-M2, M3 partial), pre-release** — installable, all
subcommands (`bench`, `predict`, `quantize`, `report`, `formats`, `cpu-info`)
work and are tested end-to-end.

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
