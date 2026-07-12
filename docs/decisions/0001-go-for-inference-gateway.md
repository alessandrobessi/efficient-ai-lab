# ADR 0001: Go for the Inference Gateway

## Context

Weeks 1-6 are entirely Python (experimentation, evaluation, analysis — see root
`README.md` §4's technology stack). FULL-ROADMAP.md's Week 7 calls for a "Go
Inference Gateway" specifically, not just "a production HTTP service," and the
program's stated stack table already lists Go for "production service, load
generation." This ADR records why, now that it's actually being built.

## Decision

The Week 7 inference gateway (`services/inference-gateway/`) is written in Go, using
only the standard library plus `prometheus/client_golang` (needed for the roadmap's
explicit Prometheus metrics requirement) — no web framework, no router library, no
ORM.

## Consequences

**Why Go over Python for this specific service:**

- **A single static binary.** The Dockerfile's final stage is `gcr.io/distroless/static`
  with nothing but the compiled binary — no interpreter, no `pip install`, no base OS
  package manager. The resulting image is 17.6MB. A Python equivalent would need a
  Python runtime plus dependencies in the final image, an order of magnitude larger,
  working against this project's CPU-first, minimal-footprint theme.
- **Goroutines make Week 8's load generator (also Go) and this gateway share an
  idiom for concurrency** — both need to handle many concurrent HTTP calls
  efficiently; Go's goroutines/channels are a more natural fit here than Python's
  asyncio for a program that is otherwise not Python-idiomatic in its concurrency
  story (Weeks 1-6 are single-threaded measurement scripts, not concurrent servers).
- **`net/http` plus `log/slog` (both stdlib, Go 1.21+/1.22+) cover request routing,
  structured logging, and graceful shutdown without a single third-party dependency**
  beyond the metrics library — consistent with root `README.md` §17's scope-control
  principle ("nothing added to this stack unless it lets us answer a research question
  we couldn't otherwise answer").

**Why not a framework (gin, echo, chi, ...) on top of `net/http`:**

Go 1.22 added method-aware routing patterns to the standard `http.ServeMux`
(`mux.HandleFunc("POST /v1/generate", ...)`), which covers everything this gateway's
4 routes need. A router library would add a dependency and an abstraction layer for
zero functional gain at this route count — revisit only if route count or matching
complexity grows meaningfully (e.g. path parameters, versioned route groups).

**Tradeoffs accepted:**

- **Two language ecosystems in one repo** (Python for experiments/evaluation, Go for
  services) means no code sharing between them — the Go gateway re-implements nothing
  from `evaluation/`, it's a from-scratch HTTP service that happens to call the same
  llama-server the Python evaluation pipeline calls.
- **Go's ecosystem for ML/data analysis is much weaker than Python's** — but that's
  not what this service does; it's a thin, typed, concurrent proxy, which is exactly
  Go's strength and exactly not what Weeks 1-6's statistical analysis needed.
