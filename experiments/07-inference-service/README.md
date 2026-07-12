# Week 7 — Go Inference Gateway

**Status:** ✅ Complete.

Unlike Weeks 1-6, this week isn't a research experiment with a hypothesis and a
measured dataset — it's a production build (Phase III, per FULL-ROADMAP.md), so it
doesn't follow the standard `hypothesis.md` / `config/` / `analysis/` structure used
elsewhere in `experiments/`. The actual deliverables live in:

- [`services/inference-gateway/`](../../services/inference-gateway/) — the Go
  service itself: source, tests, and API/configuration documentation in its own
  README.
- [`docs/architecture/inference-gateway.md`](../../docs/architecture/inference-gateway.md) —
  request-flow diagram, health-vs-readiness design, error mapping.
- [`docs/decisions/0001-go-for-inference-gateway.md`](../../docs/decisions/0001-go-for-inference-gateway.md) —
  why Go, why no web framework.
- [`infrastructure/docker/`](../../infrastructure/docker/) — Dockerfile and
  docker-compose for local reproducibility.

This directory exists only so `experiments/` has a placeholder matching
FULL-ROADMAP.md's week numbering; it intentionally holds no code of its own.
