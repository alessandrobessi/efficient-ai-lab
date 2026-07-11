# Efficient AI Systems

### A 12-Week Research and Engineering Program on CPU-First Small Language Models

> How much useful AI can we build under strict compute constraints?

This repository is a research monorepo. It documents a structured, experiment-driven
investigation into running, measuring, evaluating, and deploying **Small Language Models
(SLMs)** on **CPU-only infrastructure** — from raw transformer inference up through a
production-style Go service running on Kubernetes with full observability.

The full program design, week-by-week curriculum, methodology, and publication plan
lives in [`FULL-ROADMAP.md`](FULL-ROADMAP.md). This README covers how the repository is
organized, how to set it up, and how to run what exists today.

---

## 1. Research Question

The program does not try to prove that small models are always better than large ones.
It investigates a more useful question:

> Under which technical, economic, operational, and organizational constraints does a
> smaller and more efficient AI system become the better engineering choice?

Every week produces at least one controlled experiment. Reading and studying alone do
not count as progress — see [Guiding Principles](#3-guiding-principles) below.

---

## 2. Program Structure

The program runs in four phases across 12 weeks.

| Phase | Weeks | Central Question |
|---|---|---|
| **I — Understand CPU Inference** | 1–3 | What actually happens when an SLM runs on a CPU? |
| **II — Measure Models and Quantization** | 4–6 | What tradeoffs emerge when SLMs are compressed and compared? |
| **III — Build a Production System** | 7–9 | How do we turn a local model into a reliable, observable service? |
| **IV — Synthesize and Decide** | 10–12 | When is an SLM actually the right architectural choice? |

| Week | Topic | Status |
|---|---|---|
| 1 | Transformer Inference Fundamentals | ✅ Complete |
| 2 | llama.cpp and GGUF | ✅ Complete |
| 3 | CPU Performance Engineering | ⬜ Not started |
| 4 | Quantization Fundamentals | ⬜ Not started |
| 5 | Quantization vs Quality | ⬜ Not started |
| 6 | Small Model Comparison | ⬜ Not started |
| 7 | Go Inference Gateway | ⬜ Not started |
| 8 | Load Testing and Observability | ⬜ Not started |
| 9 | Kubernetes and Failure Engineering | ⬜ Not started |
| 10 | SLM vs LLM | ⬜ Not started |
| 11 | AI Architecture Decision Framework | ⬜ Not started |
| 12 | Final Synthesis | ⬜ Not started |

Each experiment week lives under [`experiments/`](experiments/) and follows the same
internal structure — see [§6](#6-experiment-structure) below.

---

## 3. Guiding Principles

- **Experiment first.** Every theoretical concept should produce a runnable experiment.
- **Measure everything.** No "feels faster" — only numbers with units, ideally with
  variance (e.g. *"median 14.2 tok/s, p95 TTFT 340 ms, n=20"*).
- **Depth over breadth.** A handful of carefully controlled models/quantizations beats a
  sprawling, uncontrolled matrix.
- **Separate observation from interpretation.** Every report distinguishes what happened,
  why it might have happened, what evidence supports that, and what else could explain it.
- **Reproducibility is not optional.** Every experiment records hardware, software
  versions, model, configuration, and raw data — see [§8](#8-experiment-metadata-standard).

Full rationale for each principle is in `FULL-ROADMAP.md` §3.

---

## 4. Technology Stack

| Layer | Tools |
|---|---|
| Experimentation, evaluation, analysis | Python, NumPy, pandas, matplotlib, SciPy, Hugging Face Transformers, PyTorch |
| Inference engine | [llama.cpp](https://github.com/ggerganov/llama.cpp), GGUF |
| Production service, load generation | Go |
| Containerization | Docker |
| Orchestration | Kubernetes / k3s |
| Observability | Prometheus, Grafana |

Nothing is added to this stack unless it lets us answer a research question we couldn't
otherwise answer (see `FULL-ROADMAP.md` §18, Scope Control).

---

## 5. Repository Layout

```text
efficient-ai-lab/
├── README.md                    this file
├── FULL-ROADMAP.md              full 12-week program design and methodology
├── pyproject.toml                Python project & dependencies (managed with uv)
├── Makefile                      common commands (setup, run, lint, test)
│
├── docs/                         architecture notes, methodology, hardware specs, ADRs
├── experiments/                  one directory per week, each self-contained (§6)
├── evaluation/                   shared model-evaluation framework (from Week 4 onward)
├── services/                     Go inference gateway and load generator (from Week 7)
├── infrastructure/                Docker, Kubernetes, Prometheus, Grafana configs
├── scripts/                      repo-wide utility scripts
├── results/                      canonical experiment data pipeline (§7)
│   ├── raw/<experiment-id>/       untouched experiment output
│   ├── processed/<experiment-id>/ cleaned/aggregated data
│   └── figures/<experiment-id>/   generated plots
├── reports/                       longer-form writing
│   ├── field-notes/               short public write-ups per experiment
│   ├── benchmarks/                phase-level benchmark reports
│   └── final/                     the Week 12 flagship report
└── models/                        model download docs (binaries are never committed)
```

## 6. Experiment Structure

Every experiment directory (`experiments/NN-name/`) follows the same shape:

```text
experiments/NN-name/
├── README.md        research question, hypothesis, setup, results, interpretation
├── hypothesis.md     the specific, falsifiable hypothesis for this experiment
├── config/            model/run configuration (YAML)
├── scripts/           code that runs the experiment and writes raw data
├── data/              fixed inputs (prompts, datasets) used by the scripts
└── analysis/          code that turns raw data into statistics and figures
```

Raw and processed data and figures are written to the repository-level
`results/{raw,processed,figures}/<experiment-id>/` directories (§7), not duplicated
inside the experiment folder — this keeps a single source of truth per experiment while
letting each experiment stay self-contained in terms of code.

Every experiment README answers the same eleven questions (research question,
motivation, hypothesis, setup, controlled variables, changed variables, metrics,
results, interpretation, limitations, new questions) — see `FULL-ROADMAP.md` §6.

## 7. Results Pipeline

```text
EXPERIMENT SCRIPT → results/raw/<id>/ → analysis script → results/processed/<id>/
                                                          → results/figures/<id>/ → REPORT
```

Raw data is never edited by hand. If a run is bad, re-run it — don't patch the CSV.

## 8. Experiment Metadata Standard

Every experiment run records hardware, software versions, model identity/quantization,
inference configuration, and measurement parameters (repetitions, warm-up runs, metrics
collected). The canonical schema is in `FULL-ROADMAP.md` §14; in this repo it's captured
by `scripts/common.py::environment_metadata()` in each experiment and written alongside
raw results.

---

## 9. Setup

Requirements: macOS or Linux, Python 3.11+, [`uv`](https://docs.astral.sh/uv/) for
Python dependency management. Go, Docker, and Kubernetes tooling are only needed from
Week 7 onward.

```bash
# clone, then from the repo root:
uv sync                 # creates .venv/ and installs pyproject.toml dependencies
source .venv/bin/activate
```

Model weights are downloaded on demand by each experiment's scripts via the Hugging
Face Hub (or `llama.cpp` from Week 2 onward) and cached locally under `models/` or the
Hugging Face cache — they are **never committed to git** (see `models/README.md`).

## 10. Running an Experiment

Each experiment is runnable independently from the repo root:

```bash
uv run python experiments/01-inference-basics/scripts/exp_1_1_loading_time.py
uv run python experiments/01-inference-basics/scripts/exp_1_2_prompt_length.py
uv run python experiments/01-inference-basics/scripts/exp_1_3_output_length.py
uv run python experiments/01-inference-basics/analysis/analyze.py
```

Raw CSVs land in `results/raw/01-inference-basics/`, processed summaries and figures in
`results/processed/` and `results/figures/` under the same experiment id. See
[`experiments/01-inference-basics/README.md`](experiments/01-inference-basics/README.md)
for the full write-up.

## 11. Definition of Done

- **An experiment** is done when its question, hypothesis, and limitations are
  documented, it's reproducible from a fresh checkout, and raw data is preserved.
- **A week** is done when at least one experiment is complete, code is committed,
  results are analyzed, and lessons learned are written down.
- **A phase** is done when its individual experiments are synthesized into a phase-level
  report.

Full criteria: `FULL-ROADMAP.md` §16.

## 12. What This Project Deliberately Does Not Build

No chatbot UI, no ChatGPT clone, no auth/user management, no SaaS product, no generic
agent framework, no generic RAG demo, no unnecessary microservices. This is a research
program, not a product — see `FULL-ROADMAP.md` §17 for the full scope-control rules.

## 13. Publication

Work is published at three levels: the code and raw data live here on GitHub
(reproducibility), interpreted findings are written up as field notes and reports on
**bessilabs**, and select milestones get a **Learning in Public** video. See
`FULL-ROADMAP.md` §§11–12 for the schedule.

## 14. License

See [`LICENSE`](LICENSE).
