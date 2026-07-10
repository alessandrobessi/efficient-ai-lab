# Efficient AI Systems

## A 12-Week Research and Engineering Program on CPU-First Small Language Models

**Duration:** 12 weeks  
**Primary stack:** Python, Go, llama.cpp, GGUF, Docker, Kubernetes/k3s, Prometheus, Grafana  
**Primary repository:** `efficient-ai-lab`  
**Publication platform:** bessilabs  
**Optional video series:** Learning in Public — Efficient AI Systems

---

# 1. Program Overview

## 1.1 Central Research Question

> How much useful AI can we build under strict compute constraints?

This 12-week program explores the engineering, performance, evaluation, deployment, and operational characteristics of Small Language Models running primarily on CPU infrastructure.

The program is designed as a coherent research and engineering project rather than a collection of unrelated tutorials.

The objective is to understand the complete lifecycle of an efficient AI system:

1. model inference;
2. token generation;
3. CPU performance;
4. model formats;
5. quantization;
6. model evaluation;
7. statistical analysis;
8. service architecture;
9. concurrency;
10. load testing;
11. observability;
12. containerization;
13. orchestration;
14. failure analysis;
15. architectural decision-making.

The program follows a strict experimental methodology:

```text
QUESTION
    ↓
HYPOTHESIS
    ↓
EXPERIMENT
    ↓
MEASUREMENT
    ↓
ANALYSIS
    ↓
INTERPRETATION
    ↓
NEXT QUESTION
```

Every week must produce at least one experiment.

Reading and studying alone do not count as progress.

---

# 2. Program Objectives

By the end of the 12 weeks, the program should provide a practical understanding of:

- transformer inference;
- autoregressive token generation;
- tokenization;
- prefill and decoding;
- attention;
- KV caching;
- CPU architecture;
- memory bandwidth;
- SIMD;
- threading;
- model serialization;
- GGUF;
- llama.cpp;
- quantization;
- inference benchmarking;
- statistical evaluation;
- model comparison;
- Go service development;
- concurrency;
- HTTP APIs;
- load generation;
- latency analysis;
- Prometheus metrics;
- Grafana dashboards;
- Docker;
- Kubernetes;
- resource management;
- horizontal scaling;
- failure analysis;
- AI system architecture.

The final objective is not merely to run Small Language Models.

The objective is to understand:

> What makes an AI system useful, efficient, reliable, measurable, and deployable under real-world constraints?

---

# 3. Guiding Principles

## 3.1 Experiment First

Every theoretical concept should eventually produce an experiment.

Examples:

- Does increasing CPU thread count always improve inference performance?
- How does prompt length affect Time to First Token?
- How much quality is lost when moving from Q8 to Q4?
- How does concurrency affect tail latency?
- What happens when Kubernetes throttles CPU resources?
- At what point does an SLM become preferable to a remote LLM API?

## 3.2 Measure Everything

Avoid statements such as:

> Model A feels faster.

Prefer:

> Model A achieved a median generation speed of X tokens per second with a p95 Time to First Token of Y milliseconds.

Relevant metrics include:

- model loading time;
- Time to First Token;
- inter-token latency;
- tokens per second;
- requests per second;
- p50 latency;
- p95 latency;
- p99 latency;
- CPU utilization;
- memory utilization;
- error rate;
- queue depth;
- quality score;
- model size;
- cost per request.

## 3.3 Prefer Depth Over Breadth

Do not benchmark dozens of models.

Do not test every inference engine.

Do not add technologies simply to increase the size of the stack.

Prefer:

- 4–6 carefully selected models;
- 3–5 quantization levels;
- controlled experiments;
- reproducible results;
- careful interpretation.

## 3.4 Separate Observation from Interpretation

Every report should distinguish between:

### Observation

What happened?

### Interpretation

Why might it have happened?

### Evidence

What measurements support the interpretation?

### Limitations

What alternative explanations exist?

## 3.5 Build Reproducible Experiments

Every experiment should document:

- hardware;
- operating system;
- software versions;
- model;
- model version;
- quantization;
- configuration;
- command executed;
- environment variables;
- random seed when applicable;
- raw results;
- analysis procedure.

---

# 4. Technology Stack

## Core Languages

### Python

Used for:

- experiments;
- evaluation;
- statistical analysis;
- visualization;
- model interaction;
- data processing.

Recommended libraries:

- NumPy;
- pandas;
- matplotlib;
- scipy;
- Hugging Face Transformers;
- PyTorch where necessary.

### Go

Used for:

- inference gateway;
- systems tooling;
- load generator;
- concurrency experiments;
- metrics collection;
- production-style services.

## AI Infrastructure

### llama.cpp

Primary inference engine.

Topics:

- compilation;
- runtime configuration;
- threading;
- batching;
- KV cache;
- server mode;
- model loading;
- performance measurement.

### GGUF

Primary model format.

Topics:

- model metadata;
- tensor storage;
- quantization formats;
- model loading;
- memory mapping.

## Infrastructure

### Docker

Used for:

- reproducible environments;
- service packaging;
- local deployment.

### Kubernetes / k3s

Used for:

- deployment;
- resource management;
- health checks;
- scaling;
- failure experiments.

### Prometheus

Used for:

- metrics collection;
- service monitoring;
- infrastructure monitoring.

### Grafana

Used for:

- visualization;
- dashboards;
- system analysis.

---

# 5. Repository Structure

```text
efficient-ai-lab/

├── README.md
├── LICENSE
├── Makefile
├── go.work
├── pyproject.toml
├── docker-compose.yml
│
├── docs/
│   ├── architecture/
│   ├── methodology/
│   ├── hardware/
│   └── decisions/
│
├── experiments/
│   ├── 01-inference-basics/
│   ├── 02-llama-cpp/
│   ├── 03-cpu-performance/
│   ├── 04-quantization/
│   ├── 05-quantization-quality/
│   ├── 06-model-comparison/
│   ├── 07-inference-service/
│   ├── 08-load-testing/
│   ├── 09-kubernetes/
│   ├── 10-slm-vs-llm/
│   ├── 11-decision-framework/
│   └── 12-final-synthesis/
│
├── evaluation/
│   ├── datasets/
│   ├── runners/
│   ├── metrics/
│   ├── prompts/
│   └── analysis/
│
├── services/
│   ├── inference-gateway/
│   └── load-generator/
│
├── infrastructure/
│   ├── docker/
│   ├── kubernetes/
│   ├── prometheus/
│   └── grafana/
│
├── scripts/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── reports/
│   ├── field-notes/
│   ├── benchmarks/
│   └── final/
│
└── models/
    └── README.md
```

Model binaries must not be committed to Git.

---

# 6. Standard Experiment Structure

Every experiment directory should contain:

```text
experiment-name/

├── README.md
├── hypothesis.md
├── config/
├── scripts/
├── data/
├── results/
└── analysis/
```

The experiment README should answer:

1. What question are we investigating?
2. Why does the question matter?
3. What is the hypothesis?
4. What is the experimental setup?
5. What variables are controlled?
6. What variables are changed?
7. What metrics are collected?
8. What are the results?
9. How should the results be interpreted?
10. What are the limitations?
11. What new questions emerged?

---

# 7. Phase I — Understand CPU Inference

**Duration:** Weeks 1–3

**Central question:**

> What actually happens when a Small Language Model runs on a CPU?

---

# Week 1 — Transformer Inference Fundamentals

## Learning Objectives

Understand:

- tokenization;
- embeddings;
- transformer blocks;
- self-attention;
- feed-forward networks;
- logits;
- sampling;
- autoregressive generation;
- prefill;
- decoding;
- KV caching;
- context length.

## Theory

Study the complete inference path:

```text
TEXT
  ↓
TOKENIZER
  ↓
TOKEN IDS
  ↓
EMBEDDINGS
  ↓
TRANSFORMER BLOCKS
  ↓
LOGITS
  ↓
SAMPLING
  ↓
NEXT TOKEN
  ↓
REPEAT
```

Understand the distinction between:

### Prefill

Processing the initial prompt.

### Decode

Generating tokens sequentially.

Understand why these phases have different performance characteristics.

## Practical Work

Select one model in the 1B–3B parameter range.

Run the model using Python.

Create scripts for:

- model loading;
- tokenization;
- prompt processing;
- generation;
- timing;
- memory measurement.

## Experiments

### Experiment 1.1 — Model Loading Time

Measure:

- total loading time;
- process memory before loading;
- process memory after loading.

Repeat multiple times.

### Experiment 1.2 — Prompt Length

Test multiple prompt lengths.

Suggested values:

- 32 tokens;
- 128 tokens;
- 512 tokens;
- 1024 tokens;
- 2048 tokens.

Measure:

- total latency;
- Time to First Token;
- generation speed.

### Experiment 1.3 — Output Length

Keep the prompt fixed.

Generate:

- 16 tokens;
- 64 tokens;
- 256 tokens;
- 512 tokens.

Measure total latency and generation speed.

## Deliverables

- Python inference script;
- benchmark script;
- raw CSV results;
- performance plots;
- experiment report.

## Weekly Learning Question

> Why is prompt processing different from token generation?

---

# Week 2 — llama.cpp and GGUF

## Learning Objectives

Understand:

- llama.cpp architecture;
- GGUF;
- model serialization;
- tensor storage;
- memory mapping;
- model loading;
- inference configuration.

## Theory

Study:

- why llama.cpp exists;
- how it differs from Python inference frameworks;
- GGUF metadata;
- GGUF tensors;
- quantization metadata;
- memory-mapped model loading.

## Practical Work

Compile llama.cpp locally.

Download the GGUF equivalent of the Week 1 model where available.

Run inference using:

- llama-cli;
- llama-server;
- llama-bench.

## Experiments

### Experiment 2.1 — Python vs llama.cpp

Compare:

- model loading time;
- RAM usage;
- Time to First Token;
- tokens per second.

### Experiment 2.2 — Thread Count

Test:

- 1 thread;
- 2 threads;
- 4 threads;
- 8 threads;
- maximum logical threads.

Measure:

- throughput;
- latency;
- CPU utilization.

### Experiment 2.3 — Repeatability

Execute the same benchmark at least 20 times.

Analyze:

- variance;
- standard deviation;
- outliers;
- warm-up effects.

## Deliverables

- llama.cpp installation instructions;
- benchmark scripts;
- comparison dataset;
- plots;
- experiment report.

## Field Note #1

**Title:**

> What Actually Happens When You Run a Language Model on a CPU?

---

# Week 3 — CPU Performance Engineering

## Learning Objectives

Understand:

- physical cores;
- logical cores;
- CPU caches;
- memory hierarchy;
- memory bandwidth;
- SIMD;
- compute-bound workloads;
- memory-bound workloads;
- NUMA conceptually;
- thread contention.

## Experiments

### Experiment 3.1 — Thread Scaling

Measure inference performance as thread count increases.

Plot:

```text
THREAD COUNT
    vs
TOKENS PER SECOND
```

Identify the point of diminishing returns.

### Experiment 3.2 — Context Scaling

Measure performance with increasing context lengths.

Investigate:

- Time to First Token;
- memory consumption;
- decoding speed.

### Experiment 3.3 — Background Load

Run CPU-intensive background processes.

Measure degradation in inference performance.

### Experiment 3.4 — Thermal Effects

Where observable, run extended inference workloads.

Measure whether sustained performance changes over time.

## Phase I Final Deliverable

# CPU Inference Performance Report v1

The report should summarize:

- inference architecture;
- Python vs llama.cpp;
- thread scaling;
- prompt scaling;
- memory behavior;
- observed bottlenecks;
- unanswered questions.

---

# 8. Phase II — Measure Models and Quantization

**Duration:** Weeks 4–6

**Central question:**

> What performance and quality tradeoffs emerge when Small Language Models are compressed and compared?

---

# Week 4 — Quantization Fundamentals

## Learning Objectives

Understand:

- FP32;
- FP16;
- BF16;
- INT8;
- INT4;
- quantization error;
- scales;
- groups;
- GGUF quantization formats.

## Model Selection

Select one model.

Test approximately:

- Q8;
- Q6;
- Q5;
- Q4;
- Q3.

Exact formats depend on model availability.

## Experiments

### Experiment 4.1 — Model Size

Measure disk size.

### Experiment 4.2 — Memory Consumption

Measure RAM usage.

### Experiment 4.3 — Loading Performance

Measure model loading time.

### Experiment 4.4 — Inference Performance

Measure:

- Time to First Token;
- tokens per second;
- total latency.

## Evaluation Dataset Design

Begin constructing the evaluation dataset.

Target:

100–200 examples.

Categories:

- classification;
- information extraction;
- structured output;
- summarization;
- simple reasoning;
- instruction following;
- Italian language;
- English language.

Use JSONL.

Example:

```json
{
  "id": "classification-001",
  "category": "classification",
  "language": "en",
  "prompt": "...",
  "expected": "...",
  "metadata": {}
}
```

## Deliverables

- quantization benchmark;
- evaluation dataset v1;
- benchmark report.

---

# Week 5 — Quantization vs Quality

## Central Question

> How much quality is lost as models become smaller and faster?

## Evaluation Pipeline

Build:

```text
DATASET
   ↓
MODEL RUNNER
   ↓
RAW OUTPUT
   ↓
EVALUATOR
   ↓
METRICS
   ↓
STATISTICAL ANALYSIS
```

## Metrics

Depending on task:

- accuracy;
- exact match;
- F1;
- JSON validity;
- instruction compliance;
- semantic similarity where appropriate.

Avoid relying entirely on LLM-as-a-judge evaluation.

## Statistical Analysis

Analyze:

- mean performance;
- variance;
- confidence intervals;
- effect sizes;
- failure categories.

## Main Visualization

Create the:

```text
QUALITY
   ▲
   │
   │         ●
   │      ●
   │   ●
   │ ●
   └────────────────► PERFORMANCE
```

Identify the quality-performance Pareto frontier.

## Field Note #2

**Title:**

> The Cost of Compression: What Quantization Actually Does to Small Language Models

## Optional YouTube Video

**Title:**

> I Quantized a Small Language Model Five Ways. Here Is What Changed.

---

# Week 6 — Small Model Comparison

## Model Selection

Select 4–6 models.

Criteria:

- different parameter counts;
- different model families;
- CPU compatibility;
- GGUF availability;
- multilingual capabilities.

## Experimental Controls

Keep constant:

- hardware;
- inference engine;
- benchmark methodology;
- prompts;
- sampling configuration;
- number of repetitions.

## Measure

For every model:

- model size;
- RAM;
- loading time;
- Time to First Token;
- tokens per second;
- evaluation quality;
- Italian performance;
- English performance;
- structured output reliability.

## Analysis Questions

- Does parameter count predict quality?
- Does parameter count predict latency?
- Which models provide the best performance-quality tradeoff?
- Which models perform best in Italian?
- Which models produce the most reliable structured output?
- Are some model families particularly CPU-friendly?

## Phase II Final Deliverable

# The CPU Small Language Model Benchmark

Include:

- methodology;
- hardware;
- models;
- results;
- statistical analysis;
- visualizations;
- limitations;
- reproducibility instructions.

## YouTube Video

**Title:**

> I Tested Small Language Models on CPU: What Actually Matters

---

# 9. Phase III — Build a Production System

**Duration:** Weeks 7–9

**Central question:**

> How do we turn a locally running model into a reliable and observable AI service?

---

# Week 7 — Go Inference Gateway

## Architecture

```text
CLIENT
   ↓
GO INFERENCE GATEWAY
   ↓
LLAMA.CPP SERVER
   ↓
SMALL LANGUAGE MODEL
```

## Learning Objectives

Understand:

- HTTP service design;
- Go HTTP servers;
- configuration;
- context cancellation;
- timeouts;
- error handling;
- structured logging;
- graceful shutdown;
- health checks.

## Required Endpoints

```text
GET /health
GET /ready
GET /metrics
POST /v1/generate
```

## Request Example

```json
{
  "prompt": "Explain CPU inference.",
  "max_tokens": 128,
  "temperature": 0.7
}
```

## Features

Implement:

- request validation;
- configurable timeout;
- request IDs;
- structured logging;
- llama.cpp communication;
- error mapping;
- graceful shutdown;
- Prometheus metrics.

## Metrics

Expose:

- request count;
- error count;
- request duration;
- active requests;
- generated tokens;
- generation duration.

## Deliverables

- Go service;
- tests;
- Dockerfile;
- API documentation;
- architecture diagram.

---

# Week 8 — Load Testing and Observability

## Build a Go Load Generator

The objective is educational.

Do not initially use an existing load-testing tool.

## Learn

- goroutines;
- channels;
- synchronization;
- HTTP clients;
- rate limiting;
- workload generation;
- histogram analysis;
- percentiles;
- coordinated omission conceptually.

## Load Generator Features

Configure:

- target URL;
- concurrent clients;
- requests per second;
- duration;
- prompt dataset;
- output path.

## Collect

- request latency;
- p50;
- p95;
- p99;
- throughput;
- error rate;
- Time to First Token;
- tokens per second.

## Workloads

### Workload A — Single User

One sequential client.

### Workload B — Small Team

5 concurrent clients.

### Workload C — Medium Load

20 concurrent clients.

### Workload D — Saturation

Increase concurrency until performance collapses.

## Questions

- When does throughput stop increasing?
- When does latency become unacceptable?
- What happens to p99 latency?
- Does the system queue requests?
- Does CPU usage reach 100%?
- What resource becomes the bottleneck?

## Observability

Deploy:

- Prometheus;
- Grafana.

Create dashboards for:

- request rate;
- latency;
- errors;
- CPU;
- RAM;
- active requests;
- model throughput.

## Field Note #3

**Title:**

> What Happens When Multiple Users Share a CPU Language Model?

---

# Week 9 — Kubernetes and Failure Engineering

## Architecture

```text
KUBERNETES CLUSTER

┌───────────────────────────────┐
│                               │
│     GO INFERENCE GATEWAY      │
│               ↓               │
│       LLAMA.CPP SERVER        │
│               ↓               │
│              SLM              │
│                               │
│       PROMETHEUS              │
│       GRAFANA                 │
│                               │
└───────────────────────────────┘
```

## Kubernetes Components

Implement:

- Deployment;
- Service;
- ConfigMap;
- resource requests;
- resource limits;
- liveness probes;
- readiness probes;
- persistent model storage where appropriate.

## Experiments

### Experiment 9.1 — CPU Limits

Change CPU limits.

Measure inference degradation.

### Experiment 9.2 — Memory Limits

Restrict memory.

Observe behavior.

### Experiment 9.3 — Pod Failure

Delete inference pods during traffic.

Measure:

- errors;
- recovery time;
- request loss.

### Experiment 9.4 — Load Saturation

Overload the Kubernetes deployment.

Observe:

- CPU;
- memory;
- latency;
- errors;
- scheduling behavior.

### Experiment 9.5 — Horizontal Scaling

Run multiple replicas.

Investigate:

- throughput scaling;
- latency;
- model memory duplication;
- startup cost.

## Phase III Final Deliverable

# Building and Breaking a CPU-First AI Service on Kubernetes

## YouTube Video

**Title:**

> I Built and Broke a Production AI System Without GPUs

---

# 10. Phase IV — Synthesize and Decide

**Duration:** Weeks 10–12

**Central question:**

> When is a Small Language Model actually the right architectural choice?

---

# Week 10 — SLM vs LLM

## Task Selection

Select 5–10 realistic tasks.

Examples:

- classification;
- information extraction;
- summarization;
- structured JSON generation;
- document routing;
- simple reasoning;
- Italian-language processing;
- domain-specific Q&A.

## Systems Compared

Compare:

1. CPU SLM;
2. larger local model where feasible;
3. frontier remote API model.

## Dimensions

Measure:

- quality;
- latency;
- throughput;
- cost;
- privacy;
- operational complexity;
- deployment complexity;
- observability;
- failure modes.

## Important Principle

The research question is not:

> Which model is best?

The question is:

> Under which constraints does each architecture become preferable?

## Deliverables

- comparative dataset;
- benchmark scripts;
- analysis notebook or Python scripts;
- comparison report.

---

# Week 11 — AI Architecture Decision Framework

## Objective

Transform empirical results into an engineering decision framework.

## Initial Framework

```text
NEW AI TASK
     ↓
IS THE TASK CONSTRAINED?
     │
 ┌───┴───┐
 │       │
YES      NO
 │       │
 ▼       ▼
TEST     CONSIDER
SLM      LARGER MODEL
 │
 ▼
IS QUALITY SUFFICIENT?
 │
 ┌───┴───┐
 │       │
YES      NO
 │       │
 ▼       ▼
CHECK    ADD RETRIEVAL,
SYSTEM   FINE-TUNING,
CONSTRAINTS OR LARGER MODEL
 │
 ▼
EVALUATE
 ├── PRIVACY
 ├── LATENCY
 ├── THROUGHPUT
 ├── COST
 ├── OPERATIONS
 └── GOVERNANCE
```

## Required Output

The framework must be based on experimental evidence.

For every decision point, document:

- relevant experiments;
- observed results;
- limitations;
- exceptions.

## Field Note #4

**Title:**

> When Is a Small Language Model Enough?

---

# Week 12 — Final Synthesis

## Objective

Integrate all experiments into a single coherent technical argument.

## Flagship Report

# Efficient AI Systems

## What I Learned from 12 Weeks of Building, Measuring, and Breaking CPU-First Small Language Model Systems

## Suggested Structure

### 1. Introduction

Why investigate CPU-first AI systems?

### 2. Research Questions

What did the program attempt to understand?

### 3. Experimental Environment

Hardware, software, models, methodology.

### 4. Understanding CPU Inference

What happens during inference?

### 5. Performance

What determines inference performance?

### 6. Quantization

What changes when models are compressed?

### 7. Quality

How should quality degradation be measured?

### 8. Comparing Small Models

What matters beyond parameter count?

### 9. Concurrency

What happens under multi-user workloads?

### 10. Production Infrastructure

What changes when the model becomes a service?

### 11. Kubernetes

What did orchestration solve?

What did it not solve?

### 12. Failure Engineering

How did the system break?

### 13. SLM vs LLM

Under what conditions is each approach preferable?

### 14. Decision Framework

How should organizations select an architecture?

### 15. Limitations

What did the experiments fail to answer?

### 16. Future Research

What should come next?

### 17. Conclusion

What are the main lessons from the program?

## Final YouTube Video

**Title:**

> 12 Weeks of Building AI Systems Without GPUs: What I Learned

---

# 11. Publication Strategy

The publication strategy follows three levels.

## Level 1 — GitHub

GitHub contains:

- code;
- experiments;
- configurations;
- raw data;
- processed data;
- reproducibility instructions;
- technical documentation.

GitHub answers:

> Can someone reproduce the work?

## Level 2 — bessilabs Field Notes

Field notes contain:

- research questions;
- experimental results;
- interpretation;
- unexpected findings;
- lessons learned.

bessilabs answers:

> What did the experiments teach us?

## Level 3 — YouTube

YouTube contains:

- visual explanations;
- experiment demonstrations;
- architecture explanations;
- personal learning reflections.

YouTube answers:

> Why does this work matter, and how can it be understood?

---

# 12. Publication Schedule

| Week | GitHub | bessilabs | YouTube |
|---|---|---|---|
| 1 | Inference experiments | — | — |
| 2 | llama.cpp experiments | Field Note #1 | — |
| 3 | CPU benchmark | Performance Report | — |
| 4 | Quantization experiments | — | — |
| 5 | Evaluation framework | Field Note #2 | Optional |
| 6 | Model benchmark | Benchmark Report | Video #1 |
| 7 | Go inference service | — | — |
| 8 | Load generator | Field Note #3 | — |
| 9 | Kubernetes system | Infrastructure Report | Video #2 |
| 10 | SLM vs LLM experiments | — | — |
| 11 | Decision framework | Field Note #4 | — |
| 12 | Final repository | Flagship Report | Video #3 |

---

# 13. Weekly Operating Rhythm

Each week follows the same structure.

## Day 1 — Question

Define the research question.

Study necessary theory.

Write the hypothesis.

## Day 2 — Build

Implement the experimental infrastructure.

Validate correctness.

## Day 3 — Experiment

Execute controlled experiments.

Collect raw data.

## Day 4 — Analyze

Process results.

Create visualizations.

Perform statistical analysis.

## Day 5 — Interpret

Explain results.

Document limitations.

Identify new questions.

## Weekend — Consolidate

Refactor code.

Improve documentation.

Publish when appropriate.

---

# 14. Experiment Metadata Standard

Every experiment must record:

```yaml
experiment:
  id:
  title:
  date:
  hypothesis:

hardware:
  machine:
  cpu:
  cores:
  threads:
  ram:

software:
  os:
  python:
  go:
  llama_cpp:

model:
  name:
  parameters:
  format:
  quantization:

inference:
  threads:
  context_size:
  batch_size:
  temperature:
  seed:

measurement:
  repetitions:
  warmup_runs:
  metrics:
```

---

# 15. Results Storage

Raw data should never be manually edited.

Use:

```text
results/

├── raw/
│   └── experiment-id/
│
├── processed/
│   └── experiment-id/
│
└── figures/
    └── experiment-id/
```

The pipeline should be:

```text
EXPERIMENT
    ↓
RAW DATA
    ↓
PROCESSING SCRIPT
    ↓
PROCESSED DATA
    ↓
ANALYSIS SCRIPT
    ↓
FIGURES
    ↓
REPORT
```

---

# 16. Definition of Done

An experiment is complete when:

- the research question is documented;
- the hypothesis is documented;
- the experiment is reproducible;
- raw data is preserved;
- results are analyzed;
- visualizations are generated programmatically;
- limitations are documented;
- conclusions distinguish observation from interpretation.

A week is complete when:

- at least one experiment has been completed;
- code has been committed;
- results have been analyzed;
- lessons learned have been documented.

A phase is complete when:

- individual experiments have been synthesized;
- results have been compared;
- a phase-level report has been produced.

The program is complete when:

- the GitHub repository is reproducible;
- the evaluation framework works;
- the Go inference gateway works;
- the load generator works;
- the Kubernetes deployment works;
- the monitoring stack works;
- the major experiments are complete;
- the final report is published.

---

# 17. What Not to Build

Do not build:

- a generic chatbot;
- a ChatGPT clone;
- a complex frontend;
- an authentication system;
- user management;
- a SaaS product;
- an agent framework;
- a generic RAG demo;
- unnecessary microservices.

The purpose is not to create a commercial application.

The purpose is to understand efficient AI systems.

---

# 18. Scope Control

When considering adding a feature, ask:

> Does this help answer one of the central research questions?

If no, do not add it.

When considering adding a technology, ask:

> Does this technology allow us to investigate something we could not otherwise investigate?

If no, do not add it.

---

# 19. Expected Final Artifacts

At the end of 12 weeks:

## Software

- one research monorepo;
- Python experimentation framework;
- model evaluation framework;
- Go inference gateway;
- Go load generator;
- Docker environment;
- Kubernetes deployment;
- Prometheus configuration;
- Grafana dashboards.

## Research

- 12+ documented experiments;
- CPU inference benchmark;
- quantization benchmark;
- model comparison benchmark;
- concurrency benchmark;
- Kubernetes failure experiments;
- SLM vs LLM comparison;
- architecture decision framework.

## Writing

- 4 field notes;
- 3 phase reports;
- 1 flagship technical report.

## Video

- 2–3 Learning in Public videos.

---

# 20. Possible Continuation After Week 12

The program can continue into several directions.

## Direction A — Advanced CPU Optimization

- SIMD;
- AVX;
- ARM NEON;
- NUMA;
- memory bandwidth;
- compiler optimization.

## Direction B — Alternative Inference Engines

- ONNX Runtime;
- OpenVINO;
- MLX;
- TensorFlow Lite.

## Direction C — Model Optimization

- distillation;
- pruning;
- speculative decoding;
- mixture of experts;
- model routing.

## Direction D — Edge AI

- Raspberry Pi;
- smartphones;
- embedded hardware;
- constrained devices.

## Direction E — Distributed Inference

- multi-node inference;
- model sharding;
- distributed CPU systems.

## Direction F — Domain-Specific SLMs

- financial services;
- regulated industries;
- enterprise automation;
- Italian-language models.

## Direction G — AI System Economics

- cost per request;
- infrastructure utilization;
- API vs self-hosting;
- break-even analysis;
- total cost of ownership.

---

# 21. Final Program Principle

The program should not attempt to prove that Small Language Models are always better than Large Language Models.

It should investigate a more interesting question:

> Under which technical, economic, operational, and organizational constraints does a smaller and more efficient AI system become the better engineering choice?

The objective is not technological advocacy.

The objective is understanding.

At the end of the 12 weeks, success means being able to reason clearly about:

- model architecture;
- inference performance;
- hardware constraints;
- quantization;
- quality degradation;
- system design;
- concurrency;
- observability;
- orchestration;
- reliability;
- cost;
- privacy;
- governance;
- architectural tradeoffs.

The final outcome should be more than a portfolio.

It should be a coherent body of experimental work demonstrating how efficient AI systems behave, where they succeed, where they fail, and how they should be designed.
