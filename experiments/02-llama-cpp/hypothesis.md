# Hypotheses — Week 2: llama.cpp and GGUF

## Overall

llama.cpp is a purpose-built CPU inference engine (C/C++, quantized/optimized GEMM
kernels, no Python or autograd overhead, memory-mapped GGUF loading), while Week 1's
baseline is a general-purpose training/inference framework (Hugging Face Transformers)
running eagerly in fp32. On identical hardware and an equivalent prompt, llama.cpp
should load faster, produce the first token faster, decode faster, and use less
memory.

## Experiment 2.1 — Python vs llama.cpp

> llama.cpp achieves higher decode throughput, lower Time to First Token, faster model
> loading, and lower peak memory than the Hugging Face Transformers baseline for the
> same model, under matched thread count and an equivalent (chat-templated) prompt.

Falsifiable by: llama.cpp not showing a clear advantage on these metrics, which would
suggest the Python baseline's disadvantage in Week 1 was specific to that
implementation rather than representative of "Python framework vs dedicated CPU
engine" in general.

## Experiment 2.2 — Thread Count

> Both prefill (pp) and decode (tg) throughput increase with thread count up to some
> point, then flatten or decline as threading overhead and memory bandwidth
> contention start to dominate — on a heterogeneous-core CPU (Apple M4: performance +
> efficiency cores), this point may arrive well before the full logical core count.

Falsifiable by: throughput increasing monotonically all the way to the maximum thread
count with no plateau or regression.

## Experiment 2.3 — Repeatability

> Run-to-run variance is small (single-digit percent coefficient of variation) once
> the OS file cache is warm, and the first independent invocation is measurably slower
> than subsequent ones (cold-cache effect, as in Week 1 Experiment 1.1).

Falsifiable by: variance staying large throughout with no stabilization, or the first
run not being an outlier at all.
