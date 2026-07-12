# CPU Inference Performance Report v1

### Phase I Final Deliverable — Weeks 1–3, Efficient AI Systems

**Central question:** What actually happens when a Small Language Model runs on a CPU?

This report synthesizes three weeks of controlled experiments on a single machine
(Apple M4, 4 performance + 6 efficiency cores, 10 total; macOS 15.7 arm64) running one
model (`Qwen/Qwen2.5-1.5B-Instruct`) across two engines (Hugging Face Transformers in
Python/fp32, and llama.cpp/GGUF in F16, CPU-only — Metal disabled). Full detail,
raw data, and figures for each experiment live in the linked week READMEs; this report
connects them into one narrative and states what Phase I leaves open for Phase II.

- [Week 1 — Transformer Inference Fundamentals](../../experiments/01-inference-basics/README.md)
- [Week 2 — llama.cpp and GGUF](../../experiments/02-llama-cpp/README.md)
- [Week 3 — CPU Performance Engineering](../../experiments/03-cpu-performance/README.md)
- [Open questions tracker](../../docs/methodology/open-questions.md)

---

## 1. Inference Architecture

Every experiment this phase confirms the same two-phase structure: **prefill**
(processing the whole prompt in one forward pass, producing the first token) and
**decode** (one forward pass per subsequent token, reusing the KV cache from prefill).
These have different performance characteristics because they're different
computational shapes — prefill is one large, parallelizable pass over many tokens at
once; decode is a sequence of small, latency-bound steps, one token at a time.

Week 1's from-scratch Python generation loop made this split explicit by timing them
separately (TTFT = prefill, decode time = everything after). Every later experiment —
in both Python and llama.cpp — reports the same two numbers, and the split holds up
under every stress test this phase applied to it (thread count, context length,
background load).

## 2. Python vs llama.cpp

Week 2 Experiment 2.1 ran both engines under matched conditions (56-token
chat-templated prompt, 64 generated tokens, 10 threads):

| metric | llama.cpp (F16) | Python (fp32) |
|---|---|---|
| load time | 0.84s ± 0.59 | 3.22s ± 0.51 |
| TTFT | 0.37s ± 0.04 | 1.20s ± 0.04 |
| decode speed | 17.98 ± 3.61 tok/s | 10.48 ± 0.03 tok/s |
| peak RSS | 3769 MB ± 7 | 8739 MB ± 936 |

llama.cpp won on every axis — 3.8x faster loading, 3.2x faster TTFT, 1.7x faster
decode, 2.3x lower peak memory. This isn't a fair, fully-isolated "engine only"
comparison (F16 vs fp32 numeric precision differs too — tracked as
[open question #6](../../docs/methodology/open-questions.md)), but the gap is wide
enough that "use a dedicated CPU inference engine, not a general-purpose Python
framework" is already a clear, actionable conclusion at this scale.

## 3. Thread Scaling

This is the clearest example of a hypothesis getting genuinely revised mid-phase.
Week 2 Experiment 2.2's first, coarse sweep (1/2/4/8/10 threads) found throughput
peaking at 2 threads and *degrading* — not plateauing — beyond 4. That was surprising
enough to warrant Week 3 Experiment 3.1's finer sweep (every integer 1–6, plus 8, 10),
which found:

| threads | prefill tok/s | decode tok/s |
|---|---|---|
| 1 | 302.5 | 19.0 |
| 2 | 384.9 | 22.5 |
| 3 | 380.1 | 22.4 |
| 4 | 348.5 | 22.2 |
| 5 | 230.3 | 18.9 |
| 6 | 217.7 | 15.9 |
| 8 | 195.0 | 14.3 |
| 10 | 178.4 | 9.5 |

Throughput plateaus across 2–4 threads and then falls off a cliff starting at 5 —
and `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` (no root needed)
confirms this machine has **exactly 4 performance cores** (plus 6 efficiency cores).
The collapse point lines up with the P-core count almost exactly. The practical
takeaway for every later week using this machine: **use 2–4 threads, never more** —
"more threads" is not "more throughput" on a heterogeneous-core CPU past the
performance-core boundary, it's actively worse.

## 4. Context (Prompt Length) Scaling

Week 1 Experiment 1.2 first showed TTFT growing with prompt length (32→2048 tokens,
Python/fp32) and read decode speed as "roughly constant" — but that data was noisy.
Week 3 Experiment 3.2 repeated the idea on llama.cpp, at the optimal 2 threads, out to
4096 tokens (twice Week 1's range), with much tighter measurements:

| prompt tokens (actual) | TTFT (s) | decode tok/s | peak RSS (MB) |
|---|---|---|---|
| 156 | 0.567 | 21.80 | 3742 |
| 534 | 1.962 | 19.73 | 3817 |
| 1039 | 4.019 | 18.06 | 3821 |
| 2049 | 9.696 | 15.89 | 3820 |
| 4068 | 26.397 | 13.60 | 3829 |

Two things Week 1 couldn't see clearly: TTFT grows slightly **super-linearly** (per-token
prefill cost rises from ~3.6 ms/token at 156 tokens to ~6.5 ms/token at 4068 — consistent
with self-attention's quadratic term becoming non-negligible), and decode speed
**clearly decreases with context length** (21.8 → 13.6 tok/s, a 38% drop). Week 1's
"decode is roughly context-independent" read doesn't hold up — every decode step
attends over the *entire* KV cache, so a longer context makes every subsequent decode
step more expensive too, not just the initial prefill. Peak RSS grows only modestly
(+87 MB across this range) since this model's KV cache is small relative to its
weights.

## 5. Memory Behavior

Three separate memory findings, each with a different cause:

- **llama.cpp uses far less memory than Python for the same model** (§2): ~3.8GB vs
  ~8.7GB peak RSS. Mostly the fp32-vs-F16 weight footprint plus Python/Transformers
  framework overhead.
- **Python's peak RSS grew across repeated, independent process launches** in Week 2
  Exp 2.1 (~6.1GB on the first run to ~9.2GB on later runs), despite each being a fresh
  OS process. The likely explanation, not directly verified: memory-mapped
  safetensors loading interacting with a warming OS page cache across repeated loads
  of the same file.
- **Context length has a small, not large, effect on memory** (§4): +87MB from 156 to
  4068 tokens, because this model's per-token KV cache cost is small relative to its
  ~3GB+ weight footprint.

## 6. Environmental Effects: Background Load and Repeatability

Week 3 Experiment 3.3 found inference throughput degrades monotonically under
concurrent, unrelated CPU load, but with diminishing marginal harm: 0→2 background
hogs cost ~30% of prefill throughput, 2→4 cost a further ~18%, but 4→8 barely moved
prefill further (169.6→170.1 tok/s) while decode kept declining modestly. A plausible
read: once contention saturates enough cores, more background load doesn't hurt bulk
prefill much further, but decode's many small, latency-sensitive steps stay more
sensitive to scheduling contention.

The most interesting cross-week story this phase produced, though, is about
**repeatability**. Week 2 Experiment 2.3 ran 25 independent benchmark launches at 10
threads and found a strong, steady throughput *decline* (Pearson r = −0.89 for
prefill) that plateaued low after about 13 runs — consistent, at the time, with
thermal throttling under sustained load. Week 3 Experiment 3.4 repeated the same idea
at the actual optimal thread count (2) and found the **opposite direction**: noisy but
not declining for the first ~13 runs, then a stable plateau *higher* than where it
started (pp: r=+0.32, last-10-mean 12% above first-10-mean; tg: r=+0.56, +15%). This
machine never exposed a usable thermal signal without root access (`pmset -g therm`
stayed empty throughout; `powermetrics` needs `sudo` and wasn't pursued), so this isn't
a fully instrumented answer — but the fact that the decline **disappears at the
correct thread count** is fairly strong evidence that Week 2's finding was
substantially a consequence of running at a bad (oversubscribed) thread count, not a
generic, thread-count-independent thermal effect.

## 7. Observed Bottlenecks

In order of how much they moved the numbers this phase:

1. **Thread oversubscription past the performance-core count** — the single biggest
   lever observed (2–4x swings in throughput), and specific to heterogeneous-core CPUs
   like Apple Silicon.
2. **Engine choice** (Python/Transformers vs llama.cpp) — 1.7–3.8x across every metric.
3. **Context length**, via two independent mechanisms — quadratic-ish prefill growth,
   and KV-cache-driven decode slowdown.
4. **Concurrent system load** — real, monotonic, but with diminishing marginal effect
   past moderate contention.
5. **Numeric precision** (fp32 vs F16) — present but not yet isolated from engine
   choice; Phase II (quantization) is where this gets a clean answer.

## 8. Unanswered Questions Going Into Phase II

The full, itemized register is
[`docs/methodology/open-questions.md`](../../docs/methodology/open-questions.md).
Still open at the end of Phase I:

- How much of llama.cpp's Week 2 advantage is the engine vs. the F16-vs-fp32 precision
  difference? (Directly answerable once Phase II adds multiple quantization levels.)
- Would root-level thermal instrumentation (`powermetrics`) confirm the Week
  3 Experiment 3.4 conclusion directly, via real clock-frequency data?
- Does decode speed keep declining smoothly as context approaches this model's
  8192-token limit (only ~half was tested), or break down non-smoothly near it?
- How do llama-server's HTTP-level latencies compare to the llama-cli-level numbers
  measured here? (Relevant once the Week 7 Go gateway sits in front of llama-server.)
- Would memory-bandwidth-bound background load produce a different degradation curve
  than the compute-bound busy-loops used in Experiment 3.3?

## 9. Reproducibility

Every number in this report traces back to a raw CSV and metadata JSON under
`results/raw/{01-inference-basics,02-llama-cpp,03-cpu-performance}/`, generated by the
scripts in each week's `experiments/*/scripts/` directory and processed by each week's
`analysis/analyze.py`. See each week's README §4 ("What is the experimental setup?")
for exact commands. Hardware, software versions, and configuration are captured
per-experiment per the metadata standard in root `README.md` §8.
