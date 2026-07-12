# Glossary

Plain-English explanations of the technical terms used in each week's README and
results — for readers following along who aren't systems engineers or ML
researchers by trade. Each week gets its own section, built from that week's
"Learning Objectives" in `FULL-ROADMAP.md` plus any other term that actually shows up
in that week's README/results.

**Process:** when a new week's report is written, add a `## Week N — Title` section
here covering its terms. If a term already appeared in an earlier week, don't
redefine it — link back to where it was first defined instead (e.g. `[KV
cache](#week-1--transformer-inference-fundamentals)`).

---

## Week 1 — Transformer Inference Fundamentals

**Tokenization** — splitting text into the small chunks (tokens) a model actually
reads — not always whole words. "Efficient AI" might become tokens like `Effic`,
`ient`, ` AI`. Every downstream measurement in this project (speed, cost, context
limits) is counted in tokens, not words or characters.

**Embeddings** — turning each token into a list of numbers (a vector) that captures
something about its meaning. Similar words end up with similar-looking vectors — it's
how the model represents "meaning" in a form it can do math on.

**Transformer block** — the repeating building unit of the model. A model this size
stacks many of these blocks on top of each other; each one refines the running
representation of the text a bit further before passing it to the next.

**Self-attention** — the mechanism inside a transformer block that lets each token
"look at" every other token in the prompt so far and decide which ones matter for
understanding it. It's how the model knows "it" refers back to "the model" three
words earlier.

**Feed-forward network** — the other main piece inside a transformer block, applied
to each token independently after self-attention. If self-attention is about
*relating* tokens to each other, the feed-forward network is about further
*processing* each one on its own.

**Logits** — the model's raw, unprocessed scores for "how likely is each possible
next token," one number per token in its vocabulary, before they're turned into
probabilities.

**Sampling** (including **greedy decoding**) — the rule used to turn logits into an
actual chosen next token. Greedy decoding — what this project's Python scripts use —
always picks the single highest-scoring token, which is deterministic and easy to
reproduce, at the cost of the variety a more random sampling strategy would give.

**Autoregressive generation** — generating text one token at a time, where each new
token is chosen based on everything generated so far, then fed back in to generate
the next one. This is why generation is inherently sequential — you can't compute
token 10 without having already produced tokens 1–9.

**Prefill** — the first step of generating a response: processing the entire input
prompt in one go to prime the model before it produces anything. This is one large,
parallelizable computation, which is why it can be relatively fast per token
processed.

**Decode** — every step after prefill: producing one new token at a time, each step
depending on the last. Unlike prefill, this can't be parallelized across tokens
(token 12 needs token 11 to exist first), which is why decode has a different, more
latency-bound performance profile than prefill.

**KV cache** — short for "key/value cache." During prefill and decode, the model
computes some intermediate values for each token that would be expensive to
recompute from scratch on every later step — so it caches them instead. This is what
makes decode fast: each new token only does new work, reusing everything already
cached from before.

**Context length** — the maximum number of tokens (prompt + generated response
combined) a model can keep track of at once. Longer context means the model can
"see" more text at once, but costs more compute and memory (see the KV cache above).

**Time to First Token (TTFT)** — how long a user waits from sending a prompt to
seeing the very first word appear. This is dominated by prefill time, since the first
token can't appear until prefill finishes.

**Tokens per second** — the throughput measure used everywhere in this project: how
many tokens the model produces (or processes) per second. Higher is faster/better.

**Process memory (RSS)** — "Resident Set Size," the amount of RAM a running process
is actually occupying, as reported by the operating system. Used throughout this
project to measure how much memory loading and running a model actually costs.

---

## Week 2 — llama.cpp and GGUF

**llama.cpp** — a CPU-and-GPU inference engine written in C/C++, built specifically
to run language models fast and with a small memory footprint, as opposed to
general-purpose frameworks (like the Python/Transformers stack from
[Week 1](#week-1--transformer-inference-fundamentals)) that are built for both
training and inference and carry more overhead as a result.

**GGUF** — the file format llama.cpp uses to store models: weights, tokenizer, and
metadata all packed into one file, designed to be loaded quickly and efficiently
(see memory-mapped loading, below).

**Model serialization** — the general idea of "how a model's numbers get saved to and
loaded back from a file." GGUF is one serialization format; the `.safetensors` files
Week 1's Python stack loads are another.

**Tensor** — a multi-dimensional array of numbers — the basic data structure a model
is made of (a single weight matrix, for example, is a tensor). "Tensor storage" just
means how those arrays are laid out and compressed inside a model file.

**Memory-mapped loading** — instead of copying an entire model file into RAM upfront,
the operating system maps the file directly into the process's address space and
only actually loads the parts that get used, on demand. This is a big part of why
llama.cpp can start up faster and use less memory than loading everything eagerly.

**Quantization** — storing a model's numbers with fewer bits than the original
training precision (e.g. 8 or 4 bits instead of 16 or 32), trading some numeric
precision for a smaller file and faster math. This week only uses **F16** (16-bit
floating point, no precision reduction beyond the original training format) to
isolate the *engine* comparison from precision effects — quantization itself
(testing multiple lower-bit formats and what quality they cost) is the dedicated
subject of Weeks 4–5.

**BLAS / Accelerate** — BLAS ("Basic Linear Algebra Subprograms") is a standard
interface for fast matrix math; Accelerate is Apple's own highly-optimized
implementation of it for Apple Silicon. llama.cpp uses it here as a CPU speed boost —
it's still 100% CPU computation, just using better-optimized math routines than a
naive implementation would.

**Metal** — Apple's GPU programming framework. This project explicitly **disables**
Metal when building llama.cpp, forcing all computation onto the CPU, to keep the
comparison to Week 1's forced-CPU Python baseline fair (and because the whole program
is about CPU-first inference).

**Peak RSS** — the *maximum* value [process memory (RSS)](#week-1--transformer-inference-fundamentals)
reached at any point while a process ran, not just a single before/after snapshot —
a more honest measure of a process's worst-case memory footprint.

**Coefficient of variation (CV%)** — a measure of how noisy a set of measurements is,
relative to its own average: standard deviation divided by the mean, as a percentage.
A CV of 5% means the measurements are tightly clustered; 40% means they bounce around
a lot relative to their average value.

**Pearson correlation (r)** — a number between −1 and +1 describing how strongly two
things move together in a straight-line way. +1 means "as one goes up, the other goes
up perfectly proportionally"; −1 means the same but in opposite directions; 0 means no
linear relationship. This project uses it to check whether throughput trends up or
down over a sequence of repeated runs.

---

## Week 3 — CPU Performance Engineering

**Physical core vs. logical core** — a physical core is an actual, separate
processing unit on the chip. A logical core is what the operating system sees as
schedulable "threads of execution," which can be more than the physical core count
on CPUs that support simultaneous multithreading (this project's Apple M4 does not,
so physical and logical core counts are equal — 10 — but the *type* of those 10 cores
still varies, see performance/efficiency cores below).

**CPU cache** — small, very fast pools of memory built directly into the CPU chip,
sitting between the processor and much slower main RAM, used to keep recently-used
data close at hand. Several progressively larger/slower cache levels typically exist
between the CPU core and RAM.

**Memory hierarchy** — the overall layered structure of storage a CPU uses, from
fastest/smallest (registers, then CPU cache) to slowest/largest (RAM, then disk).
Performance often comes down to how well a workload keeps its data in the fast layers.

**Memory bandwidth** — how much data can be moved between RAM and the CPU per
second. Some workloads are limited by *compute* (see below); others are limited by
simply not being able to move data in and out fast enough, regardless of how fast the
math itself could run.

**SIMD** ("Single Instruction, Multiple Data") — a CPU feature that applies the same
math operation to several numbers at once in a single instruction, instead of one at
a time. Modern CPUs (including this project's Apple M4, via ARM NEON) rely heavily on
this for the matrix math that language model inference is built from.

**Compute-bound vs. memory-bound** — a compute-bound workload's speed is limited by
how fast the CPU can do arithmetic; a memory-bound workload's speed is limited by how
fast data can be fetched from memory instead. Which one a given inference workload
is shifts with context length and batch size, and it matters because the two kinds of
bottleneck respond to completely different optimizations.

**NUMA** ("Non-Uniform Memory Access") — a multi-CPU system design where each
processor has its own directly-attached memory, and accessing another processor's
memory is slower than accessing your own. Mentioned only conceptually in this
program — the single-socket Apple M4 used throughout doesn't have this issue.

**Thread contention** — what happens when more threads want to run than there are
cores available to run them on, forcing the operating system to constantly switch
which thread is actually executing. This project's [Experiment
3.3](../../experiments/03-cpu-performance/README.md#8-what-are-the-results) measures
exactly this, by adding competing background processes.

**Performance core / efficiency core (P-core / E-core)** — Apple Silicon chips (like
the M4 used throughout this project) mix two kinds of CPU core on the same chip:
performance cores (fast, power-hungry) and efficiency cores (slower, power-sipping).
This machine has 4 performance + 6 efficiency cores. [Experiment
3.1](../../experiments/03-cpu-performance/README.md#8-what-are-the-results) found
that inference throughput collapses as soon as thread count exceeds the performance
core count — extra threads start landing on the slower efficiency cores (or incur
scheduling overhead moving between core types) instead of adding real parallel
throughput.

**Thermal throttling** — when a chip running hard for a while heats up enough that
it deliberately reduces its own clock speed to stay within safe operating
temperature, trading peak performance for not overheating. [Experiment
3.4](../../experiments/03-cpu-performance/README.md#8-what-are-the-results)
investigated whether a throughput decline seen in [Week
2](../../experiments/02-llama-cpp/README.md#8-what-are-the-results) was this effect —
the evidence pointed more toward thread contention than pure thermal throttling, but
this machine has no non-root way to directly measure temperature or clock speed to
confirm it outright.

---

## Week 4 — Quantization Fundamentals

**FP32 / FP16 / BF16** — three ways of storing a real number in a fixed number of
bits. FP32 ("32-bit float") is the standard, high-precision format most models are
originally trained in. FP16 uses half as many bits, trading some precision for a
smaller footprint (this is what Week 2's GGUF model uses, labeled "F16" there).
BF16 also uses 16 bits but allocates them differently, favoring the same numeric
*range* as FP32 at the cost of more precision than FP16 — not used directly in this
project, but a common training-time format worth knowing.

**INT8 / INT4** — storing numbers as 8-bit or 4-bit integers instead of a floating
point format, the core idea behind the more aggressive quantization levels this week
tests (e.g. Q8_0 is built around 8-bit integers). Far smaller than FP16, but numbers
have to be rescaled back to something like floating point before most of the actual
math happens — see "scales," below.

**Quantization error** — the gap between a number's true (e.g. FP32) value and what
it becomes after being squeezed into a lower-precision format and converted back.
Individually tiny, these errors accumulate across a model's billions of weights, and
measuring their real-world impact on output quality is Week 5's job — this week only
measures the speed/memory side of the trade.

**Scales** — since a low-bit integer (e.g. INT4, which only represents 16 distinct
values) can't directly represent the full range of a model's real-valued weights, GGUF
quantization formats store one (or a few) floating-point "scale" values alongside each
small block of quantized weights, and multiply by that scale to reconstruct an
approximate real value. The scale is what lets a 4-bit number stand in for a much
wider range of possible weight values.

**Groups** (a.k.a. blocks) — quantization scales aren't applied to the whole model at
once; weights are split into small groups (commonly 32 or 256 values), each with its
own scale, so the reconstruction is more accurate locally than one single global
scale could be. The "K" in formats like Q4_K_M refers to a specific, more elaborate
grouping/scaling scheme than the plain Q4_0 format uses.

**GGUF quantization formats (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, ...)** — each name
encodes roughly how many bits per weight (the leading number) and which specific
grouping/scaling scheme is used (the suffix — `_0` is the older, simpler scheme;
`_K` variants are newer and generally give better quality per bit, which is why this
project uses the `_K` variants where available). Despite the naming suggesting a
simple ordering by size, [Experiment
4.4](../../experiments/04-quantization/README.md#8-what-are-the-results) found that
speed and memory don't scale smoothly across these formats — different block
layouts interact differently with this CPU's optimized math instructions.

**REPACK** — a llama.cpp CPU backend feature (reported as a supported capability in
this project's build, alongside `NEON`/`DOTPROD`/`ACCELERATE`) that can rearrange
certain quantized formats' in-memory layout at load time for faster SIMD access
later. A plausible explanation, in [Experiment
4.3](../../experiments/04-quantization/README.md#8-what-are-the-results), for why
some quantization levels loaded slower than both the unquantized F16 model and the
smallest quantized one.

## Week 5 — Quantization vs Quality

**Bootstrap confidence interval** — a way to estimate how uncertain a summary
statistic (like a mean score) is, without assuming the data follows a particular
distribution: repeatedly resample the observed data with replacement, recompute the
statistic each time, and take a percentile range (e.g. 2.5th-97.5th) across the
resampled values as the "95% CI." Used throughout [Week
5](../../experiments/05-quantization-quality/README.md#8-what-are-the-results)
instead of a formula-based CI because quality scores aren't normally distributed
(many are 0/1).

**Paired comparison / Cohen's dz** — since every quantization level in Week 5 is
scored on the *same* 100 dataset examples, differences between two levels can be
computed per-example (`score_at_Q4_K_M - score_at_F16` for each of the 100 items)
rather than treating the two levels as independent samples — a paired design, which
is more statistically powerful than an unpaired one at the same sample size. Cohen's
dz is the standardized effect size for this paired setup: the mean of those
per-example differences divided by their standard deviation. Values below ~0.2 are
conventionally "small" — true of every quantization level tested this week.

**Statistical significance ("n.s.")** — shorthand, used in [Week
5](../../experiments/05-quantization-quality/README.md#8-what-are-the-results)'s
results table, for "the bootstrap confidence interval on this difference includes
zero," meaning the data can't rule out "no real difference" at the chosen confidence
level (95% here). Not the same as "there is no difference" — only that this
100-example benchmark isn't large enough to distinguish a small real difference from
noise.

**JSON validity** — whether a model's output, when a JSON object is expected, is
actually syntactically parseable JSON at all — a harder, more binary failure mode
than getting field *values* wrong (see `information_extraction`/`structured_output`
scoring). Week 5 found this only breaks down at the most aggressive quantization
level tested (Q3_K_M), and even then in a small minority of cases.

**Instruction compliance** — whether a model followed an explicit formatting
constraint in a prompt (e.g. "respond with only YES or NO"), tracked separately in
Week 5's scoring from whether the underlying *content* of the answer was correct —
a model can comply with the format and still be wrong, or ignore the format while
still giving the right underlying answer.

**Semantic similarity (lexical-overlap proxy)** — Week 5's summarization scorer
measures token-level overlap (a bag-of-words F1, similar in spirit to ROUGE) between
a model's output and a reference summary, used as a cheap, LLM-judge-free stand-in
for "does this mean the same thing" — a real limitation, since a correct paraphrase
with different wording would score poorly despite being a good summary.

**Pareto frontier (quality vs. performance)** — this program's Week 5 central
visualization: plotting each quantization level as a (speed, quality) point and
identifying which points aren't "dominated" — i.e. no other point is both faster
*and* higher-quality. A level *on* the frontier represents a genuine trade-off worth
considering; a level *off* the frontier is simply a worse choice than some other
level on both axes simultaneously, regardless of any use case's specific
speed/quality preference.
