# Hypotheses — Week 4: Quantization Fundamentals

## Overall (naive)

Fewer bits per weight should mean a smaller file, less memory, faster loading, and
faster inference, roughly proportionally, all the way from F16 down to Q3_K_M — the
same way Week 1-3 found simple, mostly-monotonic relationships between "less work"
and "more speed."

## Experiment 4.1 — Model Size

> Disk size decreases monotonically from F16 to Q3_K_M, roughly tracking each
> format's bits-per-weight.

Falsifiable by: any quant level being larger than a "higher-precision" one.

## Experiment 4.2 — Memory Consumption

> Peak RSS decreases monotonically alongside disk size.

Falsifiable by: a smaller-on-disk quant level using *more* peak memory than a
larger one.

## Experiment 4.3 — Loading Performance

> Load time decreases monotonically alongside disk size (less data to read/map from
> disk).

Falsifiable by: load time not tracking disk size — e.g. a smaller file taking longer
to load than a larger one.

## Experiment 4.4 — Inference Performance

> TTFT and decode speed both improve monotonically as quantization gets more
> aggressive (less data to move through memory bandwidth-bound decode, matching the
> memory-bandwidth discussion from Week 3).

Falsifiable by: decode speed or TTFT not tracking bit-width monotonically — e.g. a
more aggressively quantized format performing *worse* than a milder one.

## Why this is worth stating as a naive hypothesis to test, not an assumption

Week 3 already showed one clean counter-example to "the obvious mechanism wins"
(thread count didn't just plateau, it collapsed past the performance-core boundary).
GGUF quantization formats (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M) aren't just "the same
math with fewer bits" — they use different block/group structures for storing scales,
and llama.cpp's CPU backend can *repack* some of these layouts into a different
in-memory format for faster SIMD access (`REPACK` in this build's reported CPU
features — see the [Week 4 glossary](../../docs/methodology/glossary.md#week-4--quantization-fundamentals)).
That repacking step, if it happens at load time for some formats and not others,
is itself a plausible mechanism for a *non*-monotonic result — so the naive hypothesis
above is deliberately the one being tested, not assumed.
