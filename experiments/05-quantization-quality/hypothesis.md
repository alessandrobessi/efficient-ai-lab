# Hypotheses — Week 5: Quantization vs Quality

## Overall (naive)

Quality should degrade monotonically as quantization gets more aggressive, mirroring
Week 4's naive (and ultimately falsified) assumption about speed/memory: F16 should
score highest, Q3_K_M lowest, with the middle formats falling in between roughly in
bit-width order.

## Central hypothesis

> Mean quality score decreases monotonically from F16 to Q3_K_M, and the drop is
> small and roughly uniform across task categories.

Falsifiable by: any quantization level scoring higher than a "higher-precision" one
by more than noise (bootstrap CI overlap), or quality loss concentrating heavily in
specific categories rather than being roughly uniform.

## Why this is worth testing rather than assuming

Week 4 already falsified the equivalent naive hypothesis for speed and memory —
Q4_K_M was the fastest decode format tested, not the least quantized, and Q8_0 used
more peak memory than F16. That result came from format-specific interactions with
this CPU's SIMD kernels (block layout, `REPACK`), which have nothing to do with
numerical accuracy. There's no guarantee quality degrades in the same clean,
monotonic way bit-width alone would suggest — in particular:

- **The `_K` formats' more elaborate per-block scaling** (see [Week 4
  glossary](../../docs/methodology/glossary.md#week-4--quantization-fundamentals))
  could make quality loss non-monotonic even if bit-width is monotonic, the same way
  it made speed non-monotonic.
- **Task categories are unlikely to degrade uniformly.** Closed-label classification
  and short numeric reasoning answers have little room for a small perturbation in
  weights to change the output; free-form JSON generation and multi-field structured
  output have many more ways to go wrong (a single dropped comma breaks JSON
  validity), so quality loss is plausibly concentrated in the more structurally
  fragile categories rather than spread evenly.
- **This model is already small (1.5B parameters).** Quantization error compounds
  differently on a model with fewer redundant parameters to begin with than it would
  on a much larger model — there's no strong prior for how steep the quality cliff is
  at this scale.

## Sub-hypotheses (by category)

- **classification, reasoning** (closed-form, short answers): quality should be the
  *most* robust to quantization — little room for phrasing drift to break scoring.
- **information_extraction, structured_output** (JSON validity required): quality
  should be the *least* robust — JSON syntax errors are a hard failure mode that gets
  more likely as weight noise increases, independent of whether the model "knows" the
  right answer.
- **summarization** (continuous, lexical-overlap-scored): expected to degrade
  gradually rather than sharply, since token-F1 partially credits close-but-imperfect
  outputs.
- **instruction_following** (format-constrained): expected to sit between the two
  extremes — content is often simple, but exact format compliance (e.g. `single_emoji`,
  `all_caps`) is itself a fragile, low-redundancy signal that quantization noise could
  disrupt even when the underlying "answer" is understood correctly.
