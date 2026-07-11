# Open Questions Tracker

Every experiment README ends with a "§11 What new questions emerged?" section (see
root `README.md` §6). This file is the running register of those questions across all
weeks, so they don't just sit at the bottom of a given week's README and get
forgotten — later weeks are expected to actively try to close them out, not just
generate new ones.

**Process:**

- **Before** designing a new week's experiments, check this register for `Open` /
  `Partially answered` questions relevant to that week's topic, and fold them into
  that week's hypotheses/experiments where it fits the week's intended scope.
- **After** finishing a week's report, append its own new §11 questions here as new
  `Open` rows.
- **When** a week's results bear on an earlier open question — even if it wasn't that
  question's "officially" planned week — update the row's Status/Resolution here, and
  keep the cross-reference in that week's own README interpretation section too.

Status values: `Open`, `Partially answered`, `Answered`.

## Register

| # | Question | Raised in | Status | Resolution |
|---|---|---|---|---|
| 1 | Is the decode-speed noise at short prompt lengths a CPU frequency-ramp/scheduling effect, or something else (e.g. memory allocator warm-up)? | Week 1, [Exp 1.2](../../experiments/01-inference-basics/README.md#9-how-should-the-results-be-interpreted) | Open | — |
| 2 | How does the Python/Transformers baseline compare to llama.cpp on the same model and hardware? | Week 1, [Exp 1.2](../../experiments/01-inference-basics/README.md#11-what-new-questions-emerged) | **Answered** | Week 2, [Exp 2.1](../../experiments/02-llama-cpp/README.md#8-what-are-the-results): llama.cpp is ~3.8x faster to load, ~3.2x faster TTFT, ~1.7x faster decode, ~2.3x lower peak RSS. |
| 3 | At what thread count does decode speed stop improving on this 10-core machine? | Week 1, [Exp 1.2](../../experiments/01-inference-basics/README.md#11-what-new-questions-emerged) | Partially answered | Week 2, [Exp 2.2](../../experiments/02-llama-cpp/README.md#9-how-should-the-results-be-interpreted) found throughput peaks at just **2 threads** and *degrades* at 8–10 (not a simple plateau) — likely P-core/E-core scheduling on Apple M4. Week 3 Exp 3.1 is the planned formal follow-up to confirm the mechanism. |
| 4 | Does the roughly-linear total-latency-vs-output-length relationship hold at much longer output lengths (2048+ tokens), or does something change (thermal throttling, memory pressure)? | Week 1, [Exp 1.3](../../experiments/01-inference-basics/README.md#11-what-new-questions-emerged) | Open | Related signal in Week 2, Exp 2.3 (see #7 below) — not yet tested directly on the Week 1 Python path or at 2048+ tokens. |
| 5 | Is the Experiment 2.2 thread-scaling collapse really P-core/E-core scheduling? | Week 2, [Exp 2.2](../../experiments/02-llama-cpp/README.md#11-what-new-questions-emerged) | Open | Planned for Week 3 (`taskpolicy`/`powermetrics` core-level utilization during a thread sweep). |
| 6 | How much of llama.cpp's advantage in Exp 2.1 is the engine vs. the F16-vs-fp32 precision difference? | Week 2, [Exp 2.1](../../experiments/02-llama-cpp/README.md#11-what-new-questions-emerged) | Open | Needs a Python fp16/bf16 CPU baseline or an llama.cpp fp32 GGUF; realistically answerable once Weeks 4–5 add multiple quantization levels. |
| 7 | Is Experiment 2.3's throughput decline actually thermal throttling? | Week 2, [Exp 2.3](../../experiments/02-llama-cpp/README.md#11-what-new-questions-emerged) | Open | Planned for Week 3 Exp 3.4 (sample core temperature/clock frequency alongside throughput). |
| 8 | Does the Exp 2.3 decline-then-plateau pattern repeat if the machine cools between runs, or at the Exp 2.2-optimal 2 threads instead of 10? | Week 2, [Exp 2.3](../../experiments/02-llama-cpp/README.md#11-what-new-questions-emerged) | Open | Would help separate "thermal effect" from "bad thread count" as contributing causes — good Week 3 candidate. |
| 9 | How do llama-server's HTTP-level latencies compare to the llama-cli-level numbers measured in Week 2? | Week 2, [Exp 2.1](../../experiments/02-llama-cpp/README.md#11-what-new-questions-emerged) | Open | Relevant once the Week 7 Go gateway is built on top of llama-server. |
