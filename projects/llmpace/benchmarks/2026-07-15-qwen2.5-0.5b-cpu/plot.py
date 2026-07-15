"""Generates the 4 saturation-curve charts from results.csv. Run from this
directory: `uv run --project ../../../.. python plot.py` (uses the repo
root's pandas/matplotlib dependencies — see ../../../../pyproject.toml).
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv").sort_values("requests_per_second")
rate = df["requests_per_second"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0][0]
ax.plot(rate, rate, "--", color="gray", label="offered rate (ideal)")
ax.plot(rate, df["throughput_rps"], "o-", color="tab:blue", label="achieved throughput")
ax.set_xlabel("offered request rate (req/s)")
ax.set_ylabel("throughput (req/s)")
ax.set_title("Throughput vs. offered rate")
ax.legend()

ax = axes[0][1]
ax.plot(rate, df["naive_p99_ms"], "o-", color="tab:orange", label="naive p99 (service)")
ax.plot(rate, df["corrected_p99_ms"], "o-", color="tab:red", label="corrected p99 (scheduled)")
ax.set_xlabel("offered request rate (req/s)")
ax.set_ylabel("total latency p99 (ms)")
ax.set_title("Latency vs. offered rate")
ax.legend()

ax = axes[1][0]
ax.plot(rate, df["naive_ttft_p99_ms"], "o-", color="tab:orange", label="naive TTFT p99 (service)")
ax.plot(rate, df["corrected_ttft_p99_ms"], "o-", color="tab:red", label="corrected TTFT p99 (scheduled)")
ax.set_xlabel("offered request rate (req/s)")
ax.set_ylabel("TTFT p99 (ms)")
ax.set_title("Time-to-first-token vs. offered rate")
ax.legend()

ax = axes[1][1]
ax.plot(rate, df["peak_queue_depth"], "o-", color="tab:purple", label="peak queue depth")
ax.set_xlabel("offered request rate (req/s)")
ax.set_ylabel("peak queue depth (requests)")
ax.set_title("Client-side backlog vs. offered rate")
ax.legend()

fig.suptitle("llmpace saturation sweep: Qwen2.5-0.5B-Instruct Q4_K_M, Apple M4 CPU, llama.cpp b9960", fontsize=11)
fig.tight_layout()
fig.savefig("saturation_curves.png", dpi=150)
print("Wrote saturation_curves.png")
