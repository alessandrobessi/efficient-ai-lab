"""Week 8 analysis: turn the load generator's raw per-workload summary JSON
files into a processed CSV and the figures that answer FULL-ROADMAP.md's
Week 8 analysis questions (when does throughput stop increasing, when does
latency become unacceptable, what happens to p99, does the system queue
requests, does CPU reach 100%, what's the bottleneck)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "results" / "raw" / "08-load-testing"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "08-load-testing"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "08-load-testing"

# D-sweep concurrency levels, in the order run_workloads.sh executed them.
D_SWEEP_CONCURRENCY = [1, 2, 5, 10, 20, 40, 80]


def load_summary(label: str) -> dict:
    path = RAW_DIR / f"{label}_summary.json"
    with path.open() as f:
        return json.load(f)


def main() -> None:
    if not RAW_DIR.exists():
        print(f"skip: {RAW_DIR} not found")
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Workloads A/B/C + D sweep into one table ---
    rows = []
    for label, concurrency in [("workload_a", 1), ("workload_b", 5), ("workload_c", 20)]:
        d = load_summary(label)
        s = d["summary"]
        rows.append({"label": label, "concurrency": concurrency, **s})
    for c in D_SWEEP_CONCURRENCY:
        label = f"workload_d_c{c}"
        d = load_summary(label)
        s = d["summary"]
        rows.append({"label": label, "concurrency": c, **s})

    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "workload_summary.csv", index=False)
    print("Workload summary:\n", df[["label", "concurrency", "n", "errors", "error_rate",
                                       "throughput_rps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]])

    # --- D-sweep only, for the collapse figures ---
    sweep = df[df.label.str.startswith("workload_d_c")].sort_values("concurrency")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(sweep.concurrency, sweep.throughput_rps, marker="o", color="tab:blue")
    ax.set_xlabel("concurrency (concurrent clients)")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Throughput vs. concurrency")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sweep.concurrency, sweep.latency_p50_ms, marker="o", label="p50")
    ax.plot(sweep.concurrency, sweep.latency_p95_ms, marker="o", label="p95")
    ax.plot(sweep.concurrency, sweep.latency_p99_ms, marker="o", label="p99")
    ax.set_xlabel("concurrency (concurrent clients)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency percentiles vs. concurrency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(sweep.concurrency, sweep.error_rate * 100, marker="o", color="tab:red")
    ax.set_xlabel("concurrency (concurrent clients)")
    ax.set_ylabel("error rate (%)")
    ax.set_title("Error rate vs. concurrency (collapse)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "saturation_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'saturation_sweep.png'}")

    # --- Coordinated omission: naive vs. corrected latency ---
    co_path = RAW_DIR / "coordinated_omission_demo_summary.json"
    if co_path.exists():
        with co_path.open() as f:
            co = json.load(f)["summary"]

        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["p50", "p95", "p99"]
        naive = [co["latency_p50_ms"], co["latency_p95_ms"], co["latency_p99_ms"]]
        corrected = [co["corrected_latency_p50_ms"], co["corrected_latency_p95_ms"], co["corrected_latency_p99_ms"]]
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], naive, width, label="naive (DoneAt - SentAt)", color="tab:green")
        ax.bar([i + width / 2 for i in x], corrected, width, label="corrected (DoneAt - ScheduledAt)", color="tab:red")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("latency (ms, log scale)")
        ax.set_title("Coordinated omission: naive vs. corrected latency\n(open-loop, 3 req/s target, 2 sender slots)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "coordinated_omission.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIGURES_DIR / 'coordinated_omission.png'}")

        pd.DataFrame([co]).to_csv(PROCESSED_DIR / "coordinated_omission_summary.csv", index=False)


if __name__ == "__main__":
    main()
