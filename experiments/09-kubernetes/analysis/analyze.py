"""Week 9 analysis: turn the raw Kubernetes experiment data (9.1-9.5) into
processed CSVs and figures answering FULL-ROADMAP.md's Week 9 questions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "results" / "raw" / "09-kubernetes"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "09-kubernetes"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "09-kubernetes"

CPU_LEVELS = ["250m", "500m", "1000m", "2000m", "4000m"]
D_SWEEP = [1, 2, 5, 10, 20, 40, 80]


def load_summary(label: str) -> dict:
    with (RAW_DIR / f"{label}_summary.json").open() as f:
        return json.load(f)


def main() -> None:
    if not RAW_DIR.exists():
        print(f"skip: {RAW_DIR} not found")
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 9.1 CPU limits ----------------
    rows = []
    for cpu in CPU_LEVELS:
        s = load_summary(f"exp9_1_cpu_{cpu}")["summary"]
        rows.append({"cpu_limit": cpu, "cpu_millicores": int(cpu.replace("m", "")), **s})
    cpu_df = pd.DataFrame(rows)
    cpu_df.to_csv(PROCESSED_DIR / "exp9_1_cpu_limits.csv", index=False)
    print("9.1 CPU limits:\n", cpu_df[["cpu_limit", "n", "errors", "error_rate", "latency_p50_ms"]], "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ok = cpu_df[cpu_df.error_rate < 1.0]
    ax.plot(ok.cpu_millicores, ok.latency_p50_ms, marker="o", label="p50 latency")
    ax.plot(ok.cpu_millicores, ok.latency_p99_ms, marker="o", label="p99 latency")
    ax.set_xlabel("CPU limit (millicores)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Experiment 9.1 — Latency vs. CPU limit\n(concurrency=1; 250m point: 100% errors, omitted)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp9_1_cpu_limits.png", dpi=150)
    plt.close(fig)

    # ---------------- 9.2 Memory limits ----------------
    mem_rows = [json.loads(l) for l in (RAW_DIR / "exp9_2_memory_limits.jsonl").open()]
    mem_df = pd.DataFrame(mem_rows)
    mem_df.to_csv(PROCESSED_DIR / "exp9_2_memory_limits.csv", index=False)
    print("9.2 Memory limits:\n", mem_df, "\n")

    # ---------------- 9.3 Pod failure ----------------
    fail_rows = []
    for label, replicas in [("exp9_3_1replica", 1), ("exp9_3_2replica", 2)]:
        s = load_summary(label)["summary"]
        with (RAW_DIR / f"{label}_deletion_event.json").open() as f:
            ev = json.load(f)
        recovery_s = (
            pd.Timestamp(ev["recovered_timestamp"]) - pd.Timestamp(ev["delete_timestamp"])
        ).total_seconds()
        fail_rows.append({"replicas": replicas, "recovery_time_s": recovery_s, **s})
    fail_df = pd.DataFrame(fail_rows)
    fail_df.to_csv(PROCESSED_DIR / "exp9_3_pod_failure.csv", index=False)
    print("9.3 Pod failure:\n", fail_df[["replicas", "n", "errors", "error_rate", "recovery_time_s"]], "\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["1 replica", "2 replicas"], fail_df.error_rate * 100, color=["tab:red", "tab:green"])
    ax.set_ylabel("error rate during 60s test with mid-test pod deletion (%)")
    ax.set_title("Experiment 9.3 — Pod Failure: 1 vs. 2 replicas")
    for i, v in enumerate(fail_df.error_rate * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp9_3_pod_failure.png", dpi=150)
    plt.close(fig)

    # ---------------- 9.4 Load saturation on K8s (+ comparison to Week 8 native) ----------------
    sat_rows = []
    for c in D_SWEEP:
        s = load_summary(f"exp9_4_c{c}")["summary"]
        sat_rows.append({"concurrency": c, **s})
    sat_df = pd.DataFrame(sat_rows)
    sat_df.to_csv(PROCESSED_DIR / "exp9_4_saturation.csv", index=False)
    print("9.4 Saturation (K8s):\n", sat_df[["concurrency", "n", "errors", "error_rate", "throughput_rps", "latency_p50_ms"]], "\n")

    week8_path = REPO_ROOT / "results" / "processed" / "08-load-testing" / "workload_summary.csv"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sat_df.concurrency, sat_df.throughput_rps, marker="o", label="Week 9 (Kubernetes)")
    axes[1].plot(sat_df.concurrency, sat_df.error_rate * 100, marker="o", label="Week 9 (Kubernetes)")
    if week8_path.exists():
        w8 = pd.read_csv(week8_path)
        w8_sweep = w8[w8.label.str.startswith("workload_d_c")].sort_values("concurrency")
        axes[0].plot(w8_sweep.concurrency, w8_sweep.throughput_rps, marker="s", linestyle="--", label="Week 8 (native)")
        axes[1].plot(w8_sweep.concurrency, w8_sweep.error_rate * 100, marker="s", linestyle="--", label="Week 8 (native)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("concurrency")
    axes[0].set_ylabel("throughput (req/s)")
    axes[0].set_title("Throughput vs. concurrency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("concurrency")
    axes[1].set_ylabel("error rate (%)")
    axes[1].set_title("Error rate vs. concurrency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp9_4_saturation_vs_week8.png", dpi=150)
    plt.close(fig)

    # ---------------- 9.5 Horizontal scaling ----------------
    scale_rows = []
    for replicas in [1, 2, 3]:
        s_fixed = load_summary(f"exp9_5_replicas{replicas}")["summary"]
        s_scaled = load_summary(f"exp9_5b_replicas{replicas}")["summary"]
        scale_rows.append({
            "replicas": replicas,
            "throughput_fixed_concurrency20": s_fixed["throughput_rps"],
            "error_rate_fixed_concurrency20": s_fixed["error_rate"],
            "throughput_scaled_concurrency": s_scaled["throughput_rps"],
            "error_rate_scaled_concurrency": s_scaled["error_rate"],
        })
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(PROCESSED_DIR / "exp9_5_horizontal_scaling.csv", index=False)
    print("9.5 Horizontal scaling:\n", scale_df, "\n")

    startup_rows = [json.loads(l) for l in (RAW_DIR / "exp9_5_startup_cost.jsonl").open()]
    for r in startup_rows:
        r["startup_s"] = (pd.Timestamp(r["all_ready"]) - pd.Timestamp(r["scale_start"])).total_seconds()
    startup_df = pd.DataFrame(startup_rows)
    startup_df.to_csv(PROCESSED_DIR / "exp9_5_startup_cost.csv", index=False)
    print("9.5 Startup cost:\n", startup_df[["replicas", "startup_s"]], "\n")

    mem_per_replica = pd.DataFrame([json.loads(l) for l in (RAW_DIR / "exp9_5_memory_per_replica.jsonl").open()])
    mem_per_replica.to_csv(PROCESSED_DIR / "exp9_5_memory_per_replica.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(scale_df.replicas, scale_df.throughput_fixed_concurrency20, marker="o", label="fixed concurrency=20")
    axes[0].plot(scale_df.replicas, scale_df.throughput_scaled_concurrency, marker="s", label="concurrency=20×replicas")
    ideal = scale_df.throughput_scaled_concurrency.iloc[0] * scale_df.replicas
    axes[0].plot(scale_df.replicas, ideal, linestyle=":", color="gray", label="ideal linear scaling")
    axes[0].set_xlabel("replicas")
    axes[0].set_ylabel("throughput (req/s)")
    axes[0].set_title("Experiment 9.5 — Throughput vs. replica count")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    mem_per_replica["memory_mb"] = mem_per_replica["memory_usage"].str.replace("Mi", "").astype(float)
    for replicas, grp in mem_per_replica.groupby("replicas"):
        axes[1].scatter([replicas] * len(grp), grp.memory_mb, color="tab:blue")
    totals = mem_per_replica.groupby("replicas").memory_mb.sum()
    axes[1].plot(totals.index, totals.values, marker="D", color="tab:red", label="total across replicas")
    axes[1].set_xlabel("replicas")
    axes[1].set_ylabel("per-pod memory (MB)")
    axes[1].set_title("Experiment 9.5 — Memory duplication")
    axes[1].set_xticks([1, 2, 3])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp9_5_horizontal_scaling.png", dpi=150)
    plt.close(fig)

    print("All Week 9 figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
