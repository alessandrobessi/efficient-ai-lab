"""Week 3 analysis: turn raw CSVs into summary statistics and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "results" / "raw" / "03-cpu-performance"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "03-cpu-performance"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "03-cpu-performance"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)


def analyze_3_1() -> None:
    path = RAW_DIR / "exp_3_1_thread_scaling.csv"
    if not path.exists():
        print(f"skip 3.1: {path} not found")
        return
    df = pd.read_csv(path)
    df["test_type"] = np.where(df["n_prompt"] > 0, "pp", "tg")
    df = df[["n_threads", "test_type", "avg_ts", "stddev_ts"]].sort_values(["test_type", "n_threads"])
    df.to_csv(PROCESSED_DIR / "exp_3_1_summary.csv", index=False)
    print("Experiment 3.1 summary:")
    print(df)

    p_cores = CONFIG["performance_cores"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test_type, title, color in [
        (ax1, "pp", "Prompt processing (prefill) vs threads", "tab:blue"),
        (ax2, "tg", "Text generation (decode) vs threads", "tab:orange"),
    ]:
        sub = df[df.test_type == test_type]
        ax.errorbar(sub.n_threads, sub.avg_ts, yerr=sub.stddev_ts, marker="o", capsize=3, color=color)
        ax.axvline(p_cores, color="gray", linestyle="--", linewidth=1, label=f"{p_cores} performance cores")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Tokens/s")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_3_1_thread_scaling.png", dpi=150)
    plt.close(fig)


def analyze_3_2() -> None:
    path = RAW_DIR / "exp_3_2_context_scaling.csv"
    if not path.exists():
        print(f"skip 3.2: {path} not found")
        return
    df = pd.read_csv(path)

    metrics = ["ttft_s", "decode_tokens_per_s", "peak_rss_mb"]
    summary = df.groupby("target_prompt_tokens")[metrics].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(PROCESSED_DIR / "exp_3_2_summary.csv", index=False)
    print("Experiment 3.2 summary:")
    print(summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("ttft_s", "TTFT (s)", "tab:blue"),
        ("decode_tokens_per_s", "Decode speed (tok/s)", "tab:orange"),
        ("peak_rss_mb", "Peak RSS (MB)", "tab:green"),
    ]
    for ax, (metric, title, color) in zip(axes, specs):
        ax.errorbar(
            summary["target_prompt_tokens"], summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"],
            marker="o", capsize=3, color=color,
        )
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_3_2_context_scaling.png", dpi=150)
    plt.close(fig)


def analyze_3_3() -> None:
    path = RAW_DIR / "exp_3_3_background_load.csv"
    if not path.exists():
        print(f"skip 3.3: {path} not found")
        return
    df = pd.read_csv(path)
    summary = df.groupby(["background_hogs", "test_type"])["tokens_per_s"].agg(["mean", "std"]).reset_index()
    summary.to_csv(PROCESSED_DIR / "exp_3_3_summary.csv", index=False)
    print("Experiment 3.3 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test_type, title, color in [
        (ax1, "pp", "Prompt processing vs background load", "tab:blue"),
        (ax2, "tg", "Text generation vs background load", "tab:orange"),
    ]:
        sub = summary[summary.test_type == test_type]
        ax.errorbar(sub.background_hogs, sub["mean"], yerr=sub["std"], marker="o", capsize=3, color=color)
        ax.set_xlabel("Concurrent background CPU-hog processes")
        ax.set_ylabel("Tokens/s")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_3_3_background_load.png", dpi=150)
    plt.close(fig)


def analyze_3_4() -> None:
    path = RAW_DIR / "exp_3_4_thermal_effects.csv"
    if not path.exists():
        print(f"skip 3.4: {path} not found")
        return
    df = pd.read_csv(path)

    summary = df.groupby("test_type")["tokens_per_s"].agg(["mean", "std", "min", "max"])
    summary["cv_pct"] = 100 * summary["std"] / summary["mean"]
    summary["pearson_r_vs_run"] = [
        df[df.test_type == t]["run"].corr(df[df.test_type == t]["tokens_per_s"]) for t in summary.index
    ]
    summary.to_csv(PROCESSED_DIR / "exp_3_4_summary.csv")
    print("Experiment 3.4 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test_type, title, color in [
        (ax1, "pp", "Prompt processing tok/s per run (2 threads)", "tab:blue"),
        (ax2, "tg", "Text generation tok/s per run (2 threads)", "tab:orange"),
    ]:
        sub = df[df.test_type == test_type].sort_values("run")
        ax.plot(sub.run, sub.tokens_per_s, marker="o", color=color)
        ax.set_xlabel("Run (independent process launch, in order)")
        ax.set_ylabel("Tokens/s")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_3_4_thermal_effects.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analyze_3_1()
    analyze_3_2()
    analyze_3_3()
    analyze_3_4()


if __name__ == "__main__":
    main()
