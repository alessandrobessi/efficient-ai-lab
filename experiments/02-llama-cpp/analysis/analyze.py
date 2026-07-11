"""Week 2 analysis: turn raw CSVs into summary statistics and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "results" / "raw" / "02-llama-cpp"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "02-llama-cpp"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "02-llama-cpp"


def analyze_2_1() -> None:
    path = RAW_DIR / "exp_2_1_python_vs_llamacpp.csv"
    if not path.exists():
        print(f"skip 2.1: {path} not found")
        return
    df = pd.read_csv(path)

    metrics = ["load_time_s", "ttft_s", "decode_tokens_per_s", "peak_rss_mb"]
    summary = df.groupby("engine")[metrics].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.to_csv(PROCESSED_DIR / "exp_2_1_summary.csv")
    print("Experiment 2.1 summary:")
    print(summary)

    labels = ["load_time_s", "ttft_s", "decode_tokens_per_s", "peak_rss_mb"]
    titles = ["Load time (s)", "TTFT (s)", "Decode speed (tok/s)", "Peak RSS (MB)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    engines = summary.index.tolist()
    colors = {"python": "tab:blue", "llama.cpp": "tab:orange"}
    for ax, metric, title in zip(axes, labels, titles):
        means = [summary.loc[e, f"{metric}_mean"] for e in engines]
        stds = [summary.loc[e, f"{metric}_std"] for e in engines]
        ax.bar(engines, means, yerr=stds, capsize=4, color=[colors[e] for e in engines])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_2_1_python_vs_llamacpp.png", dpi=150)
    plt.close(fig)


def analyze_2_2() -> None:
    path = RAW_DIR / "exp_2_2_thread_scaling.csv"
    if not path.exists():
        print(f"skip 2.2: {path} not found")
        return
    df = pd.read_csv(path)
    df["test_type"] = np.where(df["n_prompt"] > 0, "pp", "tg")
    df = df[["n_threads", "test_type", "avg_ts", "stddev_ts"]]
    df.to_csv(PROCESSED_DIR / "exp_2_2_summary.csv", index=False)
    print("Experiment 2.2 summary:")
    print(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test_type, title, color in [
        (ax1, "pp", "Prompt processing (prefill) vs threads", "tab:blue"),
        (ax2, "tg", "Text generation (decode) vs threads", "tab:orange"),
    ]:
        sub = df[df.test_type == test_type].sort_values("n_threads")
        ax.errorbar(sub.n_threads, sub.avg_ts, yerr=sub.stddev_ts, marker="o", capsize=3, color=color)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Tokens/s")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_2_2_thread_scaling.png", dpi=150)
    plt.close(fig)


def analyze_2_3() -> None:
    path = RAW_DIR / "exp_2_3_repeatability.csv"
    if not path.exists():
        print(f"skip 2.3: {path} not found")
        return
    df = pd.read_csv(path)

    summary = df.groupby("test_type")["tokens_per_s"].agg(["mean", "std", "min", "max"])
    summary["cv_pct"] = 100 * summary["std"] / summary["mean"]
    summary["pearson_r_vs_run"] = [
        df[df.test_type == t]["run"].corr(df[df.test_type == t]["tokens_per_s"]) for t in summary.index
    ]
    summary.to_csv(PROCESSED_DIR / "exp_2_3_summary.csv")
    print("Experiment 2.3 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test_type, title, color in [
        (ax1, "pp", "Prompt processing tok/s per run", "tab:blue"),
        (ax2, "tg", "Text generation tok/s per run", "tab:orange"),
    ]:
        sub = df[df.test_type == test_type].sort_values("run")
        ax.plot(sub.run, sub.tokens_per_s, marker="o", color=color)
        ax.set_xlabel("Run (independent process launch, in order)")
        ax.set_ylabel("Tokens/s")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_2_3_repeatability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analyze_2_1()
    analyze_2_2()
    analyze_2_3()


if __name__ == "__main__":
    main()
