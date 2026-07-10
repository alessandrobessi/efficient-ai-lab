"""Week 1 analysis: turn raw CSVs into summary statistics and figures.

Reads results/raw/01-inference-basics/*.csv, writes summary statistics to
results/processed/01-inference-basics/*.csv and figures to
results/figures/01-inference-basics/*.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "results" / "raw" / "01-inference-basics"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "01-inference-basics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "01-inference-basics"


def summarize(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_col)[value_cols].agg(["mean", "median", "std", "min", "max"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def analyze_1_1() -> None:
    path = RAW_DIR / "exp_1_1_loading_time.csv"
    if not path.exists():
        print(f"skip 1.1: {path} not found")
        return
    df = pd.read_csv(path)

    summary = df[["load_time_s", "mem_before_mb", "mem_after_mb", "mem_delta_mb"]].agg(
        ["mean", "median", "std", "min", "max"]
    )
    summary.to_csv(PROCESSED_DIR / "exp_1_1_summary.csv")
    print("Experiment 1.1 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(df["run"], df["load_time_s"], marker="o")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Load time (s)")
    ax1.set_title("Model loading time per run")

    ax2.plot(df["run"], df["mem_before_mb"], marker="o", label="before")
    ax2.plot(df["run"], df["mem_after_mb"], marker="o", label="after")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Process RSS (MB)")
    ax2.set_title("Process memory before/after load")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_1_1_loading_time.png", dpi=150)
    plt.close(fig)


def analyze_1_2() -> None:
    path = RAW_DIR / "exp_1_2_prompt_length.csv"
    if not path.exists():
        print(f"skip 1.2: {path} not found")
        return
    df = pd.read_csv(path)

    summary = summarize(
        df, "target_prompt_tokens", ["ttft_s", "decode_time_s", "total_time_s", "decode_tokens_per_s"]
    )
    summary.to_csv(PROCESSED_DIR / "exp_1_2_summary.csv", index=False)
    print("Experiment 1.2 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(
        summary["target_prompt_tokens"],
        summary["ttft_s_mean"],
        yerr=summary["ttft_s_std"],
        marker="o",
        capsize=3,
    )
    ax1.set_xlabel("Prompt length (tokens)")
    ax1.set_ylabel("Time to First Token (s)")
    ax1.set_title("TTFT vs prompt length (prefill cost)")

    ax2.errorbar(
        summary["target_prompt_tokens"],
        summary["decode_tokens_per_s_mean"],
        yerr=summary["decode_tokens_per_s_std"],
        marker="o",
        capsize=3,
        color="tab:orange",
    )
    ax2.set_xlabel("Prompt length (tokens)")
    ax2.set_ylabel("Decode speed (tokens/s)")
    ax2.set_title("Decode speed vs prompt length")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_1_2_prompt_length.png", dpi=150)
    plt.close(fig)


def analyze_1_3() -> None:
    path = RAW_DIR / "exp_1_3_output_length.csv"
    if not path.exists():
        print(f"skip 1.3: {path} not found")
        return
    df = pd.read_csv(path)

    summary = summarize(
        df, "requested_output_tokens", ["ttft_s", "decode_time_s", "total_time_s", "decode_tokens_per_s"]
    )
    summary.to_csv(PROCESSED_DIR / "exp_1_3_summary.csv", index=False)
    print("Experiment 1.3 summary:")
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(
        summary["requested_output_tokens"],
        summary["total_time_s_mean"],
        yerr=summary["total_time_s_std"],
        marker="o",
        capsize=3,
    )
    ax1.set_xlabel("Output length (tokens)")
    ax1.set_ylabel("Total latency (s)")
    ax1.set_title("Total latency vs output length")

    ax2.errorbar(
        summary["requested_output_tokens"],
        summary["decode_tokens_per_s_mean"],
        yerr=summary["decode_tokens_per_s_std"],
        marker="o",
        capsize=3,
        color="tab:orange",
    )
    ax2.set_xlabel("Output length (tokens)")
    ax2.set_ylabel("Decode speed (tokens/s)")
    ax2.set_title("Decode speed vs output length")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_1_3_output_length.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analyze_1_1()
    analyze_1_2()
    analyze_1_3()


if __name__ == "__main__":
    main()
