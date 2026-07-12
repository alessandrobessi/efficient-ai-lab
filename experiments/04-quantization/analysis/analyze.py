"""Week 4 analysis: turn the raw quantization benchmark CSV into summary
statistics and figures covering Experiments 4.1 (size), 4.2 (memory),
4.3 (loading), and 4.4 (inference performance)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "results" / "raw" / "04-quantization"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "04-quantization"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "04-quantization"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Preserve F16 -> Q3_K_M ordering (largest/highest-fidelity to smallest) in plots,
# rather than whatever order groupby produces.
QUANT_ORDER = [level["name"] for level in CONFIG["quant_levels"]]


def main() -> None:
    path = RAW_DIR / "quantization_benchmark.csv"
    if not path.exists():
        print(f"skip: {path} not found")
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df["quant_level"] = pd.Categorical(df["quant_level"], categories=QUANT_ORDER, ordered=True)

    # disk_size_mb is constant per level (not a repeated measurement) - take first.
    size_by_level = df.groupby("quant_level", observed=True)["disk_size_mb"].first()

    metrics = ["load_time_s", "peak_rss_mb", "ttft_s", "decode_tokens_per_s"]
    summary = df.groupby("quant_level", observed=True)[metrics].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "disk_size_mb", size_by_level)
    summary = summary.sort_index().reset_index()
    summary.to_csv(PROCESSED_DIR / "quantization_benchmark_summary.csv", index=False)
    print("Quantization benchmark summary:")
    print(summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(summary.quant_level, summary.disk_size_mb, color="tab:gray")
    axes[0, 0].set_title("4.1 — Disk size (MB)")
    axes[0, 0].set_ylabel("MB")

    axes[0, 1].bar(
        summary.quant_level, summary.peak_rss_mb_mean, yerr=summary.peak_rss_mb_std, capsize=3, color="tab:green"
    )
    axes[0, 1].set_title("4.2 — Peak RSS (MB)")
    axes[0, 1].set_ylabel("MB")

    axes[1, 0].bar(
        summary.quant_level, summary.load_time_s_mean, yerr=summary.load_time_s_std, capsize=3, color="tab:blue"
    )
    axes[1, 0].set_title("4.3 — Load time (s)")
    axes[1, 0].set_ylabel("seconds")

    ax_tps = axes[1, 1]
    ax_tps.bar(
        summary.quant_level, summary.decode_tokens_per_s_mean, yerr=summary.decode_tokens_per_s_std,
        capsize=3, color="tab:orange", label="decode tok/s",
    )
    ax_tps.set_title("4.4 — Inference performance")
    ax_tps.set_ylabel("decode tok/s")
    ax_ttft = ax_tps.twinx()
    ax_ttft.plot(summary.quant_level, summary.ttft_s_mean, marker="o", color="tab:red", label="TTFT (s)")
    ax_ttft.set_ylabel("TTFT (s)", color="tab:red")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quantization_benchmark.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
