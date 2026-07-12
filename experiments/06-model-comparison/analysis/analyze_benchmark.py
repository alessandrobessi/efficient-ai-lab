"""Week 6 speed/memory analysis: turn the raw model_benchmark CSV into summary
statistics and figures, directly mirroring Week 4's analyze.py but grouped by
*model* instead of quantization level."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "results" / "raw" / "06-model-comparison"
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "06-model-comparison"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "06-model-comparison"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)

MODEL_ORDER = [m["name"] for m in CONFIG["models"]]


def main() -> None:
    path = RAW_DIR / "model_benchmark.csv"
    if not path.exists():
        print(f"skip: {path} not found")
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    size_by_model = df.groupby("model", observed=True)[["disk_size_mb", "params_b", "family"]].first()

    metrics = ["load_time_s", "peak_rss_mb", "ttft_s", "decode_tokens_per_s"]
    summary = df.groupby("model", observed=True)[metrics].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = pd.concat([size_by_model, summary], axis=1).sort_index().reset_index()
    summary.to_csv(PROCESSED_DIR / "model_benchmark_summary.csv", index=False)
    print("Model benchmark summary:")
    print(summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(summary.model, summary.disk_size_mb, color="tab:gray")
    axes[0, 0].set_title("Disk size (MB)")
    axes[0, 0].set_ylabel("MB")

    axes[0, 1].bar(
        summary.model, summary.peak_rss_mb_mean, yerr=summary.peak_rss_mb_std, capsize=3, color="tab:green"
    )
    axes[0, 1].set_title("Peak RSS (MB)")
    axes[0, 1].set_ylabel("MB")

    axes[1, 0].bar(
        summary.model, summary.load_time_s_mean, yerr=summary.load_time_s_std, capsize=3, color="tab:blue"
    )
    axes[1, 0].set_title("Load time (s)")
    axes[1, 0].set_ylabel("seconds")

    ax_tps = axes[1, 1]
    ax_tps.bar(
        summary.model, summary.decode_tokens_per_s_mean, yerr=summary.decode_tokens_per_s_std,
        capsize=3, color="tab:orange", label="decode tok/s",
    )
    ax_tps.set_title("Inference performance")
    ax_tps.set_ylabel("decode tok/s")
    ax_ttft = ax_tps.twinx()
    ax_ttft.plot(summary.model, summary.ttft_s_mean, marker="o", color="tab:red", label="TTFT (s)")
    ax_ttft.set_ylabel("TTFT (s)", color="tab:red")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_benchmark.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figure -> {FIGURES_DIR / 'model_benchmark.png'}")


if __name__ == "__main__":
    main()
