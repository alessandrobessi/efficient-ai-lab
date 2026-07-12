"""Week 5 analysis: turn scored_outputs.csv into summary statistics, bootstrap
confidence intervals, paired effect sizes vs the F16 baseline, a failure-category
breakdown, and the quality-vs-performance Pareto plot (joining Week 4's decode-speed
numbers)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "05-quantization-quality"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "05-quantization-quality"
WEEK4_SUMMARY = REPO_ROOT / "results" / "processed" / "04-quantization" / "quantization_benchmark_summary.csv"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)

QUANT_ORDER = [level["name"] for level in CONFIG["quant_levels"]]
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    path = PROCESSED_DIR / "scored_outputs.csv"
    if not path.exists():
        print(f"skip: {path} not found")
        return

    df = pd.read_csv(path)
    df["quant_level"] = pd.Categorical(df["quant_level"], categories=QUANT_ORDER, ordered=True)

    # --- Overall summary per quant level, with bootstrap 95% CI ---
    overall_rows = []
    for level in QUANT_ORDER:
        scores = df.loc[df.quant_level == level, "score"].to_numpy()
        if len(scores) == 0:
            continue
        lo, hi = bootstrap_ci(scores)
        overall_rows.append(
            {
                "quant_level": level,
                "n": len(scores),
                "mean_score": scores.mean(),
                "std_score": scores.std(ddof=1),
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(PROCESSED_DIR / "overall_summary.csv", index=False)
    print("Overall summary:\n", overall, "\n")

    # --- Per-category summary per quant level ---
    per_category = (
        df.groupby(["quant_level", "category"], observed=True)["score"].agg(["mean", "std", "count"]).reset_index()
    )
    per_category.to_csv(PROCESSED_DIR / "per_category_summary.csv", index=False)

    # --- Paired effect size vs F16 baseline ---
    wide = df.pivot(index="id", columns="quant_level", values="score")
    effect_rows = []
    baseline = wide["F16"]
    for level in QUANT_ORDER:
        if level == "F16":
            continue
        diffs = (wide[level] - baseline).to_numpy()
        mean_diff = diffs.mean()
        sd_diff = diffs.std(ddof=1)
        cohens_dz = mean_diff / sd_diff if sd_diff > 0 else float("nan")
        lo, hi = paired_bootstrap_diff_ci(diffs)
        effect_rows.append(
            {
                "quant_level": level,
                "mean_score_diff_vs_f16": mean_diff,
                "cohens_dz": cohens_dz,
                "diff_ci95_lo": lo,
                "diff_ci95_hi": hi,
            }
        )
    effect_sizes = pd.DataFrame(effect_rows)
    effect_sizes.to_csv(PROCESSED_DIR / "effect_sizes_vs_f16.csv", index=False)
    print("Effect sizes vs F16:\n", effect_sizes, "\n")

    # --- JSON validity rate (information_extraction + structured_output) ---
    json_df = df[df.category.isin(["information_extraction", "structured_output"])]
    json_validity = (
        json_df.assign(valid=json_df.valid_json.astype(str) == "True")
        .groupby("quant_level", observed=True)["valid"]
        .mean()
        .reset_index()
        .rename(columns={"valid": "json_validity_rate"})
    )
    json_validity.to_csv(PROCESSED_DIR / "json_validity.csv", index=False)

    # --- Failure category breakdown: top failure reasons per quant level ---
    failures = df[df.score < 1.0]
    failure_counts = (
        failures.groupby(["quant_level", "category", "detail"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["quant_level", "count"], ascending=[True, False])
    )
    failure_counts.to_csv(PROCESSED_DIR / "failure_breakdown.csv", index=False)

    # --- Pareto plot: quality vs decode speed (joining Week 4 processed data) ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(WEEK4_SUMMARY)
    pareto = overall.merge(perf[["quant_level", "decode_tokens_per_s_mean"]], on="quant_level")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(
        pareto.decode_tokens_per_s_mean,
        pareto.mean_score,
        yerr=[pareto.mean_score - pareto.ci95_lo, pareto.ci95_hi - pareto.mean_score],
        fmt="o",
        capsize=4,
        color="tab:blue",
    )
    for _, row in pareto.iterrows():
        ax1.annotate(row.quant_level, (row.decode_tokens_per_s_mean, row.mean_score), textcoords="offset points", xytext=(6, 4))
    ax1.set_xlabel("decode speed (tok/s) — Week 4")
    ax1.set_ylabel("mean quality score (this week, 95% CI)")
    ax1.set_title("Quality vs. Performance")
    ax1.grid(True, alpha=0.3)

    cat_pivot = per_category.pivot(index="category", columns="quant_level", values="mean")
    cat_pivot = cat_pivot[[c for c in QUANT_ORDER if c in cat_pivot.columns]]
    cat_pivot.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("mean score")
    ax2.set_title("Mean score by category")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_vs_performance.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figures -> {FIGURES_DIR / 'quality_vs_performance.png'}")


if __name__ == "__main__":
    main()
