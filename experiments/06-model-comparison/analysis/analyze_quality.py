"""Week 6 quality analysis: turn scored_outputs.csv (+ this week's own
model_benchmark_summary.csv) into the statistics needed to answer
FULL-ROADMAP.md's Week 6 analysis questions — does parameter count predict
quality/latency, which models give the best quality/performance tradeoff, which
are best in Italian, which produce the most reliable structured output, and
which families are most CPU-friendly (tokens/sec per billion parameters)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "06-model-comparison"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "06-model-comparison"
BENCHMARK_SUMMARY = PROCESSED_DIR / "model_benchmark_summary.csv"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)

MODEL_ORDER = [m["name"] for m in CONFIG["models"]]
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    path = PROCESSED_DIR / "scored_outputs.csv"
    if not path.exists():
        print(f"skip: {path} not found")
        return

    df = pd.read_csv(path)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    # --- Overall summary per model, with bootstrap 95% CI ---
    overall_rows = []
    for model in MODEL_ORDER:
        sub = df[df.model == model]
        scores = sub["score"].to_numpy()
        if len(scores) == 0:
            continue
        lo, hi = bootstrap_ci(scores)
        overall_rows.append(
            {
                "model": model,
                "family": sub["family"].iloc[0],
                "params_b": sub["params_b"].iloc[0],
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

    # --- Per-category summary per model ---
    per_category = (
        df.groupby(["model", "category"], observed=True)["score"].agg(["mean", "std", "count"]).reset_index()
    )
    per_category.to_csv(PROCESSED_DIR / "per_category_summary.csv", index=False)

    # --- Per-language summary per model (which models are best in Italian?) ---
    per_language = (
        df.groupby(["model", "language"], observed=True)["score"].agg(["mean", "std", "count"]).reset_index()
    )
    per_language.to_csv(PROCESSED_DIR / "per_language_summary.csv", index=False)
    print("Per-language summary:\n", per_language.pivot(index="model", columns="language", values="mean"), "\n")

    # --- JSON validity (structured_output + information_extraction reliability) ---
    json_df = df[df.category.isin(["information_extraction", "structured_output"])]
    json_validity = (
        json_df.assign(valid=json_df.valid_json.astype(str) == "True")
        .groupby("model", observed=True)["valid"]
        .mean()
        .reset_index()
        .rename(columns={"valid": "json_validity_rate"})
    )
    json_validity.to_csv(PROCESSED_DIR / "json_validity.csv", index=False)
    print("JSON validity rate (structured tasks):\n", json_validity, "\n")

    # --- Failure category breakdown ---
    failures = df[df.score < 1.0]
    failure_counts = (
        failures.groupby(["model", "category", "detail"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )
    failure_counts.to_csv(PROCESSED_DIR / "failure_breakdown.csv", index=False)

    # --- Join this week's own speed/memory benchmark ---
    if not BENCHMARK_SUMMARY.exists():
        print(f"skip correlations/Pareto: {BENCHMARK_SUMMARY} not found")
        return
    perf = pd.read_csv(BENCHMARK_SUMMARY)
    joined = overall.merge(
        perf[["model", "decode_tokens_per_s_mean", "peak_rss_mb_mean", "load_time_s_mean", "disk_size_mb"]],
        on="model",
    )
    joined["tokens_per_s_per_b_params"] = joined.decode_tokens_per_s_mean / joined.params_b
    joined.to_csv(PROCESSED_DIR / "quality_performance_joined.csv", index=False)

    # --- Does parameter count predict quality / latency? (n=5 models — descriptive, not confirmatory) ---
    r_quality, p_quality = stats.pearsonr(joined.params_b, joined.mean_score)
    r_speed, p_speed = stats.pearsonr(joined.params_b, joined.decode_tokens_per_s_mean)
    correlations = pd.DataFrame(
        [
            {"comparison": "params_b vs mean_score", "pearson_r": r_quality, "p_value": p_quality, "n": len(joined)},
            {"comparison": "params_b vs decode_tokens_per_s", "pearson_r": r_speed, "p_value": p_speed, "n": len(joined)},
        ]
    )
    correlations.to_csv(PROCESSED_DIR / "correlations.csv", index=False)
    print("Correlations (n=5, descriptive only):\n", correlations, "\n")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.errorbar(
        joined.decode_tokens_per_s_mean,
        joined.mean_score,
        yerr=[joined.mean_score - joined.ci95_lo, joined.ci95_hi - joined.mean_score],
        fmt="o",
        capsize=4,
        color="tab:blue",
    )
    for _, row in joined.iterrows():
        ax.annotate(f"{row.model}\n({row.family})", (row.decode_tokens_per_s_mean, row.mean_score),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("decode speed (tok/s)")
    ax.set_ylabel("mean quality score (95% CI)")
    ax.set_title("Quality vs. Performance (Pareto)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(joined.params_b, joined.mean_score, color="tab:purple")
    for _, row in joined.iterrows():
        ax.annotate(row.model, (row.params_b, row.mean_score), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("mean quality score")
    ax.set_title(f"Params vs. Quality (r={r_quality:.2f}, n=5)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(joined.params_b, joined.decode_tokens_per_s_mean, color="tab:orange")
    for _, row in joined.iterrows():
        ax.annotate(row.model, (row.params_b, row.decode_tokens_per_s_mean), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("decode speed (tok/s)")
    ax.set_title(f"Params vs. Speed (r={r_speed:.2f}, n=5)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cat_pivot = per_category.pivot(index="category", columns="model", values="mean")
    cat_pivot = cat_pivot[[c for c in MODEL_ORDER if c in cat_pivot.columns]]
    cat_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean score")
    ax.set_title("Mean score by category")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_vs_performance.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figure -> {FIGURES_DIR / 'quality_vs_performance.png'}")


if __name__ == "__main__":
    main()
