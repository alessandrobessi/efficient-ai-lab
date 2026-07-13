"""Week 10 analysis: quality, latency, and throughput comparison between the
2 local systems (CPU-SLM, Larger-Local), joined with the speed/memory
benchmark and the cost model, producing the figures for the comparison
report. The frontier API is not part of this script's measured data — see
cost_model.py and README.md for how it's handled."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "10-slm-vs-llm"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "10-slm-vs-llm"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)

SYSTEM_ORDER = [s["name"] for s in CONFIG["local_systems"]]
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
    scored_path = PROCESSED_DIR / "scored_outputs.csv"
    bench_path = PROCESSED_DIR.parent.parent / "raw" / "10-slm-vs-llm" / "system_benchmark.csv"
    if not scored_path.exists():
        print(f"skip: {scored_path} not found")
        return

    df = pd.read_csv(scored_path)
    df["system"] = pd.Categorical(df["system"], categories=SYSTEM_ORDER, ordered=True)

    # --- Overall quality summary ---
    overall_rows = []
    for system in SYSTEM_ORDER:
        scores = df.loc[df.system == system, "score"].to_numpy()
        if len(scores) == 0:
            continue
        lo, hi = bootstrap_ci(scores)
        overall_rows.append({"system": system, "n": len(scores), "mean_score": scores.mean(),
                              "ci95_lo": lo, "ci95_hi": hi})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(PROCESSED_DIR / "overall_summary.csv", index=False)
    print("Overall quality summary:\n", overall, "\n")

    # --- Paired effect size (both systems scored on the identical 100 examples) ---
    if len(SYSTEM_ORDER) == 2:
        wide = df.pivot(index="id", columns="system", values="score")
        diffs = (wide[SYSTEM_ORDER[1]] - wide[SYSTEM_ORDER[0]]).to_numpy()
        mean_diff = diffs.mean()
        sd_diff = diffs.std(ddof=1)
        cohens_dz = mean_diff / sd_diff if sd_diff > 0 else float("nan")
        n = len(diffs)
        boot = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = RNG.integers(0, n, size=n)
            boot[i] = diffs[idx].mean()
        diff_lo, diff_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        effect = pd.DataFrame([{
            "comparison": f"{SYSTEM_ORDER[1]} - {SYSTEM_ORDER[0]}",
            "mean_diff": mean_diff, "cohens_dz": cohens_dz,
            "diff_ci95_lo": diff_lo, "diff_ci95_hi": diff_hi,
        }])
        effect.to_csv(PROCESSED_DIR / "paired_effect_size.csv", index=False)
        print("Paired effect size:\n", effect, "\n")

    per_category = df.groupby(["system", "category"], observed=True)["score"].mean().reset_index()
    per_category.to_csv(PROCESSED_DIR / "per_category_summary.csv", index=False)

    per_language = df.groupby(["system", "language"], observed=True)["score"].mean().reset_index()
    per_language.to_csv(PROCESSED_DIR / "per_language_summary.csv", index=False)

    # --- Join with speed/memory benchmark ---
    if not bench_path.exists():
        print(f"skip figures: {bench_path} not found")
        return
    bench = pd.read_csv(bench_path)
    bench_summary = bench.groupby("system", observed=True).agg(
        disk_size_mb=("disk_size_mb", "first"),
        peak_rss_mb_mean=("peak_rss_mb", "mean"),
        decode_tokens_per_s_mean=("decode_tokens_per_s", "mean"),
        load_time_s_mean=("load_time_s", "mean"),
    ).reset_index()
    bench_summary.to_csv(PROCESSED_DIR / "benchmark_summary.csv", index=False)
    print("Benchmark summary:\n", bench_summary, "\n")

    joined = overall.merge(bench_summary, on="system")
    joined.to_csv(PROCESSED_DIR / "quality_performance_joined.csv", index=False)

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(joined.system, joined.mean_score,
           yerr=[joined.mean_score - joined.ci95_lo, joined.ci95_hi - joined.mean_score],
           capsize=4, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("mean quality score (95% CI)")
    ax.set_title("Quality: CPU-SLM vs. Larger-Local")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    ax.bar(joined.system, joined.decode_tokens_per_s_mean, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("decode speed (tok/s)")
    ax.set_title("Speed: CPU-SLM vs. Larger-Local")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[2]
    ax.bar(joined.system, joined.peak_rss_mb_mean, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Memory: CPU-SLM vs. Larger-Local")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_speed_memory.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'quality_speed_memory.png'}")

    # --- Per-category comparison figure ---
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_pivot = per_category.pivot(index="category", columns="system", values="score")
    cat_pivot = cat_pivot[[c for c in SYSTEM_ORDER if c in cat_pivot.columns]]
    cat_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean score")
    ax.set_title("Quality by category: CPU-SLM vs. Larger-Local")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_by_category.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'quality_by_category.png'}")


if __name__ == "__main__":
    main()
