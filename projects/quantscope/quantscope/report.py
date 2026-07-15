"""Ranks quantization formats on a Pareto frontier (speed vs. memory vs.
quality) and renders a table/plot — reusing the same "dominated" /
"Pareto-optimal" framing this program used in Weeks 5, 6, and its
architecture decision framework, rather than inventing new vocabulary here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: quantscope never assumes a display is available
import matplotlib.pyplot as plt
import pandas as pd

# Two benchmark averages within this fraction of each other are treated as
# indistinguishable rather than as a real difference — llama-bench reports
# a stddev alongside every average precisely because run-to-run noise is
# real, and treating every tiny difference as exact would let Pareto
# membership flip on measurement noise rather than a material improvement.
DEFAULT_EPSILON = 0.02


def _dominates(b: dict, a: dict, minimize: list[str], maximize: list[str], epsilon: float = DEFAULT_EPSILON) -> bool:
    """True if point b dominates point a: at least as good as a in every
    objective (within `epsilon` relative tolerance), and *materially*
    better (beyond that tolerance) in at least one. epsilon=0 recovers
    exact dominance.
    """
    strictly_better = False
    for key in minimize:
        if b[key] > a[key] * (1 + epsilon):
            return False
        if b[key] < a[key] * (1 - epsilon):
            strictly_better = True
    for key in maximize:
        if b[key] < a[key] * (1 - epsilon):
            return False
        if b[key] > a[key] * (1 + epsilon):
            strictly_better = True
    return strictly_better


def pareto_optimal_flags(
    rows: list[dict], minimize: list[str], maximize: list[str], epsilon: float = DEFAULT_EPSILON
) -> list[bool]:
    """Returns, for each row, whether it is non-dominated (Pareto-optimal)
    with respect to the given objectives. minimize objectives (e.g. file
    size) are better when smaller; maximize objectives (e.g. tokens/sec)
    are better when larger.
    """
    flags = []
    for i, a in enumerate(rows):
        dominated = any(_dominates(b, a, minimize, maximize, epsilon) for j, b in enumerate(rows) if j != i)
        flags.append(not dominated)
    return flags


def rank_table(
    rows: list[dict], minimize: list[str], maximize: list[str], epsilon: float = DEFAULT_EPSILON
) -> pd.DataFrame:
    """Builds a ranked comparison table with a `pareto_optimal` column,
    Pareto-optimal rows sorted first.
    """
    df = pd.DataFrame(rows)
    df["pareto_optimal"] = pareto_optimal_flags(rows, minimize, maximize, epsilon)
    sort_key = maximize[0] if maximize else minimize[0]
    ascending = sort_key in minimize
    df = df.sort_values(
        by=["pareto_optimal", sort_key], ascending=[False, ascending]
    ).reset_index(drop=True)
    return df


def recommend(
    df: pd.DataFrame,
    max_size_mb: float | None = None,
    min_gen_tokens_per_second: float | None = None,
    max_ppl_delta: float | None = None,
) -> pd.DataFrame:
    """Filters an already-ranked table down to rows meeting every given
    constraint — turns "here are the Pareto-optimal points" (which can
    still mean "all of them, take your pick" when every faster format is
    also larger) into an actual, actionable subset. Any constraint left as
    None is not applied. Raises if max_ppl_delta is given but the table has
    no ppl_delta column (i.e. bench was not run with a perplexity baseline).
    """
    filtered = df
    if max_size_mb is not None:
        filtered = filtered[filtered["file_size_mb"] <= max_size_mb]
    if min_gen_tokens_per_second is not None:
        filtered = filtered[filtered["gen_tokens_per_second"] >= min_gen_tokens_per_second]
    if max_ppl_delta is not None:
        if "ppl_delta" not in df.columns:
            raise ValueError(
                "--max-ppl-delta requires a ppl_delta column -- run bench with "
                "--perplexity-baseline-format to get one"
            )
        filtered = filtered[filtered["ppl_delta"] <= max_ppl_delta]
    return filtered.reset_index(drop=True)


def plot_frontier(
    df: pd.DataFrame, x: str, y: str, output_path: str, label_col: str = "format", quality_col: str | None = None
) -> None:
    """Scatter-plots x vs. y, highlighting Pareto-optimal points and
    labeling each with its format name. If quality_col is given and present
    in df, each point's label also shows that value — Pareto membership can
    depend on more objectives than just x and y (e.g. a perplexity-aware
    sweep), and a 2D plot that doesn't surface the invisible third
    dimension can make a "Pareto-optimal" point look dominated, or vice
    versa, for no visible reason.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    dominated = df[~df["pareto_optimal"]]
    optimal = df[df["pareto_optimal"]]
    ax.scatter(dominated[x], dominated[y], c="tab:gray", label="dominated", alpha=0.7)
    ax.scatter(optimal[x], optimal[y], c="tab:blue", label="Pareto-optimal", alpha=0.9)
    # Points often cluster tightly (several quantization formats at similar
    # size/speed) -- alternating the label offset up/down and left/right by
    # index keeps a dense cluster's labels from stacking directly on top of
    # each other, at the cost of not being a real collision-avoidance layout.
    offsets = [(6, 8), (6, -14), (-60, 8), (-60, -14)]
    for i, (_, row) in enumerate(df.iterrows()):
        label = str(row[label_col])
        if quality_col and quality_col in df.columns and pd.notna(row.get(quality_col)):
            label += f"\n{quality_col}={row[quality_col]:.3g}"
        ax.annotate(label, (row[x], row[y]), fontsize=8, xytext=offsets[i % len(offsets)], textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
