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


def _dominates(b: dict, a: dict, minimize: list[str], maximize: list[str]) -> bool:
    """True if point b dominates point a: at least as good as a in every
    objective, and strictly better in at least one.
    """
    at_least_as_good = True
    strictly_better = False
    for key in minimize:
        if b[key] > a[key]:
            return False
        if b[key] < a[key]:
            strictly_better = True
    for key in maximize:
        if b[key] < a[key]:
            return False
        if b[key] > a[key]:
            strictly_better = True
    return at_least_as_good and strictly_better


def pareto_optimal_flags(rows: list[dict], minimize: list[str], maximize: list[str]) -> list[bool]:
    """Returns, for each row, whether it is non-dominated (Pareto-optimal)
    with respect to the given objectives. minimize objectives (e.g. file
    size) are better when smaller; maximize objectives (e.g. tokens/sec)
    are better when larger.
    """
    flags = []
    for i, a in enumerate(rows):
        dominated = any(_dominates(b, a, minimize, maximize) for j, b in enumerate(rows) if j != i)
        flags.append(not dominated)
    return flags


def rank_table(rows: list[dict], minimize: list[str], maximize: list[str]) -> pd.DataFrame:
    """Builds a ranked comparison table with a `pareto_optimal` column,
    Pareto-optimal rows sorted first.
    """
    df = pd.DataFrame(rows)
    df["pareto_optimal"] = pareto_optimal_flags(rows, minimize, maximize)
    sort_key = maximize[0] if maximize else minimize[0]
    ascending = sort_key in minimize
    df = df.sort_values(
        by=["pareto_optimal", sort_key], ascending=[False, ascending]
    ).reset_index(drop=True)
    return df


def plot_frontier(df: pd.DataFrame, x: str, y: str, output_path: str, label_col: str = "format") -> None:
    """Scatter-plots x vs. y, highlighting Pareto-optimal points and
    labeling each with its format name.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    dominated = df[~df["pareto_optimal"]]
    optimal = df[df["pareto_optimal"]]
    ax.scatter(dominated[x], dominated[y], c="tab:gray", label="dominated", alpha=0.7)
    ax.scatter(optimal[x], optimal[y], c="tab:blue", label="Pareto-optimal", alpha=0.9)
    for _, row in df.iterrows():
        ax.annotate(str(row[label_col]), (row[x], row[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
