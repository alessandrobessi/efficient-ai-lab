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
#
# This is a *relative* tolerance, which works for objectives like file size
# or throughput where "material" scales with the value being compared. It
# breaks down for an objective measured as a delta from a zero baseline
# (e.g. ppl_delta, which is exactly 0 for whatever format was chosen as the
# quality reference): a relative band around 0 is 0, so the "2% tolerance"
# advertised for that column silently evaporates for the one row that
# needs it most. See absolute_tolerance below for the fix.
DEFAULT_EPSILON = 0.02

# ppl_delta is naturally centered on a zero baseline (see DEFAULT_EPSILON's
# note), so it needs an absolute tolerance band, not a relative one -- a
# difference of a few hundredths of a perplexity point is well within what
# llama-perplexity's own reported +/- uncertainty typically shows for a
# ~150KB evaluation window at these thread counts. This is a default, not a
# law: pass a different value (or None to disable) via rank_table's
# absolute_tolerance if your own dataset/uncertainty differs.
DEFAULT_PPL_ABSOLUTE_TOLERANCE = 0.05


def _tolerance_band(key: str, a_value: float, epsilon: float | dict[str, float], absolute_tolerance: dict[str, float] | None) -> float:
    """Returns the +/- band around a_value within which a difference on
    `key` doesn't count as material. Uses an absolute band if one is
    configured for this key (see DEFAULT_PPL_ABSOLUTE_TOLERANCE's note),
    otherwise a relative band (epsilon can be one float applied to every
    key, or a dict of per-key overrides).
    """
    if absolute_tolerance and key in absolute_tolerance and absolute_tolerance[key] is not None:
        return absolute_tolerance[key]
    key_epsilon = epsilon.get(key, DEFAULT_EPSILON) if isinstance(epsilon, dict) else epsilon
    return abs(a_value) * key_epsilon


def _dominates(
    b: dict,
    a: dict,
    minimize: list[str],
    maximize: list[str],
    epsilon: float | dict[str, float] = DEFAULT_EPSILON,
    absolute_tolerance: dict[str, float] | None = None,
) -> bool:
    """True if point b dominates point a: at least as good as a in every
    objective (within a tolerance band around a's value), and *materially*
    better (beyond that band) in at least one. epsilon=0 and no absolute
    tolerances recovers exact dominance.
    """
    strictly_better = False
    for key in minimize:
        band = _tolerance_band(key, a[key], epsilon, absolute_tolerance)
        if b[key] > a[key] + band:
            return False
        if b[key] < a[key] - band:
            strictly_better = True
    for key in maximize:
        band = _tolerance_band(key, a[key], epsilon, absolute_tolerance)
        if b[key] < a[key] - band:
            return False
        if b[key] > a[key] + band:
            strictly_better = True
    return strictly_better


def pareto_optimal_flags(
    rows: list[dict],
    minimize: list[str],
    maximize: list[str],
    epsilon: float | dict[str, float] = DEFAULT_EPSILON,
    absolute_tolerance: dict[str, float] | None = None,
) -> list[bool]:
    """Returns, for each row, whether it is non-dominated (Pareto-optimal)
    with respect to the given objectives. minimize objectives (e.g. file
    size) are better when smaller; maximize objectives (e.g. tokens/sec)
    are better when larger.
    """
    flags = []
    for i, a in enumerate(rows):
        dominated = any(
            _dominates(b, a, minimize, maximize, epsilon, absolute_tolerance)
            for j, b in enumerate(rows)
            if j != i
        )
        flags.append(not dominated)
    return flags


def rank_table(
    rows: list[dict],
    minimize: list[str],
    maximize: list[str],
    epsilon: float | dict[str, float] = DEFAULT_EPSILON,
    absolute_tolerance: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Builds a ranked comparison table with a `pareto_optimal` column,
    Pareto-optimal rows sorted first.

    absolute_tolerance, if given, maps column name -> a fixed +/- band used
    instead of a relative one for that column (see
    DEFAULT_PPL_ABSOLUTE_TOLERANCE). Callers that include "ppl_delta" in
    minimize should normally pass {"ppl_delta": DEFAULT_PPL_ABSOLUTE_TOLERANCE}
    or their own dataset-appropriate value.
    """
    if not minimize and not maximize:
        raise ValueError("rank_table needs at least one minimize or maximize column")
    df = pd.DataFrame(rows)
    df["pareto_optimal"] = pareto_optimal_flags(rows, minimize, maximize, epsilon, absolute_tolerance)
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
