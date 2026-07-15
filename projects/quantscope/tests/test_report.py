import pandas as pd
import pytest

from quantscope.report import pareto_optimal_flags, plot_frontier, rank_table, recommend

# size should be minimized, speed maximized.
ROWS = [
    {"format": "A", "size": 10, "speed": 100},
    {"format": "B", "size": 6, "speed": 70},
    {"format": "C", "size": 8, "speed": 80},
    {"format": "D", "size": 9, "speed": 75},  # dominated by C: smaller size, higher speed
]


def test_pareto_optimal_flags():
    flags = pareto_optimal_flags(ROWS, minimize=["size"], maximize=["speed"])
    by_format = dict(zip((r["format"] for r in ROWS), flags))
    assert by_format["A"] is True
    assert by_format["B"] is True
    assert by_format["C"] is True
    assert by_format["D"] is False


def test_rank_table_sorts_pareto_optimal_first():
    df = rank_table(ROWS, minimize=["size"], maximize=["speed"])
    assert df.iloc[-1]["format"] == "D"
    assert df.iloc[-1]["pareto_optimal"] == False  # noqa: E712
    assert set(df[df["pareto_optimal"]]["format"]) == {"A", "B", "C"}


def test_epsilon_dominance_ignores_noise_level_speed_difference():
    # Same size; Q5_K_M is nominally faster than Q4_K_M by ~0.7%, well
    # within measurement noise (llama-bench's own stddev on runs like this
    # is typically 1-3%). Neither should dominate on a difference this
    # small -- both remain valid, indistinguishable choices.
    rows = [
        {"format": "Q4_K_M", "size": 10, "speed": 42.8},
        {"format": "Q5_K_M", "size": 10, "speed": 43.1},
    ]
    flags = pareto_optimal_flags(rows, minimize=["size"], maximize=["speed"])
    assert flags == [True, True]


def test_epsilon_dominance_lets_a_real_size_edge_win_over_noise_level_speed():
    # Q5_K_M's slight speed edge (0.7%) is noise, but it's also 10% larger
    # -- a real, material difference. Q4_K_M is smaller for essentially the
    # same speed, so it should dominate: Q5_K_M offers no real benefit here.
    rows = [
        {"format": "Q4_K_M", "size": 10, "speed": 42.8},
        {"format": "Q5_K_M", "size": 11, "speed": 43.1},
    ]
    flags = pareto_optimal_flags(rows, minimize=["size"], maximize=["speed"])
    by_format = dict(zip((r["format"] for r in rows), flags))
    assert by_format["Q4_K_M"] is True
    assert by_format["Q5_K_M"] is False


def test_epsilon_dominance_still_catches_material_differences():
    # A clearly, materially faster format at a clearly larger size: both
    # remain Pareto-optimal (real tradeoff). But a format that is both
    # larger AND not materially faster should be dominated.
    rows = [
        {"format": "fast", "size": 10, "speed": 100},
        {"format": "same_speed_bigger", "size": 12, "speed": 100.5},  # not materially faster, bigger -> dominated
    ]
    flags = pareto_optimal_flags(rows, minimize=["size"], maximize=["speed"])
    by_format = dict(zip((r["format"] for r in rows), flags))
    assert by_format["fast"] is True
    assert by_format["same_speed_bigger"] is False


def test_recommend_filters_by_constraints():
    # recommend() filters on the real column names (file_size_mb,
    # gen_tokens_per_second), not ROWS' generic "size"/"speed".
    df2 = pd.DataFrame(
        [
            {"format": "A", "file_size_mb": 4000, "gen_tokens_per_second": 40},
            {"format": "B", "file_size_mb": 8000, "gen_tokens_per_second": 60},
            {"format": "C", "file_size_mb": 6000, "gen_tokens_per_second": 30},
        ]
    )
    result = recommend(df2, max_size_mb=6000)
    assert set(result["format"]) == {"A", "C"}
    result = recommend(df2, min_gen_tokens_per_second=40)
    assert set(result["format"]) == {"A", "B"}


def test_recommend_max_ppl_delta_requires_column():
    df = pd.DataFrame([{"format": "A", "file_size_mb": 100, "gen_tokens_per_second": 10}])
    with pytest.raises(ValueError, match="ppl_delta"):
        recommend(df, max_ppl_delta=0.1)


def test_plot_frontier_with_quality_col_does_not_error(tmp_path):
    df = rank_table(
        [
            {"format": "A", "size": 10, "speed": 100, "ppl_delta": 0.05},
            {"format": "B", "size": 6, "speed": 70, "ppl_delta": 0.20},
        ],
        minimize=["size"],
        maximize=["speed"],
    )
    out = tmp_path / "frontier.png"
    plot_frontier(df, x="size", y="speed", output_path=str(out), quality_col="ppl_delta")
    assert out.exists()
