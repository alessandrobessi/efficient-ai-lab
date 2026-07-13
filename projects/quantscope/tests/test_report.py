from quantscope.report import pareto_optimal_flags, rank_table

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
