from quantscope.estimate import estimate_size


def test_estimate_size_ranks_smaller_bpw_first():
    estimates = estimate_size(["Q8_0", "Q4_0", "F16"])
    ranked_formats = [e.format for e in estimates]
    assert ranked_formats == ["Q4_0", "Q8_0", "F16"]


def test_estimate_size_unknown_formats_sort_last_not_dropped():
    estimates = estimate_size(["Q4_0", "SOME_FUTURE_FORMAT"])
    assert len(estimates) == 2
    assert estimates[-1].format == "SOME_FUTURE_FORMAT"
    assert estimates[-1].approx_bits_per_weight is None
