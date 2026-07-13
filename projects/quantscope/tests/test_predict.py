from quantscope.predict import predict


def test_predict_ranks_smaller_bpw_first():
    predictions = predict(["Q8_0", "Q4_0", "F16"])
    ranked_formats = [p.format for p in predictions]
    assert ranked_formats == ["Q4_0", "Q8_0", "F16"]


def test_predict_unknown_formats_sort_last_not_dropped():
    predictions = predict(["Q4_0", "SOME_FUTURE_FORMAT"])
    assert len(predictions) == 2
    assert predictions[-1].format == "SOME_FUTURE_FORMAT"
    assert predictions[-1].approx_bits_per_weight is None
