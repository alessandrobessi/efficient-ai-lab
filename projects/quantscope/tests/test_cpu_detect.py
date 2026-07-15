from quantscope.cpu_detect import parse_llama_feature_line, unused_but_supported

SYSTEM_INFO_LINE = (
    "system_info: n_threads = 4 (n_threads_batch = 4) / 8 | CPU : SSE3 = 1 | "
    "SSSE3 = 1 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | "
    "F16C = 1 | LLAMAFILE = 1 | ACCELERATE = 1 |"
)


def test_parse_llama_feature_line_ignores_non_boolean_fields():
    features = parse_llama_feature_line(SYSTEM_INFO_LINE)
    assert "n_threads" not in features
    assert "N_THREADS" not in features


def test_parse_llama_feature_line_extracts_flags():
    features = parse_llama_feature_line(SYSTEM_INFO_LINE)
    assert features["AVX2"] is True
    assert features["AVX512"] is False
    assert features["FMA"] is True


def test_unused_but_supported_flags_divergence():
    llama_features = {"AVX2": False, "FMA": True}
    os_features = {"AVX2": True, "FMA": True, "AVX512": True}
    diverging = unused_but_supported(llama_features, os_features)
    # AVX2: OS says yes, build says no -> divergence (relevant feature).
    # FMA: both agree yes -> no divergence.
    # AVX512: OS says yes, build doesn't mention it at all -> divergence
    # (still a relevant, SIMD-related feature, just missing from the
    # build's own reported set entirely).
    assert diverging == ["AVX2", "AVX512"]


def test_unused_but_supported_ignores_irrelevant_os_flags():
    # /proc/cpuinfo-style flags that have nothing to do with GGML's compute
    # kernels must never be reported as a divergence, even though llama.cpp
    # never mentions them either -- llama.cpp not reporting on APIC/MSR/PAE
    # doesn't mean anything, since it was never going to report on them.
    llama_features = {"AVX2": True}
    os_features = {"AVX2": True, "APIC": True, "MSR": True, "PAE": True, "MTRR": True}
    assert unused_but_supported(llama_features, os_features) == []


def test_unused_but_supported_no_divergence_when_build_uses_everything_os_reports():
    llama_features = {"AVX2": True, "FMA": True}
    os_features = {"AVX2": True, "FMA": True}
    assert unused_but_supported(llama_features, os_features) == []


def test_unused_but_supported_ignores_os_features_not_actually_supported():
    llama_features = {}
    os_features = {"AVX512": False}
    assert unused_but_supported(llama_features, os_features) == []
