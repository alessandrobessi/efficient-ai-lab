"""Two-layer CPU feature detection.

Layer 1 (ground truth for *this build*): parse the `system_info:` line
llama.cpp binaries print at startup — e.g.
`system_info: ... | CPU : SSE3 = 1 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | ...` —
which reports what this specific compiled binary actually detected and will
use, not what the CPU is theoretically capable of.

Layer 2 (independent check): ask the OS directly (sysctl on macOS,
/proc/cpuinfo on Linux) what the CPU itself claims to support.

Comparing the two surfaces a diagnostic this program never had during Weeks
4/6: "this CPU supports AVX2, but this llama.cpp build isn't using it" —
which a benchmark alone can't distinguish from "this CPU doesn't have AVX2."
Feature-name mapping between llama.cpp's naming and the OS's naming is
necessarily approximate (see _ALIASES) — this is a v1 heuristic, not a
guarantee, and is documented as such.
"""

from __future__ import annotations

import platform
import re
import subprocess

_FEATURE_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s*=\s*([01])\b")

# Normalizes a few known OS-reported spellings to the name llama.cpp's own
# system_info line uses, for the common features this program's own
# quantization work actually cared about (Weeks 4/6's REPACK discussion).
# Not exhaustive by design — a v1 heuristic, expand as real divergences turn
# up in practice rather than guessing every possible CPU flag ahead of time.
_ALIASES = {
    "AVX2_0": "AVX2",
    "AVX512F": "AVX512",
    "FMA3": "FMA",
    "NEON_FP16": "FP16_VA",
}


def parse_llama_feature_line(text: str) -> dict[str, bool]:
    """Parses `NAME = 0`/`NAME = 1` tokens out of llama.cpp's system_info
    log line. Only matches all-caps identifiers (SSE3, AVX2, FMA, ...), so
    non-boolean fields on the same line (`n_threads = 4`) are never
    mistaken for feature flags.
    """
    return {name: value == "1" for name, value in _FEATURE_RE.findall(text)}


def get_os_reported_features() -> dict[str, bool]:
    """Independently asks the OS what CPU features are present, regardless
    of what any particular llama.cpp build reports using.
    """
    system = platform.system()
    if system == "Darwin":
        return _macos_features()
    if system == "Linux":
        return _linux_features()
    return {}


def _macos_features() -> dict[str, bool]:
    features: dict[str, bool] = {}
    try:
        out = subprocess.run(["sysctl", "-a"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return features
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            for flag in value.split():
                features[flag.upper()] = True
        elif key.startswith("hw.optional.") and value in ("0", "1"):
            features[key.removeprefix("hw.optional.").upper()] = value == "1"
    return features


def _linux_features() -> dict[str, bool]:
    features: dict[str, bool] = {}
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return features
    for line in text.splitlines():
        lowered = line.lower()
        if lowered.startswith("flags") or lowered.startswith("features"):
            _, _, value = line.partition(":")
            for flag in value.split():
                features[flag.upper()] = True
            break
    return features


def _canonicalize(name: str) -> str:
    return _ALIASES.get(name, name)


def unused_but_supported(llama_features: dict[str, bool], os_features: dict[str, bool]) -> list[str]:
    """Returns feature names the OS reports as present that this llama.cpp
    build's system_info line does not report as in use (either reported
    False, or not mentioned at all).
    """
    llama_canonical = {_canonicalize(k): v for k, v in llama_features.items()}
    diverging = []
    for name, os_supported in os_features.items():
        if not os_supported:
            continue
        canonical = _canonicalize(name)
        if llama_canonical.get(canonical) is not True:
            diverging.append(name)
    return sorted(diverging)
