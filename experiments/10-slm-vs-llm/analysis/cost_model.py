"""Week 10 cost model: turns measured per-request timing/token counts (for the
2 local systems) and a set of price parameters (for all 3 systems, including
the frontier API) into a cost-per-request comparison.

IMPORTANT — read before trusting any number this produces: the price
constants in config/model.yaml's `cost_model` section (local compute $/hour,
frontier $/million input and output tokens) are illustrative placeholders,
not verified current prices for any specific cloud provider or API. This
script's real contribution is the *formula* and the *measured* inputs (actual
generation time, actual token counts from this week's own dataset) — replace
the price constants with real, current figures for your own infrastructure
and provider before treating the dollar amounts as decisive. See README.md's
limitations section.

Local cost model: cost = (mean generation time in hours) x ($/hour). This
treats the whole machine as dedicated to one request at a time (matching
Week 8's finding that this setup serializes requests anyway) — it does not
amortize idle time, hardware purchase cost, or electricity separately.

Frontier cost model: cost = (mean prompt tokens / 1e6) x ($/M input tokens)
                           + (mean completion tokens / 1e6) x ($/M output tokens)
using this week's own dataset's actual measured token counts (from the
CPU-SLM run, as a stand-in for what any system would see, since the prompts
are identical) — not fabricated, but also not actually run against a
frontier API this week (see hypothesis.md).
"""

import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "10-slm-vs-llm"

with open(EXPERIMENT_DIR / "config" / "model.yaml") as f:
    CONFIG = yaml.safe_load(f)


def main() -> None:
    scored_path = PROCESSED_DIR / "scored_outputs.csv"
    if not scored_path.exists():
        print(f"skip: {scored_path} not found")
        return

    df = pd.read_csv(scored_path)
    cost_cfg = CONFIG["cost_model"]

    rows = []
    for system, grp in df.groupby("system"):
        mean_total_time_s = grp.total_time_s.mean()
        cost_per_request = (mean_total_time_s / 3600) * cost_cfg["local_compute_usd_per_hour"]
        rows.append({
            "system": system,
            "mean_total_time_s": mean_total_time_s,
            "mean_prompt_tokens": grp.prompt_tokens.mean(),
            "mean_completion_tokens": grp.completion_tokens.mean(),
            "cost_per_request_usd": cost_per_request,
            "cost_per_1000_requests_usd": cost_per_request * 1000,
        })

    # Frontier row: token counts borrowed from the CPU-SLM run (same prompts,
    # same dataset), cost computed from the (placeholder) per-token prices —
    # no generation was actually run against a frontier API this week.
    slm_row = df[df.system == CONFIG["local_systems"][0]["name"]]
    mean_prompt_tokens = slm_row.prompt_tokens.mean()
    mean_completion_tokens = slm_row.completion_tokens.mean()
    frontier_cost = (
        (mean_prompt_tokens / 1e6) * cost_cfg["frontier_input_usd_per_million_tokens"]
        + (mean_completion_tokens / 1e6) * cost_cfg["frontier_output_usd_per_million_tokens"]
    )
    rows.append({
        "system": "Frontier-API (documented, not measured)",
        "mean_total_time_s": None,
        "mean_prompt_tokens": mean_prompt_tokens,
        "mean_completion_tokens": mean_completion_tokens,
        "cost_per_request_usd": frontier_cost,
        "cost_per_1000_requests_usd": frontier_cost * 1000,
    })

    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(PROCESSED_DIR / "cost_comparison.csv", index=False)
    print("Cost comparison (illustrative price constants — see module docstring):")
    print(cost_df.to_string(index=False))

    with (PROCESSED_DIR / "cost_model_params.json").open("w") as f:
        json.dump(cost_cfg, f, indent=2)


if __name__ == "__main__":
    main()
