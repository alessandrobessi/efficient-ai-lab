"""quantscope CLI: bench, estimate-size, quantize, report, recommend,
formats, cpu-info.

argparse (stdlib) rather than a third-party CLI framework — this program's
own scope-control principle (root README.md §17: "nothing added ... unless
it lets us answer a research question we couldn't otherwise answer") applies
here too: quantscope's subcommand count and flag complexity don't need one.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from quantscope import __version__, cpu_detect, estimate as estimate_mod
from quantscope.bench import sweep
from quantscope.formats import applicable_formats, list_supported_formats
from quantscope.llama_bin import LlamaBinError, get_system_info
from quantscope.manifest import write_manifest
from quantscope.quantize import ImatrixRequiredError, missing_formats, produce_missing
from quantscope.report import DEFAULT_EPSILON, DEFAULT_PPL_ABSOLUTE_TOLERANCE, plot_frontier, rank_table, recommend as recommend_fn


def _parse_gguf_args(pairs: list[str]) -> dict[str, str]:
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--gguf expects FORMAT=path, got: {pair!r}")
        fmt, _, path = pair.partition("=")
        result[fmt.upper()] = path
    return result


def _manifest_path(output_path: str) -> str:
    import os

    stem, _ = os.path.splitext(output_path)
    return f"{stem}_manifest.json"


def cmd_bench(args: argparse.Namespace) -> int:
    if bool(args.llama_perplexity_bin) != bool(args.perplexity_dataset):
        raise SystemExit("quality evaluation needs both --llama-perplexity-bin and --perplexity-dataset (or neither)")
    if args.perplexity_baseline_format and not args.llama_perplexity_bin:
        raise SystemExit("--perplexity-baseline-format needs --llama-perplexity-bin and --perplexity-dataset too")

    gguf_paths = _parse_gguf_args(args.gguf)
    # Hashing every GGUF is wasted work if there's no manifest to put the
    # hashes in -- only do it when --output (and therefore a
    # <output>_manifest.json) was actually requested.
    compute_hashes = bool(args.output) and not args.skip_hash
    try:
        results, manifest = sweep(
            args.llama_bench_bin,
            gguf_paths,
            n_prompt=args.n_prompt,
            n_gen=args.n_gen,
            threads=args.threads,
            rounds=args.rounds,
            rng_seed=args.seed,
            llama_perplexity_bin=args.llama_perplexity_bin,
            perplexity_dataset=args.perplexity_dataset,
            perplexity_baseline_format=args.perplexity_baseline_format,
            compute_hashes=compute_hashes,
        )
    except ValueError as e:
        raise SystemExit(f"quantscope: {e}") from e

    rows = [dataclasses.asdict(r) for r in results]
    minimize = ["file_size_mb"]
    absolute_tolerance = None
    if args.llama_perplexity_bin:
        if args.perplexity_baseline_format:
            minimize.append("ppl_delta")
            absolute_tolerance = {"ppl_delta": args.ppl_absolute_tolerance}
        else:
            minimize.append("perplexity")
    for row in rows:
        # Raw per-round samples and perplexity_error are richer than a flat
        # CSV row should hold -- they're preserved in the manifest's
        # per-model entries below instead, not silently dropped.
        row.pop("prompt_tokens_per_second_samples", None)
        row.pop("gen_tokens_per_second_samples", None)
        if not args.llama_perplexity_bin:
            row.pop("perplexity", None)
            row.pop("perplexity_error", None)
            row.pop("ppl_delta", None)
            row.pop("ppl_ratio", None)
    df = rank_table(rows, minimize=minimize, maximize=["gen_tokens_per_second"], epsilon=args.pareto_epsilon, absolute_tolerance=absolute_tolerance)
    print(df.to_string(index=False))
    manifest.pareto_minimize = minimize
    manifest.pareto_maximize = ["gen_tokens_per_second"]
    manifest.pareto_epsilon = args.pareto_epsilon
    manifest.pareto_ppl_absolute_tolerance = args.ppl_absolute_tolerance if "ppl_delta" in minimize else None
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
        manifest_path = _manifest_path(args.output)
        write_manifest(manifest, manifest_path)
        print(f"Wrote {manifest_path}")
    if args.plot:
        quality_col = "ppl_delta" if "ppl_delta" in minimize else ("perplexity" if "perplexity" in minimize else None)
        plot_frontier(df, x="file_size_mb", y="gen_tokens_per_second", output_path=args.plot, quality_col=quality_col)
        print(f"Wrote {args.plot}")
    return 0


def cmd_estimate_size(args: argparse.Namespace) -> int:
    estimates = estimate_mod.estimate_size(args.formats)
    print(f"{'rank':>4}  {'format':<12}  {'approx bpw':>10}")
    for e in estimates:
        bpw = f"{e.approx_bits_per_weight:.1f}" if e.approx_bits_per_weight is not None else "unknown"
        print(f"{e.size_rank:>4}  {e.format:<12}  {bpw:>10}")
    print(f"\nNOTE: {estimate_mod.NOTE}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    existing = _parse_gguf_args(args.existing) if args.existing else {}
    to_produce = missing_formats(existing, args.formats) if args.skip_existing else args.formats
    if not to_produce:
        print("Nothing to do: all requested formats already exist (see --existing).")
        return 0
    try:
        produced = produce_missing(
            args.llama_quantize_bin,
            args.input,
            args.output_dir,
            to_produce,
            imatrix_path=args.imatrix,
            allow_iq_without_imatrix=args.allow_iq_without_imatrix,
        )
    except ImatrixRequiredError as e:
        raise SystemExit(f"quantscope: {e}") from e
    for fmt, path in produced.items():
        print(f"{fmt}: {path}")
    return 0


def _resolve_objectives(minimize: list[str], maximize: list[str]) -> tuple[list[str], list[str]]:
    """Defaults each of minimize/maximize independently to the most common
    column for that role (file_size_mb / gen_tokens_per_second) when not
    given -- matches recommend's original per-field defaulting. rank_table
    itself now also raises on a genuinely empty pair rather than crashing
    inside pandas/sort_values, but a clean, usable default beats forcing
    every `report` invocation to spell out both columns explicitly.
    """
    return list(minimize) or ["file_size_mb"], list(maximize) or ["gen_tokens_per_second"]


def _absolute_tolerance_for(minimize: list[str], ppl_absolute_tolerance: float) -> dict[str, float] | None:
    return {"ppl_delta": ppl_absolute_tolerance} if "ppl_delta" in minimize else None


def cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    df_in = pd.read_csv(args.csv)
    rows = df_in.to_dict(orient="records")
    minimize, maximize = _resolve_objectives(args.minimize, args.maximize)
    absolute_tolerance = _absolute_tolerance_for(minimize, args.ppl_absolute_tolerance)
    df = rank_table(rows, minimize=minimize, maximize=maximize, epsilon=args.pareto_epsilon, absolute_tolerance=absolute_tolerance)
    print(df.to_string(index=False))
    if args.plot:
        x = minimize[0] if minimize else maximize[0]
        y = maximize[0] if maximize else minimize[-1]
        quality_col = "ppl_delta" if "ppl_delta" in df.columns else ("perplexity" if "perplexity" in df.columns else None)
        plot_frontier(df, x=x, y=y, output_path=args.plot, quality_col=quality_col)
        print(f"\nWrote {args.plot}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    import pandas as pd

    df_in = pd.read_csv(args.csv)
    rows = df_in.to_dict(orient="records")
    minimize, maximize = _resolve_objectives(args.minimize, args.maximize)
    absolute_tolerance = _absolute_tolerance_for(minimize, args.ppl_absolute_tolerance)
    df = rank_table(rows, minimize=minimize, maximize=maximize, epsilon=args.pareto_epsilon, absolute_tolerance=absolute_tolerance)
    try:
        result = recommend_fn(
            df,
            max_size_mb=args.max_size_gb * 1024 if args.max_size_gb is not None else None,
            min_gen_tokens_per_second=args.min_tokens_per_second,
            max_ppl_delta=args.max_ppl_delta,
        )
    except ValueError as e:
        raise SystemExit(f"quantscope: {e}") from e
    if result.empty:
        print("No format meets all the given constraints.")
        return 1
    print(result.to_string(index=False))
    return 0


def cmd_formats(args: argparse.Namespace) -> int:
    all_formats = list_supported_formats(args.llama_quantize_bin)
    usable = applicable_formats(all_formats, has_imatrix=bool(args.imatrix))
    print("supported:", ", ".join(all_formats))
    print("applicable" + (" (with imatrix)" if args.imatrix else " (no imatrix)") + ":", ", ".join(usable))
    excluded = sorted(set(all_formats) - set(usable))
    if excluded:
        print(f"excluded (need an imatrix): {', '.join(excluded)}")
    return 0


def cmd_cpu_info(args: argparse.Namespace) -> int:
    llama_features = cpu_detect.parse_llama_feature_line(get_system_info(args.llama_bench_bin))
    os_features = cpu_detect.get_os_reported_features()
    print("reported by this llama.cpp build:")
    for name, used in sorted(llama_features.items()):
        print(f"  {name}: {'yes' if used else 'no'}")
    diverging = cpu_detect.unused_but_supported(llama_features, os_features)
    if diverging:
        print("\nCPU supports these but this build does not report using them:")
        for name in diverging:
            print(f"  {name}")
    else:
        print("\nno divergence detected between OS-reported and build-reported features")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantscope", description=__doc__)
    parser.add_argument("--version", action="version", version=f"quantscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bench", help="sweep llama-bench across pre-quantized GGUF files (CPU only: -ngl 0 always)")
    p.add_argument("--llama-bench-bin", required=True)
    p.add_argument("--gguf", action="append", required=True, metavar="FORMAT=path", help="repeatable")
    p.add_argument("--n-prompt", type=int, default=512)
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--rounds", type=int, default=10, help="independent, independently-randomized-order repetitions per format (llama-bench -r 1 each) -- not one sequential -r N pass per format, see ROADMAP.md's v0.2.1 section for why")
    p.add_argument("--seed", type=int, help="seed the round-order randomization for a reproducible sweep (default: unseeded/random)")
    p.add_argument("--output", help="write ranked results to this CSV path, plus a <path>_manifest.json")
    p.add_argument("--skip-hash", action="store_true", help="skip sha256 hashing GGUF files for the manifest (faster on huge files)")
    p.add_argument("--plot", help="write a Pareto-frontier plot (file size vs. gen tokens/sec) to this path")
    p.add_argument("--pareto-epsilon", type=float, default=DEFAULT_EPSILON, help=f"relative tolerance before a difference counts as material, not noise (default {DEFAULT_EPSILON})")
    p.add_argument("--ppl-absolute-tolerance", type=float, default=DEFAULT_PPL_ABSOLUTE_TOLERANCE, help=f"absolute (not relative) tolerance for ppl_delta -- a relative tolerance is meaningless around a zero baseline (default {DEFAULT_PPL_ABSOLUTE_TOLERANCE})")
    p.add_argument("--llama-perplexity-bin", help="also measure quality via llama-perplexity (needs --perplexity-dataset too)")
    p.add_argument("--perplexity-dataset", help="text file passed to llama-perplexity -f (needs --llama-perplexity-bin too)")
    p.add_argument("--perplexity-baseline-format", help="one of --gguf's formats to treat as the quality reference (e.g. F16); adds ppl_delta/ppl_ratio columns")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("estimate-size", help="rank formats by approximate storage size only -- NOT a speed estimate, see NOTE")
    p.add_argument("formats", nargs="+")
    p.set_defaults(func=cmd_estimate_size)

    p = sub.add_parser("quantize", help="produce missing GGUF formats via llama-quantize")
    p.add_argument("--llama-quantize-bin", required=True)
    p.add_argument("--input", required=True, help="source GGUF (e.g. an F16 file)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("formats", nargs="+")
    p.add_argument("--existing", action="append", default=[], metavar="FORMAT=path", help="repeatable; skip these if --skip-existing")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--imatrix", metavar="PATH", help="calibration file passed to llama-quantize --imatrix; required for IQ* formats")
    p.add_argument("--allow-iq-without-imatrix", action="store_true", help="produce IQ* formats even without --imatrix, accepting the quality risk")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("report", help="rank a bench CSV on a Pareto frontier")
    p.add_argument("--csv", required=True)
    p.add_argument("--minimize", action="append", default=[], metavar="COLUMN", help="repeatable (default: file_size_mb)")
    p.add_argument("--maximize", action="append", default=[], metavar="COLUMN", help="repeatable (default: gen_tokens_per_second)")
    p.add_argument("--pareto-epsilon", type=float, default=DEFAULT_EPSILON, help=f"relative tolerance before a difference counts as material, not noise (default {DEFAULT_EPSILON})")
    p.add_argument("--ppl-absolute-tolerance", type=float, default=DEFAULT_PPL_ABSOLUTE_TOLERANCE, help=f"absolute (not relative) tolerance for ppl_delta (default {DEFAULT_PPL_ABSOLUTE_TOLERANCE})")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("recommend", help="filter a bench CSV to formats meeting explicit constraints")
    p.add_argument("--csv", required=True)
    p.add_argument("--minimize", action="append", default=[], metavar="COLUMN", help="repeatable (default: file_size_mb)")
    p.add_argument("--maximize", action="append", default=[], metavar="COLUMN", help="repeatable (default: gen_tokens_per_second)")
    p.add_argument("--pareto-epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--ppl-absolute-tolerance", type=float, default=DEFAULT_PPL_ABSOLUTE_TOLERANCE, help=f"absolute (not relative) tolerance for ppl_delta (default {DEFAULT_PPL_ABSOLUTE_TOLERANCE})")
    p.add_argument("--max-size-gb", type=float, help="e.g. 5 for formats no larger than 5GB")
    p.add_argument("--min-tokens-per-second", type=float, help="e.g. 30 for formats at least that fast")
    p.add_argument("--max-ppl-delta", type=float, help="e.g. 0.10 for at most +0.10 perplexity vs. the baseline (requires bench --perplexity-baseline-format)")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("formats", help="list formats this llama-quantize binary supports")
    p.add_argument("--llama-quantize-bin", required=True)
    p.add_argument("--imatrix", metavar="PATH", help="calibration file you have available; if given, IQ* formats are listed as applicable")
    p.set_defaults(func=cmd_formats)

    p = sub.add_parser("cpu-info", help="cross-check CPU features this build uses vs. what the OS reports")
    p.add_argument("--llama-bench-bin", required=True)
    p.set_defaults(func=cmd_cpu_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LlamaBinError as e:
        print(f"quantscope: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
