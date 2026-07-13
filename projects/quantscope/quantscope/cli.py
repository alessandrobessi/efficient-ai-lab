"""quantscope CLI: bench, predict, quantize, report, formats, cpu-info.

argparse (stdlib) rather than a third-party CLI framework — this program's
own scope-control principle (root README.md §17: "nothing added ... unless
it lets us answer a research question we couldn't otherwise answer") applies
here too: quantscope's subcommand count and flag complexity don't need one.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from quantscope import __version__, cpu_detect, predict as predict_mod
from quantscope.bench import sweep
from quantscope.formats import applicable_formats, list_supported_formats
from quantscope.llama_bin import LlamaBinError, get_system_info
from quantscope.quantize import missing_formats, produce_missing
from quantscope.report import plot_frontier, rank_table


def _parse_gguf_args(pairs: list[str]) -> dict[str, str]:
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--gguf expects FORMAT=path, got: {pair!r}")
        fmt, _, path = pair.partition("=")
        result[fmt.upper()] = path
    return result


def cmd_bench(args: argparse.Namespace) -> int:
    if bool(args.llama_perplexity_bin) != bool(args.perplexity_dataset):
        raise SystemExit("--quality-eval needs both --llama-perplexity-bin and --perplexity-dataset")

    gguf_paths = _parse_gguf_args(args.gguf)
    results = sweep(
        args.llama_bench_bin,
        gguf_paths,
        n_prompt=args.n_prompt,
        n_gen=args.n_gen,
        threads=args.threads,
        repetitions=args.repetitions,
        llama_perplexity_bin=args.llama_perplexity_bin,
        perplexity_dataset=args.perplexity_dataset,
    )
    rows = [dataclasses.asdict(r) for r in results]
    minimize = ["file_size_mb"]
    if args.llama_perplexity_bin:
        minimize.append("perplexity")
    else:
        for row in rows:
            row.pop("perplexity", None)
    df = rank_table(rows, minimize=minimize, maximize=["gen_tokens_per_second"])
    print(df.to_string(index=False))
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
    if args.plot:
        plot_frontier(df, x="file_size_mb", y="gen_tokens_per_second", output_path=args.plot)
        print(f"Wrote {args.plot}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    predictions = predict_mod.predict(args.formats)
    print(f"{'rank':>4}  {'format':<12}  {'approx bpw':>10}")
    for p in predictions:
        bpw = f"{p.approx_bits_per_weight:.1f}" if p.approx_bits_per_weight is not None else "unknown"
        print(f"{p.predicted_rank:>4}  {p.format:<12}  {bpw:>10}")
    print(f"\nNOTE: {predict_mod.CONFIDENCE_CAVEAT}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    existing = _parse_gguf_args(args.existing) if args.existing else {}
    to_produce = missing_formats(existing, args.formats) if args.skip_existing else args.formats
    if not to_produce:
        print("Nothing to do: all requested formats already exist (see --existing).")
        return 0
    produced = produce_missing(args.llama_quantize_bin, args.input, args.output_dir, to_produce)
    for fmt, path in produced.items():
        print(f"{fmt}: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    df_in = pd.read_csv(args.csv)
    rows = df_in.to_dict(orient="records")
    df = rank_table(rows, minimize=args.minimize, maximize=args.maximize)
    print(df.to_string(index=False))
    if args.plot:
        x = args.minimize[0] if args.minimize else args.maximize[0]
        y = args.maximize[0] if args.maximize else args.minimize[-1]
        plot_frontier(df, x=x, y=y, output_path=args.plot)
        print(f"\nWrote {args.plot}")
    return 0


def cmd_formats(args: argparse.Namespace) -> int:
    all_formats = list_supported_formats(args.llama_quantize_bin)
    usable = applicable_formats(all_formats, has_imatrix=args.imatrix)
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

    p = sub.add_parser("bench", help="sweep llama-bench across pre-quantized GGUF files")
    p.add_argument("--llama-bench-bin", required=True)
    p.add_argument("--gguf", action="append", required=True, metavar="FORMAT=path", help="repeatable")
    p.add_argument("--n-prompt", type=int, default=512)
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output", help="write ranked results to this CSV path")
    p.add_argument("--plot", help="write a Pareto-frontier plot to this path")
    p.add_argument("--llama-perplexity-bin", help="also measure quality via llama-perplexity (needs --perplexity-dataset too)")
    p.add_argument("--perplexity-dataset", help="text file passed to llama-perplexity -f (needs --llama-perplexity-bin too)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("predict", help="heuristic format ranking, no benchmarking (see CONFIDENCE_CAVEAT)")
    p.add_argument("formats", nargs="+")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("quantize", help="produce missing GGUF formats via llama-quantize")
    p.add_argument("--llama-quantize-bin", required=True)
    p.add_argument("--input", required=True, help="source GGUF (e.g. an F16 file)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("formats", nargs="+")
    p.add_argument("--existing", action="append", default=[], metavar="FORMAT=path", help="repeatable; skip these if --skip-existing")
    p.add_argument("--skip-existing", action="store_true")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("report", help="rank a bench CSV on a Pareto frontier")
    p.add_argument("--csv", required=True)
    p.add_argument("--minimize", action="append", default=[], metavar="COLUMN", help="repeatable")
    p.add_argument("--maximize", action="append", default=[], metavar="COLUMN", help="repeatable")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("formats", help="list formats this llama-quantize binary supports")
    p.add_argument("--llama-quantize-bin", required=True)
    p.add_argument("--imatrix", action="store_true", help="include formats that need an imatrix")
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
