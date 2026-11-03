import argparse
from pathlib import Path
from .config import TrackerConfig, VARIANTS, SUPPORTED_VARIANTS
from .data import load_cache
from .evaluation import evaluate_run
from .experiment import run_suite
from .runlog import RunLog, run_directory
from .synthetic import create_synthetic
from .analysis import analyze_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="ERA-MASA-OA reproducible experiments with durable logs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, help="New run directory (must not already exist)")
        p.add_argument("--config", type=Path)
        p.add_argument("--variants", nargs="+", choices=SUPPORTED_VARIANTS, default=list(VARIANTS))
        p.add_argument("--diagnostics", choices=["none", "matched", "all"], default="matched")
        p.add_argument("--no-subsets", action="store_true")
        p.add_argument("--evaluate", action="store_true")
        p.add_argument("--trackeval-root", type=Path, default=Path("third_party/TrackEval"))
        if name == "run":
            p.add_argument("--cache", type=Path, nargs="+", required=True)
        else:
            p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--trackeval-root", type=Path, default=Path("third_party/TrackEval"))
    p.add_argument("--no-subsets", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("extract", help="Requires separate MASA/PyTorch/OpenMMLab environment")
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--benchmark", choices=["MOT15", "MOT16", "MOT17", "MOT20", "CTMC-v1"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--masa-root", type=Path, default=Path("third_party/MASA-OA"))
    p.add_argument("--masa-config", type=Path)
    p.add_argument("--cache-output", type=Path, required=True)
    p.add_argument("--gt-policy", choices=["valid", "paper-all"], default="valid")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    values = {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
              for k, v in vars(args).items()}
    if args.command == "evaluate":
        destination = run_directory(args.run / "evaluation_logs", "evaluate")
        with RunLog(destination, values) as log:
            evaluate_run(args.run, args.trackeval_root, log, not args.no_subsets)
        print(f"Evaluation logs: {destination.resolve()}")
        return
    if args.command == "analyze":
        destination = run_directory(args.run / "analysis_logs", "analyze")
        with RunLog(destination, values) as log:
            analyze_run(args.run, log)
        print(f"Analysis logs: {destination.resolve()}")
        return
    destination = args.output or run_directory(name=args.command)
    with RunLog(destination, values) as log:
        log.message(f"Run directory: {destination.resolve()}")
        if args.command == "extract":
            from .extraction import extract_sequence
            extract_sequence(args, log)
        else:
            config = TrackerConfig.load(args.config)
            log.save("tracker_config.json", config.to_dict())
            paths = create_synthetic(destination / "synthetic_cache", args.seed) if args.command == "demo" else args.cache
            if args.command == "demo":
                log.message("SYNTHETIC VALIDATION ONLY: these results do not measure benchmark performance.")
            sequences = [load_cache(path) for path in paths]
            run_suite(sequences, config, args.variants, log, args.diagnostics, not args.no_subsets)
            if args.evaluate:
                evaluate_run(destination, args.trackeval_root, log, not args.no_subsets)
            if args.diagnostics != "none":
                analyze_run(destination, log)
    print(f"Results and logs: {destination.resolve()}")


if __name__ == "__main__":
    main()
