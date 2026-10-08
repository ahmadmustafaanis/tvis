"""Command-line entry point: ``tvis run | open | agent | target``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tvis", description="Step-through debugger for PyTorch training.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="run a training script and record its first N steps",
        description="Run SCRIPT in-process (same as `python SCRIPT ARGS`), record the first N optimizer "
        "steps end to end, then stop.",
    )
    run.add_argument("--steps", type=int, default=3, help="optimizer steps to capture (default: 3)")
    run.add_argument(
        "--out", type=Path, default=None, help="directory for runs (default: <project>/tvis_runs)"
    )
    run.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project root whose functions are traced (default: script dir)",
    )
    run.add_argument(
        "--max-elems",
        type=int,
        default=2_000_000,
        help="store full values for tensors up to this many elements; larger ones keep stats + leading rows",
    )
    run.add_argument("--max-batches", type=int, default=64, help="safety cap on captured batches")
    run.add_argument("--no-functions", action="store_true", help="don't trace project functions")
    run.add_argument("--no-ops", action="store_true", help="don't record line-level ops")
    run.add_argument("script", type=Path)
    run.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    raise AssertionError(args.command)


def _cmd_run(args: argparse.Namespace) -> int:
    from tvis import runner

    if args.steps < 1:
        print("tvis: --steps must be >= 1", file=sys.stderr)
        return 2
    options = runner.RunOptions(
        script=args.script,
        script_args=args.script_args,
        steps=args.steps,
        out_dir=args.out,
        project_dir=args.project,
        max_elems=args.max_elems,
        max_batches=args.max_batches,
        trace_functions=not args.no_functions,
        record_ops=not args.no_ops,
    )
    try:
        result = runner.run(options)
    except FileNotFoundError as exc:
        print(f"tvis: {exc}", file=sys.stderr)
        return 2
    print(runner.summary(result, args.steps), file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
