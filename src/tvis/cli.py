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

    open_ = sub.add_parser(
        "open",
        help="open the viewer for runs on this machine or a remote one",
        description="Serve the viewer on 127.0.0.1 and open it in your browser. TARGET is a local runs "
        "directory, HOST:PATH (HOST may be any ~/.ssh/config alias, ProxyJump hops included), or a "
        "saved target name.",
    )
    open_.add_argument("target")
    open_.add_argument(
        "--via", action="append", default=[], metavar="HOP", help="jump host (repeatable, in order)"
    )
    open_.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    open_.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    open_.add_argument(
        "--remote-tvis",
        default=None,
        help="how to invoke tvis on the remote host (default: tvis), e.g. ~/venv/bin/tvis",
    )

    agent = sub.add_parser("agent", help="(internal) answer viewer requests over stdin/stdout")
    agent.add_argument("--stdio", action="store_true", required=True)
    agent.add_argument("--root", type=Path, required=True)

    target = sub.add_parser("target", help="manage saved remote targets")
    target_sub = target.add_subparsers(dest="target_command", required=True)
    add = target_sub.add_parser("add", help="save a target")
    add.add_argument("name")
    add.add_argument("--host", required=True)
    add.add_argument("--path", required=True, help="runs directory on the host")
    add.add_argument("--via", default="", help="comma-separated jump hosts")
    add.add_argument("--remote-tvis", default=None)
    target_sub.add_parser("list", help="list saved targets")
    rm = target_sub.add_parser("remove", help="delete a saved target")
    rm.add_argument("name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "open":
        return _cmd_open(args)
    if args.command == "agent":
        from tvis.viewer.transport import serve_stdio

        serve_stdio(args.root.expanduser())
        return 0
    if args.command == "target":
        return _cmd_target(args)
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


def _cmd_open(args: argparse.Namespace) -> int:
    import webbrowser

    from tvis.viewer import targets
    from tvis.viewer.api import ApiError
    from tvis.viewer.server import ViewerServer
    from tvis.viewer.transport import LocalBackend, StdioBackend

    try:
        target = targets.resolve(args.target, via=args.via, remote_tvis=args.remote_tvis)
    except targets.TargetError as exc:
        print(f"tvis: {exc}", file=sys.stderr)
        return 2
    if target.is_local:
        backend = LocalBackend(Path(target.path))
    else:
        print(f"tvis: connecting to {target.describe()} …", file=sys.stderr)
        backend = StdioBackend(targets.ssh_command(target), description=target.describe())
        try:
            backend.request("ping", {})
        except ApiError as exc:
            backend.close()
            print(
                f"tvis: could not start the agent on {target.host}: {exc}\n"
                f"      is tvis installed there? try --remote-tvis /path/to/venv/bin/tvis",
                file=sys.stderr,
            )
            return 1
    try:
        server = ViewerServer(backend, port=args.port)
    except OSError as exc:
        backend.close()
        print(f"tvis: cannot listen on port {args.port}: {exc}", file=sys.stderr)
        return 1
    print(f"tvis: viewing {target.describe()}", file=sys.stderr)
    print(f"tvis: open {server.url}  (Ctrl-C to stop)", file=sys.stderr)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.close()
    return 0


def _cmd_target(args: argparse.Namespace) -> int:
    from tvis.viewer import targets

    if args.target_command == "add":
        via = [h.strip() for h in args.via.split(",") if h.strip()]
        target = targets.Target(
            host=args.host,
            path=args.path,
            via=via,
            remote_tvis=args.remote_tvis or targets.DEFAULT_REMOTE_TVIS,
        )
        targets.save_target(args.name, target)
        print(f"saved {args.name}: {target.describe()}")
        return 0
    if args.target_command == "list":
        for name, target in targets.load_targets().items():
            print(f"{name}\t{target.describe()}")
        return 0
    if args.target_command == "remove":
        if not targets.remove_target(args.name):
            print(f"tvis: no saved target named {args.name!r}", file=sys.stderr)
            return 1
        return 0
    raise AssertionError(args.target_command)


if __name__ == "__main__":
    sys.exit(main())
