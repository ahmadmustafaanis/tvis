"""``tvis run``: execute a training script in-process under capture, stop after N steps."""

from __future__ import annotations

import os
import runpy
import secrets
import socket
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tvis.store import schema

DEFAULT_OUT_DIR = "tvis_runs"


@dataclass
class RunOptions:
    script: Path
    script_args: list[str]
    steps: int = 3
    out_dir: Path | None = None
    project_dir: Path | None = None
    max_elems: int = 2_000_000
    max_batches: int = 64
    trace_functions: bool = True
    record_ops: bool = True


@dataclass
class RunResult:
    exit_code: int
    run_dir: Path
    status: str
    steps_captured: int
    batches_captured: int


def new_run_id(script: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{script.stem}_{secrets.token_hex(2)}"


def run(options: RunOptions) -> RunResult:
    from tvis.capture.session import CaptureConfig, Session, StopCapture

    script = options.script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"training script not found: {options.script}")
    project_dir = (options.project_dir or script.parent).resolve()
    out_dir = (options.out_dir or project_dir / DEFAULT_OUT_DIR).resolve()
    run_dir = out_dir / new_run_id(script)

    config = CaptureConfig(
        run_dir=run_dir,
        project_dir=project_dir,
        steps=options.steps,
        max_batches=options.max_batches,
        max_elems=options.max_elems,
        trace_functions=options.trace_functions,
        record_ops=options.record_ops,
        script=str(script),
        argv=list(options.script_args),
    )
    session = Session(config)

    saved_argv, saved_path = sys.argv[:], sys.path[:]
    sys.argv = [str(script), *options.script_args]
    sys.path.insert(0, str(script.parent))
    exit_code, status, error = 0, schema.STATUS_COMPLETE, None
    session.install()
    try:
        runpy.run_path(str(script), run_name="__main__")
    except StopCapture:
        pass
    except SystemExit as exc:
        exit_code = _exit_code(exc)
        if exit_code != 0:
            status, error = schema.STATUS_ERROR, f"script exited with code {exit_code}"
    except BaseException as exc:
        exit_code, status = 1, schema.STATUS_ERROR
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error, file=sys.stderr, end="")
    finally:
        session.finish(status, error)
        sys.argv, sys.path[:] = saved_argv, saved_path

    return RunResult(
        exit_code=exit_code,
        run_dir=run_dir,
        status=status,
        steps_captured=len(session.steps),
        batches_captured=session.batches_written,
    )


def summary(result: RunResult, requested_steps: int) -> str:
    host = socket.gethostname()
    lines = [
        f"tvis: captured {result.steps_captured}/{requested_steps} steps "
        f"({result.batches_captured} batches) → {result.run_dir}",
    ]
    if result.status == schema.STATUS_ERROR:
        lines.append("tvis: the script failed; everything captured before the failure was saved")
    lines.append("tvis: view it from your laptop with")
    lines.append(f"        tvis open {host}:{result.run_dir.parent}")
    lines.append("      (use your ssh alias for this machine, or add --via <jump-host> for multi-hop access)")
    if os.environ.get("SSH_CONNECTION") is None:
        lines.append(f"      or locally: tvis open {result.run_dir.parent}")
    return "\n".join(lines)


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1
