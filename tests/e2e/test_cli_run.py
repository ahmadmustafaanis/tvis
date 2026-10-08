"""`tvis run` as a user runs it: a subprocess around an unmodified script."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tvis.store import schema
from tvis.store.reader import RunsRoot

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


def tvis_run(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tvis.cli", "run", *args], cwd=cwd, capture_output=True, text=True, timeout=300
    )


def python(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True, timeout=300)


def only_run(out: Path):
    runs = RunsRoot(out).list_runs()
    assert len(runs) == 1, runs
    return RunsRoot(out).run(runs[0]["run_id"])


def read_losses(path: Path) -> list[float]:
    return [json.loads(line)["loss"] for line in path.read_text().splitlines()]


@pytest.fixture
def script(tmp_path: Path):
    def write(body: str) -> Path:
        path = tmp_path / "proj" / "train.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(textwrap.dedent(body))
        return path

    return write


TOY_TRAINING = """
    import sys
    import torch

    def main():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(10):
            model(torch.randn(8, 4)).pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
        {tail}

    if __name__ == "__main__":
        main()
"""


@pytest.mark.parametrize(
    ("example", "args", "steps", "batches"),
    [
        ("image_classifier", ["--accum", "2"], 2, 4),
        ("text_lm", [], 2, 2),
    ],
)
def test_examples_are_captured_end_to_end(tmp_path: Path, example, args, steps, batches):
    out = tmp_path / "runs"
    proc = tvis_run("--steps", str(steps), "--out", str(out), str(EXAMPLES / example / "train.py"), *args)

    assert proc.returncode == 0, proc.stderr
    assert f"captured {steps}/{steps} steps ({batches} batches)" in proc.stderr
    assert "tvis open " in proc.stderr
    meta = only_run(out).meta()
    assert meta["status"] == schema.STATUS_COMPLETE
    assert meta["steps_captured"] == steps and meta["batches_captured"] == batches
    assert meta["sources"] == ["train.py"]


def test_training_under_tvis_matches_plain_python_bit_for_bit(tmp_path: Path):
    train = str(EXAMPLES / "image_classifier" / "train.py")
    plain_log, tvis_log = tmp_path / "plain.jsonl", tmp_path / "tvis.jsonl"

    plain = python(train, "--workers", "0", "--log", str(plain_log))
    captured = tvis_run(
        "--steps", "3", "--out", str(tmp_path / "runs"), train, "--workers", "0", "--log", str(tvis_log)
    )

    assert plain.returncode == 0 and captured.returncode == 0, captured.stderr
    tvis_losses = read_losses(tvis_log)
    assert len(tvis_losses) == 3
    assert tvis_losses == read_losses(plain_log)[:3]  # exact float equality


def test_script_arguments_and_main_guard_behave_like_python(tmp_path: Path, script):
    path = script(
        """
        import sys, json, pathlib
        if __name__ == "__main__":
            pathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv[1:]))
        """
    )
    record = tmp_path / "argv.json"

    proc = tvis_run("--out", str(tmp_path / "runs"), str(path), "--flag", str(record), "positional")

    assert proc.returncode == 0, proc.stderr
    assert json.loads(record.read_text()) == ["--flag", str(record), "positional"]


def test_script_exit_code_is_propagated(tmp_path: Path, script):
    path = script(TOY_TRAINING.format(tail="sys.exit(3)").replace("range(10)", "range(1)"))

    proc = tvis_run("--steps", "5", "--out", str(tmp_path / "runs"), str(path))

    assert proc.returncode == 3
    assert only_run(tmp_path / "runs").meta()["status"] == schema.STATUS_ERROR


def test_crashing_script_keeps_everything_captured_before_the_crash(tmp_path: Path, script):
    path = script(
        TOY_TRAINING.format(tail="")
        .replace("for _ in range(10):", "for i in range(10):")
        .replace("opt.zero_grad()", "opt.zero_grad()\n            if i == 0: raise ValueError('boom')")
    )

    proc = tvis_run("--steps", "5", "--out", str(tmp_path / "runs"), str(path))

    assert proc.returncode == 1
    assert "ValueError: boom" in proc.stderr
    meta = only_run(tmp_path / "runs").meta()
    assert meta["status"] == schema.STATUS_ERROR
    assert "ValueError: boom" in meta["error"]
    assert meta["steps_captured"] == 1


def test_capture_stops_the_script_after_the_requested_steps(tmp_path: Path, script):
    path = script(TOY_TRAINING.format(tail="print('REACHED END')"))

    proc = tvis_run("--steps", "2", "--out", str(tmp_path / "runs"), str(path))

    assert proc.returncode == 0, proc.stderr
    assert "REACHED END" not in proc.stdout
    assert only_run(tmp_path / "runs").meta()["steps_captured"] == 2


def test_default_output_directory_is_next_to_the_script(script):
    path = script(TOY_TRAINING.format(tail=""))

    proc = tvis_run("--steps", "1", str(path), cwd=path.parent)

    assert proc.returncode == 0, proc.stderr
    assert len(RunsRoot(path.parent / "tvis_runs").list_runs()) == 1


def test_missing_script_is_a_usage_error(tmp_path: Path):
    proc = tvis_run(str(tmp_path / "nope.py"))

    assert proc.returncode == 2
    assert "not found" in proc.stderr
