"""Capture-test fixtures. Runs code under a real Session and reads the result back from disk with
the viewer-side reader, so every test exercises the full capture → store → read path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tvis.capture.session import CaptureConfig, Session, StopCapture
from tvis.store.reader import Run

PROJECT_DIR = Path(__file__).parent


@dataclass
class Captured:
    session: Session
    run: Run
    stopped: bool

    @property
    def meta(self) -> dict[str, Any]:
        return self.run.meta()

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self.run.steps()

    def batch(self, index: int = 0) -> dict[str, Any]:
        return self.run.batch(index)

    def calls(self, batch: int = 0, *, kind: str | None = None, name: str | None = None) -> list[dict]:
        return [
            c
            for c in self.batch(batch)["calls"]
            if (kind is None or c["kind"] == kind) and (name is None or c["name"] == name)
        ]

    def call(self, batch: int = 0, *, kind: str | None = None, name: str | None = None) -> dict:
        found = self.calls(batch, kind=kind, name=name)
        assert len(found) == 1, f"expected one {kind or ''} call named {name!r}, got {len(found)}"
        return found[0]

    def children(self, call: dict, batch: int = 0) -> list[dict]:
        return [c for c in self.batch(batch)["calls"] if c["parent"] == call["id"]]

    def tensor(self, ref: dict | str, batch: int = 0) -> dict:
        tid = ref if isinstance(ref, str) else ref["tid"]
        b = ref["batch"] if isinstance(ref, dict) else batch
        return self.batch(b)["tensors"][tid]

    def value(self, ref: dict, *, grad: bool = False) -> np.ndarray:
        return np.array(self.run.array(ref["batch"], ref["tid"], grad=grad))

    def output_ref(self, call: dict, index: int = 0) -> dict:
        value = call["outputs"][index]["value"]
        assert value["kind"] == "tensor", value
        return value

    def input_ref(self, call: dict, name: str) -> dict:
        for entry in call["inputs"]:
            if entry["name"] == name:
                return entry["value"]
        raise AssertionError(f"{call['name']} has no input {name!r}: {[e['name'] for e in call['inputs']]}")


@pytest.fixture
def capture(tmp_path: Path) -> Callable[..., Captured]:
    """`capture(fn, steps=..., **config)` runs `fn` under a session and returns what was recorded."""
    counter = iter(range(1000))

    def run(fn: Callable[[], Any], *, steps: int = 3, **config: Any) -> Captured:
        run_dir = tmp_path / f"run{next(counter)}"
        session = Session(CaptureConfig(run_dir=run_dir, project_dir=PROJECT_DIR, steps=steps, **config))
        session.install()
        stopped = False
        try:
            fn()
        except StopCapture:
            stopped = True
        finally:
            session.finish()
        return Captured(session=session, run=Run(run_dir), stopped=stopped)

    return run


@pytest.fixture(autouse=True)
def _no_instrumentation_leaks():
    """Every test must leave torch exactly as it found it."""
    from torch.nn.modules import module as torch_module
    from torch.optim import optimizer as torch_optimizer
    from torch.utils.data import DataLoader

    before = (
        torch.autograd.backward,
        torch.optim.Optimizer.__init__,
        DataLoader.__iter__,
        DataLoader.__init__,
        torch.nn.functional.cross_entropy,
        len(torch_module._global_forward_pre_hooks),
        len(torch_module._global_forward_hooks),
        len(torch_optimizer._global_optimizer_pre_hooks),
        len(torch_optimizer._global_optimizer_post_hooks),
    )
    yield
    after = (
        torch.autograd.backward,
        torch.optim.Optimizer.__init__,
        DataLoader.__iter__,
        DataLoader.__init__,
        torch.nn.functional.cross_entropy,
        len(torch_module._global_forward_pre_hooks),
        len(torch_module._global_forward_hooks),
        len(torch_optimizer._global_optimizer_pre_hooks),
        len(torch_optimizer._global_optimizer_post_hooks),
    )
    assert after == before, "instrumentation leaked past Session.finish()"
