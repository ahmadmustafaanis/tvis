"""Root fixtures. Must not import torch: tests/store and tests/viewer run in a torch-free CI job."""

from __future__ import annotations

from pathlib import Path

import pytest

from tvis.store.writer import RunWriter


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def make_run(runs_root: Path):
    """Create a minimal valid run directory; returns a factory `(run_id, **meta) -> RunWriter`."""
    writers: list[RunWriter] = []

    def factory(run_id: str, **meta) -> RunWriter:
        writer = RunWriter(runs_root / run_id)
        writer.write_meta({"run_id": run_id, "status": "complete", **meta})
        writers.append(writer)
        return writer

    yield factory
    for writer in writers:
        writer.close()
