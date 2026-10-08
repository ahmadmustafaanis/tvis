"""Viewer fixtures: a synthetic run written with the real writer. No torch here (CI runs these
tests in an environment without torch, like a laptop running `tvis open`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tvis.store import schema
from tvis.store.writer import RunWriter

RUN_ID = "20261008-120000_train_abcd"
BATCH = 2  # batch size of the synthetic run


def tensor_meta(tid: str, shape: list[int], **extra) -> dict:
    return {
        "tid": tid,
        "batch": 0,
        "kind": "tensor",
        "shape": shape,
        "dtype": "float32",
        "stored": "full",
        **extra,
    }


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    writer = RunWriter(root / RUN_ID)
    writer.write_meta(
        {
            "run_id": RUN_ID,
            "status": schema.STATUS_COMPLETE,
            "script": "/proj/train.py",
            "started_at": "2026-10-08T12:00:00+00:00",
            "steps_captured": 1,
            "batches_captured": 1,
            "data": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}},
        }
    )
    writer.write_steps([{"index": 0, "batches": [0], "params": []}])
    images = np.zeros((BATCH, 3, 4, 4), np.float32)  # normalised zeros → mid-grey after un-normalising
    hidden = np.arange(BATCH * 3 * 5, dtype=np.float32).reshape(BATCH, 3, 5)
    merged = np.arange(BATCH * 3 * 4, dtype=np.float32).reshape(BATCH * 3, 4)  # [B*T, V]
    pixels = np.full((4, 6, 3), 200, np.uint8)
    writer.write_batch(
        0,
        {
            "index": 0,
            "batch_size": BATCH,
            "calls": [{"id": 0, "parent": None, "kind": "module", "name": "Net"}],
            "tensors": {
                "t0": tensor_meta(
                    "t0", [BATCH, 3, 4, 4], batch_dim={"dim": 0, "block": 1, "inferred": False}
                ),
                "t1": tensor_meta(
                    "t1",
                    [BATCH, 3, 5],
                    batch_dim={"dim": 0, "block": 1, "inferred": False},
                    grad_stored="full",
                ),
                "t2": tensor_meta("t2", [BATCH * 3, 4], batch_dim={"dim": 0, "block": 3, "inferred": True}),
                "t3": {**tensor_meta("t3", [4, 6, 3]), "kind": "image", "dtype": "uint8:RGB"},
                "t4": tensor_meta(
                    "t4",
                    [8, 5],
                    batch_dim={"dim": 0, "block": 1, "inferred": False},
                    stored="rows",
                    stored_rows=1,
                ),
            },
            "ops": [],
            "samples": [],
        },
    )
    writer.save_array(0, schema.value_array_name("t0"), images)
    writer.save_array(0, schema.value_array_name("t1"), hidden)
    writer.save_array(0, schema.grad_array_name("t1"), -hidden)
    writer.save_array(0, schema.value_array_name("t2"), merged)
    writer.save_array(0, schema.value_array_name("t3"), pixels)
    writer.save_array(0, schema.value_array_name("t4"), np.ones((1, 5), np.float32))
    writer.write_source("train.py", "print('hi')\n")
    writer.close()
    return root
