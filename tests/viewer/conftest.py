"""Viewer fixtures: a synthetic run written with the real writer. No torch here (CI runs these
tests in an environment without torch, like a laptop running `tvis open`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tvis.store import schema
from tvis.store.writer import RunWriter

RUN_ID = "20261008-120000_train_abcd"
VOCAB = ["[PAD]", "the", "cat", "sat", "mat"]
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
            "data": {
                "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
                "vocab": VOCAB,
            },
        }
    )
    writer.write_steps(
        [
            {
                "index": 0,
                "batches": [0],
                "params": [
                    {
                        "name": "embed.weight",
                        "shape": [5, 3],
                        "weight": {"kind": "tensor", "tid": "t9", "batch": 0},
                    }
                ],
            }
        ]
    )
    images = np.zeros((BATCH, 3, 4, 4), np.float32)  # normalised zeros → mid-grey after un-normalising
    hidden = np.arange(BATCH * 3 * 5, dtype=np.float32).reshape(BATCH, 3, 5)
    merged = np.arange(BATCH * 3 * 4, dtype=np.float32).reshape(BATCH * 3, 4)  # [B*T, V]
    pixels = np.full((4, 6, 3), 200, np.uint8)
    writer.write_batch(
        0,
        {
            "index": 0,
            "batch_size": BATCH,
            "calls": [
                {
                    "id": 0,
                    "parent": None,
                    "kind": "module",
                    "name": "Net",
                    "phase": "forward",
                    "inputs": [],
                    "outputs": [],
                },
                {
                    "id": 1,
                    "parent": 0,
                    "kind": "module",
                    "name": "enc",
                    "module_path": "enc",
                    "phase": "forward",
                    "inputs": [],
                    "outputs": [{"name": "output", "value": {"kind": "tensor", "tid": "t1", "batch": 0}}],
                },
                {
                    "id": 2,
                    "parent": None,
                    "kind": "loss",
                    "name": "cross_entropy",
                    "phase": "loss",
                    "inputs": [
                        {"name": "input", "value": {"kind": "tensor", "tid": "t5", "batch": 0}},
                        {"name": "target", "value": {"kind": "tensor", "tid": "t6", "batch": 0}},
                    ],
                    "outputs": [{"name": "loss", "value": {"kind": "tensor", "tid": "t8", "batch": 0}}],
                    "extra": {
                        "reduction": "mean",
                        "ignore_index": -100,
                        "per_element": {"kind": "tensor", "tid": "t7", "batch": 0},
                    },
                },
            ],
            "samples": [
                {
                    "position": 0,
                    "target": {"id": 1, "name": "dog"},
                    "prediction": {"correct": True},
                    "loss": 0.5,
                },
                {
                    "position": 1,
                    "target": {"id": 0, "name": "cat"},
                    "prediction": {"correct": False},
                    "loss": 2.0,
                },
            ],
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
                "t5": tensor_meta("t5", [6, 4], batch_dim={"dim": 0, "block": 3, "inferred": True}),
                "t6": {**tensor_meta("t6", [6]), "dtype": "int64"},
                "t7": tensor_meta("t7", [6]),
                "t8": {**tensor_meta("t8", []), "stats": {"mean": 1.25}},
                "t9": tensor_meta("t9", [5, 3]),
                "t4": tensor_meta(
                    "t4",
                    [8, 5],
                    batch_dim={"dim": 0, "block": 1, "inferred": False},
                    stored="rows",
                    stored_rows=1,
                ),
            },
            "ops": [],
        },
    )
    writer.save_array(0, schema.value_array_name("t0"), images)
    writer.save_array(0, schema.value_array_name("t1"), hidden)
    writer.save_array(0, schema.grad_array_name("t1"), -hidden)
    writer.save_array(0, schema.value_array_name("t2"), merged)
    writer.save_array(0, schema.value_array_name("t3"), pixels)
    writer.save_array(0, schema.value_array_name("t4"), np.ones((1, 5), np.float32))
    logits = np.array(
        [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 5, 0, 0]], np.float32
    )
    writer.save_array(0, schema.value_array_name("t5"), logits)
    writer.save_array(0, schema.value_array_name("t6"), np.array([0, 1, -100, 2, 3, 1], np.int64))
    writer.save_array(0, schema.value_array_name("t7"), np.array([0.2, 0.1, 0.0, 1.5, 0.3, 0.01], np.float32))
    writer.save_array(0, schema.value_array_name("t8"), np.array(0.42, np.float32))
    writer.save_array(0, schema.value_array_name("t9"), np.arange(15, dtype=np.float32).reshape(5, 3) ** 1.5)
    writer.write_source("train.py", "print('hi')\n")
    writer.close()
    return root
