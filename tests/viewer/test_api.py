from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from tests.viewer.conftest import RUN_ID
from tvis.store import schema
from tvis.viewer.api import Api, ApiError


@pytest.fixture
def api(run_root: Path) -> Api:
    return Api(run_root)


def test_runs_lists_the_synthetic_run(api: Api):
    runs = api.dispatch("runs", {})

    assert [r["run_id"] for r in runs] == [RUN_ID]
    assert runs[0]["status"] == schema.STATUS_COMPLETE


def test_run_returns_meta_and_steps(api: Api):
    run = api.dispatch("run", {"run": RUN_ID})

    assert run["meta"]["script"] == "/proj/train.py"
    assert [s["batches"] for s in run["steps"]] == [[0]]
    assert run["steps"][0]["params"][0]["name"] == "embed.weight"


def test_tensor_returns_a_view_of_value_or_gradient(api: Api):
    value = api.dispatch("tensor", {"run": RUN_ID, "batch": 0, "tid": "t1", "fixed": [1]})
    grad = api.dispatch("tensor", {"run": RUN_ID, "batch": 0, "tid": "t1", "grad": True, "fixed": [1]})

    assert value["shape"] == [2, 3, 5] and value["grad"] is False
    assert (value["min"], value["max"]) == (15.0, 29.0)
    assert (grad["min"], grad["max"]) == (-29.0, -15.0)


def test_tensor_for_one_sample_uses_the_batch_dimension(api: Api):
    view = api.dispatch("tensor", {"run": RUN_ID, "batch": 0, "tid": "t2", "sample": 1})

    assert view["shape"] == [3, 4]  # sample 1's block of the [B*T, V] tensor
    assert view["min"] == 12.0


def test_image_of_a_normalised_tensor_sample(api: Api):
    image = api.dispatch("image", {"run": RUN_ID, "batch": 0, "tid": "t0", "sample": 0})

    assert (image["height"], image["width"]) == (4, 4)


def test_batched_tensors_are_unnormalised_by_default_and_can_opt_out(api: Api):
    params = {"run": RUN_ID, "batch": 0, "tid": "t0", "sample": 0}

    default = api.dispatch("image", params)
    raw = api.dispatch("image", {**params, "unnormalize": False})

    assert base64.b64decode(default["data"])[0] == 128  # 0 * 0.25 + 0.5 → mid-grey
    assert base64.b64decode(raw["data"])[0] == 0  # zeros stretched → black


def test_image_of_captured_pil_pixels(api: Api):
    image = api.dispatch("image", {"run": RUN_ID, "batch": 0, "tid": "t3"})

    assert (image["height"], image["width"]) == (4, 6)


def test_source_returns_the_snapshot(api: Api):
    assert api.dispatch("source", {"run": RUN_ID, "path": "train.py"})["text"] == "print('hi')\n"


@pytest.mark.parametrize(
    ("method", "params", "status"),
    [
        ("nope", {}, 404),
        ("__init__", {}, 404),
        ("run", {"run": "missing"}, 404),
        ("run", {"unexpected": 1}, 400),
        ("tensor", {"run": RUN_ID, "batch": 0, "tid": "t999"}, 404),
        ("tensor", {"run": RUN_ID, "batch": 0, "tid": "t4", "sample": 5}, 400),
        ("source", {"run": RUN_ID, "path": "../meta.json"}, 400),
        ("batch", {"run": RUN_ID, "batch": 7}, 404),
    ],
    ids=[
        "unknown",
        "dunder",
        "missing-run",
        "bad-params",
        "missing-tensor",
        "unstored-sample",
        "traversal",
        "no-batch",
    ],
)
def test_errors_map_to_api_errors_with_status(api: Api, method, params, status):
    with pytest.raises(ApiError) as info:
        api.dispatch(method, params)

    assert info.value.status == status


def test_batch_cache_is_invalidated_when_the_file_changes(api: Api, run_root: Path):
    assert api.dispatch("batch", {"run": RUN_ID, "batch": 0})["calls"][0]["name"] == "Net"

    path = run_root / RUN_ID / schema.BATCHES_DIR / schema.batch_dir_name(0) / schema.BATCH_FILE
    doc = json.loads(path.read_text())
    doc["calls"][0]["name"] = "Renamed"
    path.write_text(json.dumps(doc))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert api.dispatch("batch", {"run": RUN_ID, "batch": 0})["calls"][0]["name"] == "Renamed"
