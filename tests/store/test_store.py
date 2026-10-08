from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tvis.store import schema
from tvis.store.reader import Run, RunNotFound, RunsRoot
from tvis.store.writer import RunWriter, write_json_atomic


class TestAtomicJson:
    def test_write_leaves_no_temp_file_behind(self, tmp_path: Path):
        target = tmp_path / "doc.json"
        write_json_atomic(target, {"a": 1})

        assert json.loads(target.read_text()) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["doc.json"]

    def test_rewrite_replaces_previous_document(self, tmp_path: Path):
        target = tmp_path / "doc.json"
        write_json_atomic(target, {"version": 1})
        write_json_atomic(target, {"version": 2})

        assert json.loads(target.read_text()) == {"version": 2}

    def test_non_finite_floats_are_rejected_not_written_as_invalid_json(self, tmp_path: Path):
        # Stats must encode NaN/Inf explicitly (as counts / None); bare NaN would break JS JSON.parse.
        with pytest.raises(ValueError):
            write_json_atomic(tmp_path / "doc.json", {"mean": float("nan")})

    def test_numpy_scalars_are_serialised_as_python_numbers(self, tmp_path: Path):
        target = tmp_path / "doc.json"
        write_json_atomic(target, {"n": np.int64(3), "x": np.float32(0.5)})

        assert json.loads(target.read_text()) == {"n": 3, "x": 0.5}


class TestRunRoundTrip:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.bool_, np.uint8])
    def test_arrays_round_trip_exactly_with_dtype(self, tmp_path: Path, dtype):
        values = (np.arange(24) % 5).astype(dtype).reshape(2, 3, 4)
        writer = RunWriter(tmp_path / "run")
        writer.write_meta({"run_id": "run"})
        writer.save_array(0, schema.value_array_name("t1"), values)
        writer.close()

        loaded = Run(tmp_path / "run").array(0, "t1")

        assert loaded.dtype == values.dtype
        np.testing.assert_array_equal(loaded, values)

    def test_value_and_gradient_arrays_are_stored_separately(self, tmp_path: Path):
        writer = RunWriter(tmp_path / "run")
        writer.save_array(2, schema.value_array_name("t7"), np.ones(3, np.float32))
        writer.save_array(2, schema.grad_array_name("t7"), np.full(3, -2.0, np.float32))
        writer.close()
        run = Run(tmp_path / "run")

        np.testing.assert_array_equal(run.array(2, "t7"), [1, 1, 1])
        np.testing.assert_array_equal(run.array(2, "t7", grad=True), [-2, -2, -2])

    def test_meta_records_format_version(self, tmp_path: Path):
        writer = RunWriter(tmp_path / "run")
        writer.write_meta({"run_id": "run"})
        writer.close()

        assert Run(tmp_path / "run").meta()["format_version"] == schema.FORMAT_VERSION

    def test_batch_and_steps_documents_round_trip(self, tmp_path: Path):
        writer = RunWriter(tmp_path / "run")
        writer.write_batch(1, {"index": 1, "calls": [{"id": 0, "name": "forward"}]})
        writer.write_steps([{"index": 0, "batches": [0, 1]}])
        writer.close()
        run = Run(tmp_path / "run")

        assert run.batch(1)["calls"][0]["name"] == "forward"
        assert run.steps() == [{"index": 0, "batches": [0, 1]}]

    def test_missing_steps_file_reads_as_empty(self, tmp_path: Path):
        RunWriter(tmp_path / "run").close()

        assert Run(tmp_path / "run").steps() == []

    def test_missing_batch_raises_run_not_found(self, tmp_path: Path):
        RunWriter(tmp_path / "run").close()

        with pytest.raises(RunNotFound):
            Run(tmp_path / "run").batch(5)

    def test_source_snapshot_preserves_nested_paths(self, tmp_path: Path):
        writer = RunWriter(tmp_path / "run")
        writer.write_source("models/attn.py", "def f():\n    return 1\n")
        writer.close()

        assert Run(tmp_path / "run").source("models/attn.py") == "def f():\n    return 1\n"


class TestPathSafety:
    @pytest.mark.parametrize("bad", ["../meta.json", "/etc/passwd", "a/../../b", "", ".."])
    def test_source_paths_cannot_escape_run_directory(self, tmp_path: Path, bad: str):
        RunWriter(tmp_path / "run").close()

        with pytest.raises(ValueError):
            Run(tmp_path / "run").source(bad)

    def test_array_names_cannot_escape_arrays_directory(self, tmp_path: Path):
        RunWriter(tmp_path / "run").close()

        with pytest.raises(ValueError):
            Run(tmp_path / "run").array(0, "../../../meta")


class TestRunsRoot:
    def test_lists_runs_newest_first(self, make_run, runs_root: Path):
        make_run("old", started_at="2026-10-01T10:00:00", script="a.py")
        make_run("new", started_at="2026-10-08T10:00:00", script="b.py")

        runs = RunsRoot(runs_root).list_runs()

        assert [r["run_id"] for r in runs] == ["new", "old"]
        assert runs[0]["script"] == "b.py"

    def test_ignores_directories_without_meta(self, make_run, runs_root: Path):
        make_run("real")
        (runs_root / "not-a-run").mkdir()

        assert [r["run_id"] for r in RunsRoot(runs_root).list_runs()] == ["real"]

    def test_skips_runs_with_corrupt_meta(self, make_run, runs_root: Path):
        make_run("good")
        (runs_root / "broken").mkdir()
        (runs_root / "broken" / schema.META_FILE).write_text("{not json")

        assert [r["run_id"] for r in RunsRoot(runs_root).list_runs()] == ["good"]

    def test_root_pointing_at_a_single_run_lists_that_run(self, make_run, runs_root: Path):
        make_run("only")

        runs = RunsRoot(runs_root / "only").list_runs()

        assert [r["run_id"] for r in runs] == ["only"]

    def test_unknown_run_id_raises(self, runs_root: Path):
        with pytest.raises(RunNotFound):
            RunsRoot(runs_root).run("nope")

    def test_nonexistent_root_lists_nothing(self, tmp_path: Path):
        assert RunsRoot(tmp_path / "missing").list_runs() == []
