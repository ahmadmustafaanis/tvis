"""What ends up in the call tree: modules, project functions, call sites, line-level ops."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.capture import models


def run_mlp(capture, **config):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return capture(lambda: models.train(model, opt, models.make_batches(2)), steps=1, **config)


def test_modules_are_named_by_their_path_in_the_model(capture):
    result = run_mlp(capture)

    names = [c["name"] for c in result.calls(kind="module")]
    assert names == ["ToyMLP", "fc1", "act", "drop", "fc2"]
    assert result.call(kind="module", name="fc1")["cls"] == "Linear"


def test_project_function_called_inside_forward_is_a_child_of_the_model_call(capture):
    result = run_mlp(capture)
    root = result.call(kind="module", name="ToyMLP")
    fn = result.call(kind="function", name="scale_and_shift")

    assert fn["parent"] == root["id"]
    assert [i["name"] for i in fn["inputs"]] == ["x", "scale", "shift"]
    assert fn["file"] == "models.py"
    assert fn["line"] == models.scale_and_shift.__code__.co_firstlineno


def test_user_module_forward_is_merged_into_the_module_call_not_duplicated(capture):
    result = run_mlp(capture)

    assert result.calls(kind="function", name="forward") == []
    root = result.call(kind="module", name="ToyMLP")
    assert root["file"] == "models.py" and root["line"] == models.ToyMLP.forward.__code__.co_firstlineno
    assert [i["name"] for i in root["inputs"]] == ["x"]


def test_call_sites_point_at_the_project_line_that_made_the_call(capture):
    result = run_mlp(capture)

    fc1 = result.call(kind="module", name="fc1")
    assert fc1["call_site"] == {
        "file": "models.py",
        "line": models.line_of(models.ToyMLP.forward, "self.fc1(x)"),
    }
    loss_fn = result.call(kind="function", name="compute_loss")
    assert loss_fn["call_site"]["line"] == models.line_of(models.train, "compute_loss(model(x), y)")


def test_loss_computed_outside_the_model_is_in_the_loss_phase(capture):
    result = run_mlp(capture)

    assert result.call(kind="function", name="compute_loss")["phase"] == "loss"
    ce = result.call(kind="loss", name="cross_entropy")
    assert ce["phase"] == "loss"
    assert ce["parent"] == result.call(kind="function", name="compute_loss")["id"]


def test_library_functions_and_dunder_methods_are_not_traced(capture):
    result = run_mlp(capture)

    traced = {c["name"] for c in result.calls(kind="function")}
    assert traced == {"scale_and_shift", "compute_loss"}


def test_function_tracing_can_be_disabled(capture):
    result = run_mlp(capture, trace_functions=False)

    assert result.calls(kind="function") == []
    assert result.calls(kind="module")  # modules are still recorded


def test_children_time_never_exceeds_parent_time(capture):
    result = run_mlp(capture)
    calls = result.batch(0)["calls"]

    for parent in calls:
        children = [c for c in calls if c["parent"] == parent["id"] and "ms" in c]
        if children and "ms" in parent:
            assert sum(c["ms"] for c in children) <= parent["ms"] * 1.05 + 0.05, parent["name"]


def test_children_lie_within_their_parent_on_the_corrected_timeline(capture):
    result = run_mlp(capture)
    calls = {c["id"]: c for c in result.batch(0)["calls"]}

    for call in calls.values():
        parent = calls.get(call["parent"])
        if parent is None or "start_ms" not in call or "start_ms" not in parent:
            continue
        assert parent["start_ms"] - 1e-6 <= call["start_ms"] <= call["end_ms"] <= parent["end_ms"] + 1e-6
        assert call["end_ms"] - call["start_ms"] == pytest.approx(call["ms"], abs=1e-6)


def test_backward_time_is_reported_for_modules_and_functions_and_labelled_approximate(capture):
    result = run_mlp(capture)

    for name in ("ToyMLP", "fc1", "fc2"):
        call = result.call(kind="module", name=name)
        assert call["bwd_ms"] >= 0 and call["bwd_approx"] is True
    assert "bwd_ms" in result.call(kind="function", name="scale_and_shift")


def test_backward_and_optimizer_calls_are_recorded(capture):
    result = run_mlp(capture)

    backward = result.call(kind="backward", name="backward")
    assert backward["phase"] == "backward"
    assert backward["call_site"]["line"] == models.line_of(models.train, "loss.backward()")
    optimizer = result.call(kind="optimizer", name="SGD")
    assert optimizer["extra"]["hyper"][0]["lr"] == 0.1


def test_line_level_ops_record_the_shape_produced_by_each_line(capture):
    model = models.AttnModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 5, 4)

    def loop():
        model(x).sum().backward()
        opt.step()

    result = capture(loop, steps=1)

    ops = {(o["line"], o["op"]): o for o in result.batch(0)["ops"]}
    scores_line = models.line_of(models.attention_like, "scores = q @ k")
    softmax_line = models.line_of(models.attention_like, "softmax")
    assert ops[(scores_line, "transpose")]["shapes"] == [[2, 4, 5]]
    assert ops[(scores_line, "matmul")]["shapes"] == [[2, 5, 5]]
    assert ops[(softmax_line, "softmax")]["shapes"] == [[2, 5, 5]]
    attention_call = result.call(kind="function", name="attention_like")
    assert {o["call"] for o in result.batch(0)["ops"] if o["line"] == scores_line} == {attention_call["id"]}


def test_ops_inside_library_modules_are_not_attributed_to_user_lines(capture):
    result = run_mlp(capture)

    op_names = {o["op"] for o in result.batch(0)["ops"]}
    assert "linear" not in op_names  # runs inside nn.Linear.forward
    assert {"mul", "add"} <= op_names  # scale_and_shift's own lines


def test_ops_recording_can_be_disabled(capture):
    assert run_mlp(capture, record_ops=False).batch(0)["ops"] == []


def test_batchnorm_in_training_mode_is_flagged_as_mixing_samples(capture):
    model = models.BatchNormNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def loop():
        model(torch.randn(4, 4)).sum().backward()
        opt.step()

    result = capture(loop, steps=1)

    assert result.call(kind="module", name="bn")["flags"] == ["mixes_samples"]
    assert "flags" not in result.call(kind="module", name="fc")


def test_source_files_that_executed_are_snapshotted(capture):
    result = run_mlp(capture)

    assert "models.py" in result.meta["sources"]
    assert result.run.source("models.py") == Path(models.__file__).read_text()


def test_model_called_through_methods_keeps_its_layer_paths(capture):
    model = models.SplitForwardNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    result = capture(lambda: models.train_split(model, opt, models.make_batches(3)), steps=2)

    assert [m["cls"] for m in result.meta["models"]] == ["SplitForwardNet"]
    modules = {c["name"] for c in result.calls(kind="module")}
    assert {"embed", "blocks.0", "blocks.1", "blocks.2", "head"} <= modules
    assert {p["name"] for p in result.steps[0]["params"]} == {n for n, _ in model.named_parameters()}
    assert [s["batches"] for s in result.steps] == [[0], [1]]
