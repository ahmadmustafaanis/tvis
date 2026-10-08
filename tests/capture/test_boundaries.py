"""Batch / step state machine: how a training loop is cut into batches and steps."""

from __future__ import annotations

import torch

from tests.capture import models
from tvis.store import schema


def test_each_optimizer_step_closes_a_step_and_capture_stops_after_n(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    result = capture(lambda: models.train(model, opt, models.make_batches(10)), steps=3)

    assert result.stopped
    assert [s["batches"] for s in result.steps] == [[0], [1], [2]]
    assert result.meta["status"] == schema.STATUS_COMPLETE
    assert result.meta["steps_captured"] == 3 and result.meta["batches_captured"] == 3


def test_training_stops_right_after_the_last_captured_step(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    seen: list[torch.Tensor] = []

    capture(lambda: models.train(model, opt, models.make_batches(10), losses=seen), steps=2)

    assert len(seen) == 2  # the third batch's forward never ran


def test_gradient_accumulation_groups_micro_batches_into_one_step(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    result = capture(lambda: models.train(model, opt, models.make_batches(8), accum=2), steps=2)

    assert [s["batches"] for s in result.steps] == [[0, 1], [2, 3]]
    assert [result.batch(i)["step"] for i in range(4)] == [0, 0, 1, 1]
    assert all(s["optimizer_steps"] == 1 for s in result.steps)


def test_without_torch_optim_every_backward_is_a_step(capture):
    model = models.ToyMLP()

    result = capture(lambda: models.train_manual_sgd(model, models.make_batches(6)), steps=3)

    assert result.stopped
    assert [s["batches"] for s in result.steps] == [[0], [1], [2]]
    assert all(s["optimizer_steps"] == 0 for s in result.steps)


def test_two_optimizers_close_steps_lazily_at_the_next_batch(capture):
    gen, disc = models.ToyMLP(n_classes=6), models.ToyMLP(d_in=6, n_classes=1)
    opt_g = torch.optim.SGD(gen.parameters(), lr=0.1)
    opt_d = torch.optim.SGD(disc.parameters(), lr=0.1)

    def loop():
        for x, _ in models.make_batches(6):
            d_loss = disc(x).mean()
            d_loss.backward()
            opt_d.step()
            g_loss = disc(gen(x)).mean()
            g_loss.backward()
            opt_g.step()

    result = capture(loop, steps=2)

    assert result.stopped
    assert len(result.steps) == 2
    assert all(s["optimizer_steps"] >= 1 for s in result.steps)
    assert len(result.meta["optimizers"]) == 2


def test_repeated_grad_enabled_forwards_before_backward_share_one_batch(capture):
    # siamese / contrastive training: two views through the same model, one loss
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    (a, _), (b, _) = models.make_batches(2)

    def loop():
        (model(a) - model(b)).pow(2).mean().backward()
        opt.step()

    result = capture(loop, steps=1)

    assert result.meta["batches_captured"] == 1
    assert len(result.calls(0, kind="module", name="ToyMLP")) == 2


def test_forward_without_backward_is_recorded_and_flagged(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    (x, _y), *_ = models.make_batches(1)

    def loop():
        with torch.no_grad():
            model(x)  # e.g. a sanity-check forward before training
        models.train(model, opt, models.make_batches(3))

    result = capture(loop, steps=1)

    assert "no_backward" in result.batch(0)["flags"]
    assert "no_backward" not in result.batch(1)["flags"]


def test_code_before_the_first_batch_is_not_recorded(capture):
    def loop():
        model = models.ToyMLP()  # construction: module __init__, not a batch
        models.scale_and_shift(torch.ones(2), 1.0, torch.zeros(2))  # project function outside a batch
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        models.train(model, opt, models.make_batches(2))

    result = capture(loop, steps=1)

    first = result.batch(0)["calls"][0]
    assert first["kind"] == "module" and first["name"] == "ToyMLP"


def test_max_batches_bounds_a_run_that_never_steps(capture):
    model = models.ToyMLP()

    def inference_only():
        with torch.no_grad():
            for x, _ in models.make_batches(20):
                model(x)

    result = capture(inference_only, steps=1, max_batches=5)

    assert result.stopped
    assert result.meta["batches_captured"] == 5
    assert any("max_batches" in n["message"] for n in result.meta["notices"])


def test_script_that_ends_early_still_writes_partial_results(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    result = capture(lambda: models.train(model, opt, models.make_batches(2)), steps=5)

    assert not result.stopped
    assert result.meta["steps_captured"] == 2
    assert result.meta["status"] == schema.STATUS_COMPLETE


def test_failure_inside_instrumentation_never_breaks_training(capture, monkeypatch):
    import tvis.capture.tensors as tensors

    def broken_stats(_tensor):
        raise RuntimeError("stats exploded")

    monkeypatch.setattr(tensors, "tensor_stats", broken_stats)
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses: list[torch.Tensor] = []

    result = capture(lambda: models.train(model, opt, models.make_batches(3), losses=losses), steps=2)

    assert len(losses) == 2 and result.stopped
    assert any("stats exploded" in n["message"] for n in result.meta["notices"])


def test_phase_timings_are_reported_per_batch(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    result = capture(lambda: models.train(model, opt, models.make_batches(2)), steps=1)

    timing = result.batch(0)["timing"]
    assert {"forward", "loss", "backward", "optimizer"} <= set(timing["phases_ms"])
    assert all(v >= 0 for v in timing["phases_ms"].values())
    assert timing["total_ms"] == sum(timing["phases_ms"].values())
    assert timing["clock"] == "perf_counter"
