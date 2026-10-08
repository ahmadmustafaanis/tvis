"""The core accuracy guarantees: captured gradients are the real gradients, and observing
training never changes it."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.capture import models


def fresh_model(seed: int = 0, **kwargs) -> models.ToyMLP:
    torch.manual_seed(seed)
    return models.ToyMLP(**kwargs)


def train_reference(model, batches, steps: int, lr: float = 0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[torch.Tensor] = []
    models.train(model, opt, batches[:steps], losses=losses)
    return losses


class TestTrainingIsUnchanged:
    def test_losses_and_parameters_are_bit_identical_with_and_without_capture(self, capture):
        batches = models.make_batches(4)
        ref_model = fresh_model(dropout=0.3)
        torch.manual_seed(123)  # dropout RNG stream
        ref_losses = train_reference(ref_model, batches, steps=3)

        model = fresh_model(dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        losses: list[torch.Tensor] = []
        torch.manual_seed(123)
        result = capture(lambda: models.train(model, opt, batches, losses=losses), steps=3)

        assert result.stopped
        assert len(losses) == len(ref_losses) == 3
        for got, want in zip(losses, ref_losses, strict=True):
            assert torch.equal(got, want)
        for (name, got), (_, want) in zip(
            model.named_parameters(), ref_model.named_parameters(), strict=True
        ):
            assert torch.equal(got, want), name

    def test_rng_state_is_not_consumed_by_capture(self, capture):
        model = fresh_model(dropout=0.5)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        torch.manual_seed(7)
        capture(lambda: models.train(model, opt, models.make_batches(2)), steps=2)
        after_capture = torch.rand(3)

        model = fresh_model(dropout=0.5)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        torch.manual_seed(7)
        models.train(model, opt, models.make_batches(2))
        after_plain = torch.rand(3)

        assert torch.equal(after_capture, after_plain)

    def test_shuffled_dataloader_yields_the_same_batches_under_capture(self, capture):
        data = TensorDataset(torch.arange(32.0).reshape(16, 2), torch.zeros(16, dtype=torch.long))

        def first_batches(out: list):
            torch.manual_seed(3)
            for x, _ in DataLoader(data, batch_size=4, shuffle=True):
                out.append(x)

        plain: list = []
        first_batches(plain)
        captured: list = []
        capture(lambda: first_batches(captured), steps=1, max_batches=10)

        assert len(captured) == len(plain)
        assert all(torch.equal(a, b) for a, b in zip(captured, plain, strict=True))


class TestGradientsAreCorrect:
    def test_activation_gradient_equals_autograd_reference(self, capture):
        (x, y), *_ = models.make_batches(1)
        model = fresh_model()
        reference = copy.deepcopy(model)

        hidden = reference.fc1(x)
        hidden.retain_grad()
        h = models.scale_and_shift(reference.act(hidden), 2.0, reference.shift)
        models.compute_loss(reference.fc2(reference.drop(h)), y).backward()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

        fc1_out = result.output_ref(result.call(kind="module", name="fc1"))
        assert result.tensor(fc1_out)["grad_stored"] == "full"
        torch.testing.assert_close(torch.from_numpy(result.value(fc1_out, grad=True)), hidden.grad)
        torch.testing.assert_close(torch.from_numpy(result.value(fc1_out)), hidden.detach())

    def test_parameter_update_records_match_the_optimizer_step(self, capture):
        (x, y), *_ = models.make_batches(1)
        model = fresh_model()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

        result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

        params = {p["name"]: p for p in result.steps[0]["params"]}
        assert set(params) == set(before)
        for name, p in model.named_parameters():
            record = params[name]
            weight, update = result.value(record["weight"]), result.value(record["update"])
            torch.testing.assert_close(torch.from_numpy(weight), before[name])
            torch.testing.assert_close(torch.from_numpy(update), p.detach() - before[name])
            if record["weight_norm"] == 0:
                assert record["update_ratio"] is None  # e.g. zero-initialised bias: ratio undefined
            else:
                assert record["update_ratio"] == record["update_norm"] / record["weight_norm"]

    def test_sgd_update_equals_minus_lr_times_recorded_gradient(self, capture):
        (x, y), *_ = models.make_batches(1)
        model = fresh_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

        for record in result.steps[0]["params"]:
            grad, update = result.value(record["grad"]), result.value(record["update"])
            torch.testing.assert_close(torch.from_numpy(update), -0.1 * torch.from_numpy(grad))
