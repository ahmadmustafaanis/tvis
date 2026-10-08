"""Data captured for the insight pages: Grad-CAM, attention weights, per-element loss."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.capture import models


def conv_model(seed: int = 0, **kwargs) -> models.SmallConvNet:
    torch.manual_seed(seed)
    return models.SmallConvNet(**kwargs)


def reference_gradcam(model, x, conv_name: str, classes: torch.Tensor) -> torch.Tensor:
    """Grad-CAM computed the textbook way on an independent copy of the model."""
    activations = {}
    handle = getattr(model, conv_name).register_forward_hook(lambda m, i, o: activations.update(a=o))
    logits = model(x)
    handle.remove()
    act = activations["a"]
    (grad,) = torch.autograd.grad(logits.gather(1, classes[:, None]).sum(), act)
    cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1))
    return cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)


class TestGradCam:
    def test_cam_matches_a_textbook_computation_for_predicted_and_target_class(self, capture):
        (x, y), *_ = models.make_image_batches(1)
        model = conv_model()
        reference = copy.deepcopy(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

        with torch.no_grad():
            predicted = reference(x).argmax(1)
        for conv in ("conv1", "conv2"):
            extra = result.call(kind="module", name=conv)["extra"]
            for which, classes in (("pred", predicted), ("target", y)):
                got = torch.from_numpy(result.value(extra["gradcam"][which]))
                assert got.shape == (4, 8, 8)
                torch.testing.assert_close(got, reference_gradcam(copy.deepcopy(reference), x, conv, classes))

    def test_training_is_bit_identical_with_the_extra_gradient_pass(self, capture):
        batches = models.make_image_batches(3)
        ref_model = conv_model(dropout=0.3)
        ref_opt = torch.optim.Adam(ref_model.parameters(), lr=0.01)
        ref_losses: list[torch.Tensor] = []
        torch.manual_seed(5)
        models.train(ref_model, ref_opt, batches, losses=ref_losses)

        model = conv_model(dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        losses: list[torch.Tensor] = []
        torch.manual_seed(5)
        result = capture(lambda: models.train(model, opt, batches, losses=losses), steps=3)

        assert result.call(kind="module", name="conv1")["extra"]["gradcam"]
        assert all(torch.equal(a, b) for a, b in zip(losses, ref_losses, strict=True))
        for (name, got), (_, want) in zip(
            model.named_parameters(), ref_model.named_parameters(), strict=True
        ):
            assert torch.equal(got, want), name

    def test_recorded_activation_gradients_are_loss_gradients_not_class_score_gradients(self, capture):
        (x, y), *_ = models.make_image_batches(1)
        model = conv_model()
        reference = copy.deepcopy(model)
        acts = {}

        def keep(_module, _inputs, output):
            output.retain_grad()
            acts["a"] = output

        reference.conv2.register_forward_hook(keep)
        F.cross_entropy(reference(x), y).backward()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

        out = result.output_ref(result.call(kind="module", name="conv2"))
        torch.testing.assert_close(torch.from_numpy(result.value(out, grad=True)), acts["a"].grad)

    def test_can_be_disabled(self, capture):
        model = conv_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        result = capture(
            lambda: models.train(model, opt, models.make_image_batches(1)), steps=1, gradcam=False
        )

        assert "gradcam" not in result.call(kind="module", name="conv1").get("extra", {})

    def test_models_without_conv_layers_get_no_gradcam(self, capture):
        model = models.ToyMLP()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        result = capture(lambda: models.train(model, opt, models.make_batches(1)), steps=1)

        assert not any("gradcam" in c.get("extra", {}) for c in result.batch(0)["calls"])


def test_attention_weights_from_a_project_softmax_are_stored(capture):
    model = models.AttnModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 5, 4)

    def loop():
        model(x).sum().backward()
        opt.step()

    result = capture(loop, steps=1)

    (softmax,) = [o for o in result.batch(0)["ops"] if o["op"] == "softmax"]
    assert softmax["attention"] is True
    with torch.no_grad():
        expected = (model.q(x) @ model.k(x).transpose(-2, -1)).softmax(-1)
    torch.testing.assert_close(
        torch.from_numpy(result.value(softmax["value"])), expected, rtol=1e-5, atol=1e-6
    )


def test_class_probability_softmax_is_not_mistaken_for_attention(capture):
    model = models.ToyMLP()

    def loop():
        model(torch.randn(4, 6)).softmax(-1).sum().backward()

    result = capture(loop, steps=1)

    assert not any(o.get("attention") for o in result.batch(0)["ops"])


def test_unreduced_loss_is_stored_per_element(capture):
    model = models.ToyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    (x, y), *_ = models.make_batches(1)
    expected = F.cross_entropy(copy.deepcopy(model)(x), y, reduction="none")

    result = capture(lambda: models.train(model, opt, [(x, y)]), steps=1)

    loss = result.call(kind="loss", name="cross_entropy")
    torch.testing.assert_close(
        torch.from_numpy(result.value(loss["extra"]["per_element"])), expected.detach()
    )


def test_tokenizer_vocabulary_is_recorded_for_labelling(capture):
    pytest.importorskip("transformers")
    from tests.capture import datasets
    from tests.capture.test_data import TestTextPipeline

    tokenizer = datasets.build_tokenizer()
    result = capture(TestTextPipeline().lm_loop(tokenizer))

    vocab = result.meta["data"]["vocab"]
    assert len(vocab) == len(tokenizer)
    assert vocab[tokenizer.convert_tokens_to_ids("cat")] == "cat"
