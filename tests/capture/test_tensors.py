from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from tvis.capture.tensors import TensorRegistry, to_numpy
from tvis.capture.timing import Timer
from tvis.capture.values import capture_value, iter_tensor_refs


@pytest.fixture
def registry() -> TensorRegistry:
    timer = Timer()
    return TensorRegistry(
        max_elems=1000,
        mark=timer.mark,
        internal=contextlib.nullcontext,
        guard=lambda _: contextlib.nullcontext(),
    )


class TestIdentity:
    def test_same_tensor_object_is_recorded_once(self, registry: TensorRegistry):
        x = torch.randn(3)

        assert registry.capture(x).tid == registry.capture(x).tid

    def test_in_place_modification_creates_a_new_value(self, registry: TensorRegistry):
        x = torch.zeros(3)
        before = registry.capture(x)
        x.add_(1)
        after = registry.capture(x)

        assert before.tid != after.tid
        assert torch.equal(before.value, torch.zeros(3))
        assert torch.equal(after.value, torch.ones(3))

    def test_captured_value_is_a_copy_not_a_view(self, registry: TensorRegistry):
        x = torch.zeros(3)
        record = registry.capture(x)
        x[0] = 5.0

        assert record.value[0] == 0.0

    def test_inference_mode_tensors_can_be_captured(self, registry: TensorRegistry):
        with torch.inference_mode():
            x = torch.ones(2)
        assert registry.capture(x).stats["mean"] == 1.0


class TestGradients:
    def test_gradient_hook_records_the_gradient_autograd_computes(self, registry: TensorRegistry):
        w = torch.randn(4, requires_grad=True)
        h = w * 3
        record = registry.capture(h)
        (h**2).sum().backward()

        torch.testing.assert_close(record.grad, 2 * h.detach())
        assert record.grad_stats["norm"] == pytest.approx(float((2 * h.detach()).norm()), rel=1e-6)
        assert record.grad_arrival is not None

    def test_gradient_hook_leaves_the_gradient_unchanged(self, registry: TensorRegistry):
        torch.manual_seed(0)
        w = torch.randn(5, requires_grad=True)
        (torch.sin(w) * 2).sum().backward()
        expected = w.grad.clone()

        w.grad = None
        h = torch.sin(w)
        registry.capture(h)
        (h * 2).sum().backward()

        assert torch.equal(w.grad, expected)

    def test_hook_registered_before_in_place_change_sees_the_pre_change_gradient(
        self, registry: TensorRegistry
    ):
        w = torch.ones(3, requires_grad=True)
        h = w * 2
        record = registry.capture(h)
        h.mul_(5)  # in-place after capture
        h.sum().backward()

        torch.testing.assert_close(record.grad, torch.full((3,), 5.0))

    def test_tensors_without_grad_get_no_hook(self, registry: TensorRegistry):
        record = registry.capture(torch.ones(2))

        assert record.requires_grad is False and record.grad_stats is None


class TestStorageLimits:
    def test_small_tensors_are_stored_in_full(self, registry: TensorRegistry):
        record = registry.capture(torch.arange(10.0))

        assert record.stored == "full"

    def test_large_tensors_keep_leading_rows_that_fit(self, registry: TensorRegistry):
        x = torch.arange(3000.0).reshape(30, 100)  # 3000 elems > max 1000
        record = registry.capture(x)

        assert record.stored == "rows" and record.stored_rows == 10
        assert torch.equal(record.value, x[:10])
        assert record.stats["numel"] == 3000  # stats always describe the whole tensor

    def test_tensors_whose_single_row_is_too_large_are_not_stored(self, registry: TensorRegistry):
        record = registry.capture(torch.zeros(2, 2000))

        assert record.stored == "none" and record.value is None
        assert record.stats["numel"] == 4000


class TestBatches:
    def test_take_batch_returns_only_that_batch_and_releases_references(self, registry: TensorRegistry):
        registry.batch_index = 0
        a = registry.capture(torch.ones(1))
        registry.batch_index = 1
        b = registry.capture(torch.zeros(1))

        taken = registry.take_batch(0)

        assert [r.tid for r in taken] == [a.tid]
        assert list(registry.records) == [b.tid]


def test_to_numpy_converts_bfloat16_to_float32():
    array = to_numpy(torch.tensor([1.5], dtype=torch.bfloat16))

    assert array.dtype == np.float32 and array[0] == 1.5


class TestValueRefs:
    def test_nested_containers_reference_each_tensor(self, registry: TensorRegistry):
        x, y = torch.ones(2), torch.zeros(3)
        ref = capture_value({"pair": (x, y), "n": 3, "name": "batch"}, registry)

        tids = [r["tid"] for r in iter_tensor_refs(ref)]
        assert len(tids) == 2
        assert ref["items"]["n"] == {"kind": "scalar", "value": 3}
        assert ref["items"]["name"]["text"] == "batch"

    def test_long_text_is_truncated_and_flagged(self, registry: TensorRegistry):
        ref = capture_value("x" * 5000, registry)

        assert ref["truncated"] is True and len(ref["text"]) == 2000

    def test_non_finite_python_floats_are_json_safe(self, registry: TensorRegistry):
        assert capture_value(float("nan"), registry)["value"] is None

    def test_pil_images_are_captured_as_pixels_once_per_image(self, registry: TensorRegistry):
        Image = pytest.importorskip("PIL.Image")
        image = Image.new("RGB", (4, 3), (10, 20, 30))

        first = capture_value(image, registry)
        second = capture_value(image, registry)

        assert first["kind"] == "image" and first["tid"] == second["tid"]
        record = registry.records[first["tid"]]
        assert record.shape == [3, 4, 3]  # H, W, C
        assert record.value[0, 0].tolist() == [10, 20, 30]

    def test_unknown_objects_are_described_by_type_and_repr(self, registry: TensorRegistry):
        class Opaque:
            def __repr__(self) -> str:
                return "Opaque()"

        ref = capture_value(Opaque(), registry)

        assert ref == {"kind": "object", "type": "Opaque", "repr": "Opaque()"}

    def test_depth_limit_stops_recursion(self, registry: TensorRegistry):
        nested: list = []
        cursor = nested
        for _ in range(10):
            cursor.append([])
            cursor = cursor[0]

        ref = capture_value(nested, registry)
        depth = 0
        while ref["kind"] == "seq":
            ref = ref["items"][0]
            depth += 1
        assert ref["kind"] == "object" and depth == 4
