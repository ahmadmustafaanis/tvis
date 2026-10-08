from __future__ import annotations

import base64

import numpy as np
import pytest

from tvis.viewer import views


def decode_f32(view: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(view["data"]), dtype="<f4").reshape(view["rows"], view["cols"])


def decode_rgb(view: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(view["data"]), dtype=np.uint8).reshape(
        view["height"], view["width"], 3
    )


class TestTensorView:
    def test_last_two_dims_form_the_window_and_leading_dims_are_fixed(self):
        array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

        view = views.tensor_view(array, fixed=[1])

        assert view["leading"] == [2] and view["fixed"] == [1]
        np.testing.assert_array_equal(decode_f32(view), array[1])

    def test_missing_fixed_indices_default_to_zero(self):
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)

        view = views.tensor_view(array)

        assert view["fixed"] == [0, 0]
        np.testing.assert_array_equal(decode_f32(view), array[0, 0])

    def test_one_dimensional_tensor_is_a_single_row(self):
        view = views.tensor_view(np.array([1.0, 2.0, 3.0]))

        assert (view["rows"], view["cols"]) == (1, 3)
        assert view["shape"] == [3]

    def test_scalar_is_a_one_by_one_grid(self):
        view = views.tensor_view(np.float32(4.5))

        assert (view["rows"], view["cols"]) == (1, 1)
        assert decode_f32(view)[0, 0] == 4.5

    def test_large_windows_are_strided_down(self):
        array = np.arange(1000 * 600, dtype=np.float32).reshape(1000, 600)

        view = views.tensor_view(array, max_side=256)

        assert view["stride"] == [4, 3]
        assert view["rows"] <= 256 and view["cols"] <= 256
        np.testing.assert_array_equal(decode_f32(view), array[::4, ::3])

    def test_min_max_ignore_non_finite_values(self):
        view = views.tensor_view(np.array([[np.nan, 1.0], [np.inf, -2.0]], np.float32))

        assert (view["min"], view["max"]) == (-2.0, 1.0)

    def test_out_of_range_index_is_rejected(self):
        with pytest.raises(views.ViewError):
            views.tensor_view(np.zeros((2, 3, 4)), fixed=[2])


class TestSelectSample:
    def test_exact_batch_dimension_selects_one_row(self):
        array = np.arange(12).reshape(3, 4)
        meta = {"batch_dim": {"dim": 0, "block": 1}, "stored": "full"}

        np.testing.assert_array_equal(views.select_sample(array, meta, 1), array[1])

    def test_merged_batch_dimension_selects_the_samples_block(self):
        array = np.arange(6 * 2).reshape(6, 2)  # [B*T, V] with B=2, T=3
        meta = {"batch_dim": {"dim": 0, "block": 3}, "stored": "full"}

        np.testing.assert_array_equal(views.select_sample(array, meta, 1), array[3:6])

    def test_samples_beyond_the_stored_rows_are_reported_not_misread(self):
        meta = {"batch_dim": {"dim": 0, "block": 1}, "stored": "rows", "stored_rows": 1}

        with pytest.raises(views.ViewError, match="not stored"):
            views.select_sample(np.ones((1, 5)), meta, 3)

    def test_gradient_storage_limits_are_checked_separately(self):
        meta = {
            "batch_dim": {"dim": 0, "block": 1},
            "stored": "full",
            "grad_stored": "rows",
            "grad_stored_rows": 1,
        }

        with pytest.raises(views.ViewError):
            views.select_sample(np.ones((1, 5)), meta, 1, grad=True)

    def test_tensor_without_batch_dimension_cannot_be_sliced_by_sample(self):
        with pytest.raises(views.ViewError):
            views.select_sample(np.ones((3, 3)), {}, 0)


class TestImageView:
    def test_normalised_tensor_is_un_normalised_before_display(self):
        chw = np.zeros((3, 2, 2), np.float32)  # value 0 after Normalize(mean=.5, std=.25) == 0.5 raw

        view = views.image_view(chw, kind="tensor", normalize={"mean": [0.5] * 3, "std": [0.25] * 3})

        assert (decode_rgb(view) == 128).all()

    def test_tensor_without_known_normalisation_is_stretched_to_the_value_range(self):
        chw = np.stack([np.array([[-1.0, 1.0]], np.float32)] * 3)

        pixels = decode_rgb(views.image_view(chw, kind="tensor"))

        assert pixels[0, 0].tolist() == [0, 0, 0] and pixels[0, 1].tolist() == [255, 255, 255]

    def test_single_channel_tensors_render_as_grey(self):
        pixels = decode_rgb(views.image_view(np.full((1, 2, 2), 1.0, np.float32), kind="tensor"))

        assert pixels.shape == (2, 2, 3) and (pixels == 255).all()

    def test_gradients_render_as_magnitude(self):
        grad = np.zeros((3, 1, 2), np.float32)
        grad[1, 0, 1] = -4.0

        pixels = decode_rgb(views.image_view(grad, kind="tensor", grad=True))

        assert pixels[0, 0].tolist() == [0, 0, 0] and pixels[0, 1].tolist() == [255, 255, 255]

    def test_captured_pil_pixels_pass_through_unchanged(self):
        hwc = np.random.default_rng(0).integers(0, 255, (3, 5, 3), dtype=np.uint8)

        np.testing.assert_array_equal(decode_rgb(views.image_view(hwc, kind="image")), hwc)

    def test_non_image_shapes_are_rejected(self):
        with pytest.raises(views.ViewError):
            views.image_view(np.zeros((5, 4, 4)), kind="tensor")
