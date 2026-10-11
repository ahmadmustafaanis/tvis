from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tvis.capture.stats import HIST_BINS, tensor_stats


def test_moments_match_numpy_population_statistics():
    x = torch.tensor([[1.0, -2.0, 3.5], [0.0, 4.0, -1.5]])
    stats = tensor_stats(x)
    ref = x.numpy().ravel()

    assert stats["numel"] == 6
    assert stats["mean"] == pytest.approx(ref.mean())
    assert stats["std"] == pytest.approx(ref.std())  # population std (ddof=0)
    assert stats["min"] == -2.0 and stats["max"] == 4.0
    assert stats["abs_mean"] == pytest.approx(np.abs(ref).mean())
    assert stats["norm"] == pytest.approx(np.linalg.norm(ref))
    assert stats["zero_frac"] == pytest.approx(1 / 6)


def test_non_finite_values_are_counted_and_excluded_from_moments():
    x = torch.tensor([1.0, float("nan"), 3.0, float("inf"), float("-inf")])
    stats = tensor_stats(x)

    assert stats["nan"] == 1
    assert stats["inf"] == 2
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["max"] == 3.0
    assert sum(stats["hist"]["counts"]) == 2


def test_all_non_finite_tensor_yields_null_statistics_that_are_valid_json():
    stats = tensor_stats(torch.full((3,), float("nan")))

    assert stats["nan"] == 3
    assert stats["mean"] is None and stats["hist"] is None
    json.dumps(stats, allow_nan=False)


def test_empty_tensor_is_handled():
    stats = tensor_stats(torch.empty(0, 5))

    assert stats["numel"] == 0
    assert stats["mean"] is None


def test_histogram_covers_all_finite_values_across_the_value_range():
    x = torch.linspace(-1, 1, 1000)
    hist = tensor_stats(x)["hist"]

    assert len(hist["counts"]) == HIST_BINS
    assert sum(hist["counts"]) == 1000
    assert hist["lo"] == -1.0 and hist["hi"] == 1.0
    assert min(hist["counts"]) > 0  # uniform data populates every bin


def test_constant_tensor_puts_everything_in_one_bin():
    hist = tensor_stats(torch.full((10,), 7.0))["hist"]

    assert hist["counts"][0] == 10
    assert sum(hist["counts"]) == 10


@pytest.mark.parametrize(
    ("tensor", "expected_mean"),
    [
        (torch.tensor([1, 2, 3], dtype=torch.int64), 2.0),
        (torch.tensor([True, False, True, True]), 0.75),
        (torch.tensor([1.0, 3.0], dtype=torch.bfloat16), 2.0),
        (torch.tensor([1.0, 3.0], dtype=torch.float16), 2.0),
    ],
    ids=["int64", "bool", "bfloat16", "float16"],
)
def test_non_float32_dtypes_are_summarised(tensor, expected_mean):
    assert tensor_stats(tensor)["mean"] == pytest.approx(expected_mean)


def test_complex_tensors_are_summarised_by_magnitude():
    stats = tensor_stats(torch.tensor([3 + 4j, 0 + 0j]))

    assert stats["max"] == pytest.approx(5.0)


def test_statistics_do_not_modify_or_require_grad_on_the_input():
    x = torch.randn(4, requires_grad=True)
    before = x.detach().clone()

    tensor_stats(x)

    assert torch.equal(x.detach(), before)
    assert x.grad is None


def _reference_stats(tensor: torch.Tensor, bins: int = HIST_BINS) -> dict:
    """The original single-path implementation (finite mask first), kept as the oracle for the
    finite fast path."""
    x = tensor.detach()
    if x.is_complex():
        x = x.abs()
    x = x.reshape(-1).to(torch.float32)
    finite = torch.isfinite(x)
    xf = x[finite]
    stats = {
        "numel": x.numel(),
        "nan": int(torch.isnan(x).sum()),
        "inf": int(torch.isinf(x).sum()),
        "zero_frac": int((x == 0).sum()) / x.numel(),
    }
    lo, hi = float(xf.min()), float(xf.max())
    counts = (
        torch.histc(xf, bins=bins, min=lo, max=hi)
        if lo != hi
        else torch.zeros(bins).index_fill(0, torch.tensor([0]), xf.numel())
    )
    stats.update(
        mean=float(xf.mean()),
        std=float(xf.std(unbiased=False)),
        min=lo,
        max=hi,
        abs_mean=float(xf.abs().mean()),
        norm=float(torch.linalg.vector_norm(xf)),
        hist={"lo": lo, "hi": hi, "counts": [int(c) for c in counts.tolist()]},
    )
    return stats


def _with_non_finite(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().reshape(-1)
    x[3], x[17], x[40] = float("nan"), float("inf"), float("-inf")
    return x


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(16, 8, 12, 12, generator=torch.Generator().manual_seed(0)),
        torch.randn(4, 64, generator=torch.Generator().manual_seed(1)).relu(),
        torch.randn(300, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16),
        torch.randint(-5, 5, (7, 9), generator=torch.Generator().manual_seed(3)),
        torch.full((5, 5), -3.25),
        torch.randn(64, 64, generator=torch.Generator().manual_seed(4)).t(),  # non-contiguous
        _with_non_finite(torch.randn(100, generator=torch.Generator().manual_seed(5))),
    ],
    ids=["activations", "relu-zeros", "bfloat16", "int", "constant", "transposed", "non-finite"],
)
def test_statistics_match_reference_implementation(tensor):
    stats, ref = tensor_stats(tensor), _reference_stats(tensor)

    assert stats.keys() == ref.keys()
    for key in ("numel", "nan", "inf", "zero_frac", "min", "max"):
        assert stats[key] == ref[key], key
    for key in ("mean", "std", "abs_mean", "norm"):
        assert stats[key] == pytest.approx(ref[key], rel=1e-6, abs=1e-7), key
    assert stats["hist"] == ref["hist"]


def test_finite_values_whose_mean_overflows_are_not_reported_as_infinite():
    stats = tensor_stats(torch.full((4,), 3e38))

    assert stats["nan"] == 0 and stats["inf"] == 0
    assert stats["min"] == pytest.approx(3e38) and stats["max"] == pytest.approx(3e38)
