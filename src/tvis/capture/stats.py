"""Tensor statistics computed on the tensor's own device with a single device→host transfer."""

from __future__ import annotations

import math
from typing import Any

import torch

HIST_BINS = 32


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_stats(tensor: torch.Tensor, bins: int = HIST_BINS) -> dict[str, Any]:
    """Summary statistics over all elements.

    Non-finite values are counted (``nan``, ``inf``) and excluded from every other statistic, so the
    result is always JSON-safe. Statistics are ``None`` when there are no finite elements.
    """
    x = tensor.detach()
    stats: dict[str, Any] = {"numel": x.numel()}
    if x.numel() == 0:
        return {**stats, **_empty_stats()}
    if x.is_complex():
        x = x.abs()
    x = x.reshape(-1).to(torch.float32)
    numel = x.numel()

    # NaN and ±Inf propagate through a mean, so a finite mean proves every element is finite: the
    # moments are then taken over `x` itself, skipping the isfinite/isnan/isinf passes and the
    # finite-only copy. Activations and gradients are almost always finite, and this path costs
    # one device→host transfer fewer.
    zeros = numel - torch.count_nonzero(x)
    head = torch.cat([_moments(x), zeros.to(torch.float32).reshape(1)]).tolist()
    if math.isfinite(head[0]):
        stats.update(nan=0, inf=0, zero_frac=int(head[-1]) / numel)
        return _with_moments(stats, x, head[:-1], numel, bins)

    finite = torch.isfinite(x)
    counts = torch.stack(
        [finite.sum().to(torch.float32), torch.isnan(x).sum().float(), torch.isinf(x).sum().float()]
    ).tolist()
    n_finite, nan, inf = (int(v) for v in counts)
    stats.update(nan=nan, inf=inf, zero_frac=int(head[-1]) / numel)
    if n_finite == 0:
        return {**_empty_stats(), **stats}
    xf = x[finite]
    return _with_moments(stats, xf, _moments(xf).tolist(), n_finite, bins)


def _moments(x: torch.Tensor) -> torch.Tensor:
    """[mean, std, min, max, abs_mean, norm] of a 1-D float32 tensor, still on its device."""
    return torch.stack(
        [x.mean(), x.std(unbiased=False), x.min(), x.max(), x.abs().mean(), torch.linalg.vector_norm(x)]
    )


def _with_moments(
    stats: dict[str, Any], xf: torch.Tensor, moments: list[float], n_finite: int, bins: int
) -> dict[str, Any]:
    mean, std, mn, mx, abs_mean, norm = moments
    if mn == mx:
        counts = [0] * bins
        counts[0] = n_finite
    else:
        counts = [int(c) for c in torch.histc(xf, bins=bins, min=mn, max=mx).tolist()]
    stats.update(
        mean=mean,
        std=std,
        min=mn,
        max=mx,
        abs_mean=abs_mean,
        norm=norm,
        hist={"lo": mn, "hi": mx, "counts": counts},
    )
    return stats


def _empty_stats() -> dict[str, Any]:
    return {
        "nan": 0,
        "inf": 0,
        "zero_frac": None,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "abs_mean": None,
        "norm": None,
        "hist": None,
    }
