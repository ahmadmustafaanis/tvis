"""Tensor statistics computed on the tensor's own device with a single device→host transfer."""

from __future__ import annotations

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

    finite = torch.isfinite(x)
    n_finite = finite.sum()
    nan = torch.isnan(x).sum()
    inf = torch.isinf(x).sum()
    zeros = (x == 0).sum()
    xf = x[finite]
    head = torch.stack([n_finite.to(torch.float32), nan.float(), inf.float(), zeros.float()]).tolist()
    n_finite_i, nan_i, inf_i, zeros_i = (int(v) for v in head)
    stats.update(nan=nan_i, inf=inf_i, zero_frac=zeros_i / x.numel())
    if n_finite_i == 0:
        return {**_empty_stats(), **stats}

    lo, hi = xf.min(), xf.max()
    packed = torch.stack(
        [xf.mean(), xf.std(unbiased=False), lo, hi, xf.abs().mean(), torch.linalg.vector_norm(xf)]
    )
    if bool(lo == hi):
        counts = torch.zeros(bins, dtype=torch.float32, device=xf.device)
        counts[0] = n_finite_i
    else:
        counts = torch.histc(xf, bins=bins, min=float(lo), max=float(hi))
    mean, std, mn, mx, abs_mean, norm = packed.tolist()
    stats.update(
        mean=mean,
        std=std,
        min=mn,
        max=mx,
        abs_mean=abs_mean,
        norm=norm,
        hist={"lo": mn, "hi": mx, "counts": [int(c) for c in counts.tolist()]},
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
