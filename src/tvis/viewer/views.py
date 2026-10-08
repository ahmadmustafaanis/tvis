"""Turn stored arrays into what the UI draws: 2-D slices of tensors and RGB images.

numpy only. Arrays travel as base64 little-endian float32 / uint8 inside JSON.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

MAX_SIDE = 256


class ViewError(ValueError):
    pass


def select_sample(array: np.ndarray, meta: dict[str, Any], sample: int, *, grad: bool = False) -> np.ndarray:
    """Slice out one sample along the tensor's batch dimension (see ``batch_dim`` in the meta)."""
    batch_dim = meta.get("batch_dim")
    if not batch_dim:
        raise ViewError("tensor has no batch dimension")
    block = int(batch_dim.get("block", 1))
    stored = meta.get("grad_stored" if grad else "stored")
    rows = meta.get("grad_stored_rows" if grad else "stored_rows")
    start, stop = sample * block, (sample + 1) * block
    if stored == "rows" and rows is not None and stop > rows:
        raise ViewError(f"sample {sample} was not stored (only the first {rows} rows were kept)")
    if stop > array.shape[0]:
        raise ViewError(f"sample {sample} out of range")
    return array[start] if block == 1 else array[start:stop]


def tensor_view(
    array: np.ndarray,
    *,
    fixed: list[int] | None = None,
    max_side: int = MAX_SIDE,
) -> dict[str, Any]:
    """A 2-D window of ``array``: the last two dims, with every leading dim fixed to an index.

    1-D tensors become one row; scalars a 1x1 grid. Large windows are strided down to ``max_side``.
    """
    array = np.asarray(array)
    full_shape = list(array.shape)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    leading = list(array.shape[:-2])
    fixed = list(fixed or [])[: len(leading)]
    fixed += [0] * (len(leading) - len(fixed))
    for i, (index, size) in enumerate(zip(fixed, leading, strict=True)):
        if not 0 <= index < size:
            raise ViewError(f"index {index} out of range for dim {i} of size {size}")
    window = array[tuple(fixed)] if leading else array
    step_r = max(1, -(-window.shape[0] // max_side))
    step_c = max(1, -(-window.shape[1] // max_side))
    window = np.asarray(window[::step_r, ::step_c], dtype=np.float32)
    finite = window[np.isfinite(window)]
    return {
        "shape": full_shape,
        "leading": leading,
        "fixed": fixed,
        "rows": int(window.shape[0]),
        "cols": int(window.shape[1]),
        "stride": [step_r, step_c],
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "data": base64.b64encode(np.ascontiguousarray(window, dtype="<f4").tobytes()).decode("ascii"),
    }


def image_view(
    array: np.ndarray,
    *,
    kind: str,
    normalize: dict[str, list[float]] | None = None,
    grad: bool = False,
) -> dict[str, Any]:
    """Render a captured image or an image-shaped tensor (C,H,W with C in {1,3}) as RGB bytes.

    Values tensors are un-normalised with the pipeline's Normalize mean/std when given; gradients
    are shown as per-pixel magnitude, scaled to the maximum.
    """
    array = np.asarray(array)
    if kind == "image":
        pixels = array if array.ndim == 3 else np.repeat(array[..., None], 3, axis=2)
        return _rgb(pixels.astype(np.uint8))
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ViewError(f"not an image tensor: shape {list(array.shape)}")
    chw = array.astype(np.float32)
    if grad:
        magnitude = np.abs(chw).max(axis=0)
        peak = float(magnitude.max()) or 1.0
        gray = (magnitude / peak * 255).astype(np.uint8)
        return _rgb(np.repeat(gray[..., None], 3, axis=2))
    if normalize and len(normalize.get("mean", [])) == chw.shape[0]:
        mean = np.asarray(normalize["mean"], np.float32)[:, None, None]
        std = np.asarray(normalize["std"], np.float32)[:, None, None]
        chw = chw * std + mean
    elif chw.min() < 0 or chw.max() > 1:  # unknown scaling: stretch to the value range
        lo, hi = float(np.nanmin(chw)), float(np.nanmax(chw))
        chw = (chw - lo) / ((hi - lo) or 1.0)
    hwc = np.clip(np.nan_to_num(chw.transpose(1, 2, 0)), 0.0, 1.0)
    if hwc.shape[2] == 1:
        hwc = np.repeat(hwc, 3, axis=2)
    return _rgb((hwc * 255 + 0.5).astype(np.uint8))


def _rgb(pixels: np.ndarray) -> dict[str, Any]:
    h, w = pixels.shape[:2]
    return {
        "height": int(h),
        "width": int(w),
        "data": base64.b64encode(np.ascontiguousarray(pixels[..., :3]).tobytes()).decode("ascii"),
    }


MAX_ARRAY_ELEMS = 2_000_000


def full_array(array: np.ndarray) -> dict[str, Any]:
    """All values (float32), for small tensors the UI lays out itself (conv kernels, CAMs)."""
    array = np.asarray(array)
    if array.size > MAX_ARRAY_ELEMS:
        raise ViewError(f"tensor too large to send whole ({array.size} > {MAX_ARRAY_ELEMS} elements)")
    values = np.ascontiguousarray(array, dtype="<f4")
    finite = values[np.isfinite(values)]
    return {
        "shape": list(array.shape),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def thumbnail(array: np.ndarray, size: int = 48) -> dict[str, Any]:
    """A small 2-D summary of one sample's activation: channel-wise mean |x| for [C, H, W] (and
    any higher-rank tensor), the tensor itself for 2-D, one row for 1-D."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim >= 3:
        array = np.abs(array).mean(axis=tuple(range(array.ndim - 2)))
    return {
        "reduced": "mean |x| over leading dims" if np.asarray(array).ndim == 2 else None,
        **tensor_view(array, max_side=size),
    }


def pca2(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project rows onto their first two principal components; returns (coords [N, 2], variance ratios)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ViewError("need at least two vectors for a projection")
    x = np.nan_to_num(x - x.mean(axis=0, keepdims=True))
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(2, vt.shape[0])
    coords = x @ vt[:k].T
    if k < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - k)))
    variance = s**2
    total = variance.sum() or 1.0
    return coords, [float(v / total) for v in variance[:2]] + [0.0] * (2 - k)


def pool_features(array: np.ndarray) -> np.ndarray:
    """One vector per sample: spatial mean for [B, C, H, W], sequence mean for [B, T, D]."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        return array.mean(axis=(2, 3))
    if array.ndim == 3:
        return array.mean(axis=1)
    if array.ndim == 2:
        return array
    if array.ndim > 4:
        return array.reshape(array.shape[0], array.shape[1], -1).mean(axis=2)
    raise ViewError(f"cannot pool a tensor of shape {list(array.shape)} into per-sample vectors")
