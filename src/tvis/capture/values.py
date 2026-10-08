"""Turn arbitrary Python values flowing through calls into small JSON *value refs*.

Tensors (and PIL images) are captured into the registry and referenced by ``tid``; strings,
numbers and containers are described inline with size limits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from tvis.capture.tensors import TensorRegistry, capturable

MAX_DEPTH = 4
MAX_ITEMS = 64
MAX_TEXT = 2000


def capture_value(value: Any, registry: TensorRegistry, *, hook_grad: bool = True, depth: int = 0) -> dict:
    if value is None or isinstance(value, bool):
        return {"kind": "scalar", "value": value}
    if isinstance(value, torch.Tensor):
        if not capturable(value):
            return {"kind": "object", "type": type(value).__name__, "repr": "unsupported tensor layout"}
        record = registry.capture(value, hook_grad=hook_grad)
        return {"kind": "tensor", "tid": record.tid, "batch": record.batch}
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return {"kind": "scalar", "value": None, "repr": repr(value)}
        return {"kind": "scalar", "value": value}
    if isinstance(value, str):
        return {"kind": "text", "text": value[:MAX_TEXT], "truncated": len(value) > MAX_TEXT}
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        record = registry.capture(torch.from_numpy(np.ascontiguousarray(value)), hook_grad=False)
        return {"kind": "tensor", "tid": record.tid, "batch": record.batch, "source": "numpy"}
    if _is_pil_image(value):
        pixels = np.asarray(value.convert("RGB") if value.mode not in ("RGB", "L") else value)
        record = registry.capture_image(value, pixels, value.mode)
        return {"kind": "image", "tid": record.tid, "batch": record.batch, "size": list(value.size)}
    if depth >= MAX_DEPTH:
        return {"kind": "object", "type": type(value).__name__, "repr": _short_repr(value)}
    if isinstance(value, Mapping):
        items = list(value.items())
        return {
            "kind": "map",
            "type": type(value).__name__,
            "items": {
                str(k): capture_value(v, registry, hook_grad=hook_grad, depth=depth + 1)
                for k, v in items[:MAX_ITEMS]
            },
            "len": len(items),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "seq",
            "type": type(value).__name__,
            "items": [
                capture_value(v, registry, hook_grad=hook_grad, depth=depth + 1) for v in value[:MAX_ITEMS]
            ],
            "len": len(value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "map",
            "type": type(value).__name__,
            "items": {
                f.name: capture_value(getattr(value, f.name), registry, hook_grad=hook_grad, depth=depth + 1)
                for f in dataclasses.fields(value)[:MAX_ITEMS]
            },
        }
    return {"kind": "object", "type": type(value).__name__, "repr": _short_repr(value)}


def iter_tensor_refs(ref: dict):
    """Yield every tensor/image ref nested inside a value ref."""
    kind = ref.get("kind")
    if kind in ("tensor", "image"):
        yield ref
    elif kind == "seq":
        for item in ref["items"]:
            yield from iter_tensor_refs(item)
    elif kind == "map":
        for item in ref["items"].values():
            yield from iter_tensor_refs(item)


def _is_pil_image(value: Any) -> bool:
    cls = type(value)
    return any(c.__module__.startswith("PIL.") and c.__name__ == "Image" for c in cls.__mro__)


def _short_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return text[:200]
