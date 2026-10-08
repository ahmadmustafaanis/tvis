"""Tensor registry: gives every distinct tensor *value* seen during a batch a stable id, records its
statistics, keeps a copy of its value, and attaches a gradient hook to capture dL/d(tensor).

A tensor value is identified by ``(id(tensor), tensor._version)``: the output of layer *i* that is
the input of layer *i+1* is stored once, and an in-place modification creates a new value.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tvis.capture.stats import dtype_name, tensor_stats
from tvis.capture.timing import Mark

_NUMPY_UNSUPPORTED = {torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2} | {
    getattr(torch, n) for n in ("uint16", "uint32", "uint64") if hasattr(torch, n)
}


@dataclass(slots=True)
class TensorRecord:
    tid: str
    batch: int
    shape: list[int]
    dtype: str
    device: str
    requires_grad: bool
    kind: str = "tensor"  # "tensor" | "image" (uint8 HWC pixels from a PIL image)
    stats: dict[str, Any] | None = None
    stored: str = "none"  # "full" | "rows" (first `stored_rows` along dim 0) | "none"
    stored_rows: int | None = None
    grad_stats: dict[str, Any] | None = None
    grad_stored: str = "none"
    grad_stored_rows: int | None = None
    grad_arrival: Mark | None = None
    batch_dim: dict[str, Any] | None = None
    phase: str | None = None
    normalized: bool = False  # produced by a Normalize transform (the viewer un-normalises it)
    value: torch.Tensor | None = field(default=None, repr=False)
    grad: torch.Tensor | None = field(default=None, repr=False)

    def to_json(self, ms: Callable[[Mark], float]) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "tid": self.tid,
            "batch": self.batch,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "stats": self.stats,
            "stored": self.stored,
            "phase": self.phase,
        }
        if self.stored_rows is not None:
            doc["stored_rows"] = self.stored_rows
        if self.grad_stats is not None:
            doc["grad_stats"] = self.grad_stats
            doc["grad_stored"] = self.grad_stored
            if self.grad_stored_rows is not None:
                doc["grad_stored_rows"] = self.grad_stored_rows
        if self.grad_arrival is not None:
            doc["grad_arrival_ms"] = ms(self.grad_arrival)
        if self.batch_dim is not None:
            doc["batch_dim"] = self.batch_dim
        if self.normalized:
            doc["normalized"] = True
        return doc


def tensor_version(tensor: torch.Tensor) -> int:
    try:
        return tensor._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return -1


def capturable(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type != "meta"
        and not value.is_nested
    )


class TensorRegistry:
    """Per-run registry. Records are grouped by the batch that captured them."""

    def __init__(
        self,
        *,
        max_elems: int,
        mark: Callable[[], Mark],
        internal: Callable[[], Any],
        guard: Callable[[str], Any],
    ):
        self.max_elems = max_elems
        self._mark = mark
        self._internal = internal
        self._guard = guard
        self._ids = itertools.count()
        self._by_key: dict[tuple[int, int], TensorRecord] = {}
        self._refs: dict[str, Any] = {}  # keeps captured tensors alive so id() stays unique
        self.records: dict[str, TensorRecord] = {}
        self.batch_index = 0
        self.phase: str | None = None

    # -- capture -------------------------------------------------------------------------------
    def capture(self, tensor: torch.Tensor, *, hook_grad: bool = True) -> TensorRecord:
        key = (id(tensor), tensor_version(tensor))
        existing = self._by_key.get(key)
        if existing is not None and self._refs.get(existing.tid) is tensor:
            return existing

        tid = f"t{next(self._ids)}"
        record = TensorRecord(
            tid=tid,
            batch=self.batch_index,
            shape=list(tensor.shape),
            dtype=dtype_name(tensor.dtype),
            device=str(tensor.device),
            requires_grad=bool(tensor.requires_grad),
            phase=self.phase,
        )
        record.stats = tensor_stats(tensor)
        record.value, record.stored, record.stored_rows = self._keep(tensor)
        self._by_key[key] = record
        self._refs[tid] = tensor
        self.records[tid] = record
        if hook_grad and tensor.requires_grad:
            tensor.register_hook(self._make_grad_hook(record))
        return record

    def capture_image(self, owner: Any, pixels: np.ndarray, mode: str) -> TensorRecord:
        """Capture HWC/HW uint8 pixels of a PIL image, deduplicated by the image object."""
        key = (id(owner), -2)
        existing = self._by_key.get(key)
        if existing is not None and self._refs.get(existing.tid) is owner:
            return existing
        record = self.capture(torch.from_numpy(np.array(pixels, copy=True)), hook_grad=False)
        record.kind = "image"
        record.dtype = f"{record.dtype}:{mode}"
        self._by_key[key] = record
        self._refs[record.tid] = owner
        return record

    def _keep(self, tensor: torch.Tensor) -> tuple[torch.Tensor | None, str, int | None]:
        t = tensor.detach()
        if t.numel() <= self.max_elems:
            return t.clone(), "full", None
        if t.dim() >= 1 and t.shape[0] > 0:
            per_row = max(1, t[0].numel())
            rows = min(t.shape[0], self.max_elems // per_row)
            if rows >= 1:
                return t[:rows].clone(), "rows", rows
        return None, "none", None

    def _make_grad_hook(self, record: TensorRecord) -> Callable[[torch.Tensor], None]:
        def hook(grad: torch.Tensor) -> None:
            arrival = self._mark()
            with self._guard("grad hook"), self._internal():
                if record.grad_arrival is None:
                    record.grad_arrival = arrival
                record.grad_stats = tensor_stats(grad)
                record.grad, record.grad_stored, record.grad_stored_rows = self._keep(grad)
            # returning None leaves the gradient untouched

        return hook

    # -- batch lifecycle -----------------------------------------------------------------------
    def take_batch(self, batch_index: int) -> list[TensorRecord]:
        """Detach and return the records captured for ``batch_index`` (their values stay attached)."""
        taken = [r for r in self.records.values() if r.batch == batch_index]
        for r in taken:
            del self.records[r.tid]
            self._refs.pop(r.tid, None)
        live = set(self._refs)
        self._by_key = {k: v for k, v in self._by_key.items() if v.tid in live}
        return taken


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach()
    if t.is_complex():
        t = t.abs()
    if t.dtype in _NUMPY_UNSUPPORTED:
        t = t.to(torch.float32)
    return t.cpu().numpy()
