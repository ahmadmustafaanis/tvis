"""Call records: one per module call, project-function call, data/pipeline stage, loss, backward,
or optimizer step. Calls form a tree via ``parent``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tvis.capture.timing import Mark, Timer

# kinds
MODULE = "module"
FUNCTION = "function"
DATA = "data"  # DataLoader fetch
SAMPLE = "sample"  # dataset[i]
PIPELINE = "pipeline"  # torchvision Compose
TRANSFORM = "transform"  # one transform stage
TOKENIZE = "tokenize"
LOSS = "loss"
BACKWARD = "backward"
OPTIMIZER = "optimizer"


@dataclass(slots=True, eq=False)
class Call:
    id: int
    parent: int | None
    kind: str
    name: str
    phase: str
    batch: int
    depth: int
    qualname: str | None = None
    module_path: str | None = None
    cls: str | None = None
    file: str | None = None  # project-relative path of the definition (None for library code)
    line: int | None = None
    call_site: dict[str, Any] | None = None  # {"file", "line"} of the project line that made the call
    inputs: list[dict[str, Any]] = field(default_factory=list)  # [{"name", "value": ValueRef}]
    outputs: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    start: Mark | None = None
    end: Mark | None = None
    frame_id: int | None = None
    module_ref: Any = None  # the nn.Module object for module calls (not serialised)
    truncated: bool = False

    def to_json(self, timer: Timer, tensor_arrivals: dict[str, Mark], param_arrivals: list[Mark]) -> dict:
        """``ms``: device time of the call itself; ``bwd_ms``: its share of the backward pass."""
        doc: dict[str, Any] = {
            "id": self.id,
            "parent": self.parent,
            "kind": self.kind,
            "name": self.name,
            "phase": self.phase,
            "depth": self.depth,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        for key in ("qualname", "module_path", "cls", "file", "line", "call_site"):
            value = getattr(self, key)
            if value is not None:
                doc[key] = value
        if self.extra:
            doc["extra"] = self.extra
        if self.flags:
            doc["flags"] = self.flags
        if self.truncated:
            doc["truncated"] = True
        if self.start is not None and self.end is not None:
            doc["start_ms"] = timer.ms(self.start)
            doc["end_ms"] = timer.ms(self.end)
            doc["ms"] = timer.duration(self.start, self.end)
            doc["wall_ms"] = (self.end.wall - self.start.wall) * 1000.0
        bwd = self._backward_window(tensor_arrivals, param_arrivals)
        if bwd is not None:
            start, end = bwd
            doc["bwd_start_ms"] = timer.ms(start)
            doc["bwd_end_ms"] = timer.ms(end)
            doc["bwd_ms"] = timer.duration(start, end)
            doc["bwd_approx"] = True
        return doc

    def _backward_window(
        self, tensor_arrivals: dict[str, Mark], param_arrivals: list[Mark]
    ) -> tuple[Mark, Mark] | None:
        """Backward of a call starts when dL/d(output) arrives and ends when the gradients it
        produces arrive: the accumulated grads of the parameters it used (its own, or those of the
        modules it called) if any, else dL/d(input)."""
        out_marks = [tensor_arrivals[t] for t in _tids(self.outputs) if t in tensor_arrivals]
        if not out_marks:
            return None
        start = min(out_marks, key=lambda m: m.seq)
        end_marks = [m for m in param_arrivals if m.seq > start.seq]
        if not end_marks:
            end_marks = [
                tensor_arrivals[t]
                for t in _tids(self.inputs)
                if t in tensor_arrivals and tensor_arrivals[t].seq > start.seq
            ]
        if not end_marks:
            return None
        return start, max(end_marks, key=lambda m: m.seq)


def _tids(values: list[dict[str, Any]]) -> list[str]:
    from tvis.capture.values import iter_tensor_refs

    return [ref["tid"] for v in values for ref in iter_tensor_refs(v["value"]) if ref["kind"] == "tensor"]
