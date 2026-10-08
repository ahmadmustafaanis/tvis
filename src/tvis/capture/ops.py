"""Line-level op recorder.

A ``TorchFunctionMode`` sees every torch function / tensor method call. For each call made
*directly from a line of the user's project* (possibly via thin torch Python wrappers such as
``torch.nn.functional``), it records the op name and its output shapes against that file:line, so
the Source view can show the shape produced by every line of the user's functions. Ops executed
inside library modules (``torch.nn.modules``, transformers, …) are not attributed to user lines.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import torch
from torch.overrides import TorchFunctionMode

from tvis.capture.project import is_tvis_file
from tvis.capture.stats import dtype_name
from tvis.capture.values import capture_value

if TYPE_CHECKING:
    from tvis.capture.session import Session

_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep
_TORCH_MODULES_DIR = os.path.join(_TORCH_DIR, "nn", "modules") + os.sep
_RECORDED_PHASES = ("forward", "loss")
_MAX_WALK = 8

_PROJECT, _WRAPPER, _OPAQUE = "project", "wrapper", "opaque"


class OpRecorder(TorchFunctionMode):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._kind: dict[Any, tuple[str, str | None]] = {}
        self._installed = False

    def install(self) -> None:
        self.__enter__()
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self._installed = False
            self.__exit__(None, None, None)

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        result = func(*args, **(kwargs or {}))
        s = self.session
        if (
            s.recording
            and s.batch is not None
            and s.batch.phase in _RECORDED_PHASES
            and len(s.batch.ops) < s.MAX_OPS_PER_BATCH
        ):
            with s.guard("op recorder"):
                self._record(func, result, sys._getframe(1))
        return result

    def _record(self, func: Any, result: Any, frame: Any) -> None:
        site = self._site(frame)
        if site is None:
            return
        outputs = _tensor_outputs(result)
        if not outputs:
            return
        s = self.session
        assert s.batch is not None
        name = _op_name(func)
        record = {
            "call": s.stack[-1].id if s.stack else None,
            "file": site[0],
            "line": site[1],
            "op": name,
            "shapes": [list(t.shape) for t in outputs],
            "dtype": dtype_name(outputs[0].dtype),
        }
        if name == "softmax" and _looks_like_attention(outputs[0]):
            with s.internal():
                record["value"] = capture_value(outputs[0], s.registry, hook_grad=False)
            record["attention"] = True
        s.batch.ops.append(record)

    def _site(self, frame: Any) -> tuple[str, int] | None:
        for _ in range(_MAX_WALK):
            if frame is None:
                return None
            kind, rel = self._classify(frame.f_code)
            if kind == _PROJECT:
                return rel, frame.f_lineno  # type: ignore[return-value]
            if kind == _OPAQUE:
                return None
            frame = frame.f_back
        return None

    def _classify(self, code: Any) -> tuple[str, str | None]:
        cached = self._kind.get(code)
        if cached is not None:
            return cached
        filename = code.co_filename
        if is_tvis_file(filename):
            result = (_WRAPPER, None)
        elif filename.startswith(_TORCH_MODULES_DIR):
            result = (_OPAQUE, None)
        elif filename.startswith(_TORCH_DIR):
            result = (_WRAPPER, None)
        else:
            rel = self.session.project.relpath(filename)
            result = (_PROJECT, rel) if rel is not None else (_OPAQUE, None)
        self._kind[code] = result
        return result


def _looks_like_attention(t: torch.Tensor) -> bool:
    """Softmax weights over keys: [B, H, Tq, Tk] or square [B, T, T]."""
    return t.dim() == 4 or (t.dim() == 3 and t.shape[-1] == t.shape[-2] and t.shape[-1] > 1)


def _tensor_outputs(result: Any) -> list[torch.Tensor]:
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return [r for r in result if isinstance(r, torch.Tensor)][:8]
    return []


def _op_name(func: Any) -> str:
    name = getattr(func, "__name__", None)
    if name == "__get__":  # property access such as `x.T`
        owner = getattr(func, "__self__", None)
        return getattr(owner, "__name__", "attr")
    return name or repr(func)
