"""Capture session: owns the call stack, batch/step state machine, hooks and patches.

Batch and step boundaries
-------------------------
* A **batch** starts at a DataLoader fetch (unless the current batch is still only fetching data),
  at a top-level forward of a module with parameters after the previous batch's backward, or at
  a backward/optimizer step when no batch is open. Calls outside a batch are not recorded.
* A batch is finalised (written to disk) when the next batch starts, when its step closes, or at
  the end of the run.
* A **step** closes right after ``optimizer.step()`` when the script has exactly one optimizer,
  otherwise lazily when the next batch starts. Without any ``torch.optim`` optimizer, every batch
  with a backward is its own step.
* After ``config.steps`` steps the session finishes and raises :class:`StopCapture` out of the
  hook, unwinding the user's script.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import socket
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tvis.capture import calls as K
from tvis.capture.calls import Call
from tvis.capture.project import ProjectFiles, is_tvis_file
from tvis.capture.tensors import TensorRecord, TensorRegistry, to_numpy
from tvis.capture.timing import Mark, Timer
from tvis.capture.values import capture_value, iter_tensor_refs
from tvis.store import schema
from tvis.store.writer import RunWriter

PHASES = ("data", "forward", "loss", "backward", "post_backward", "optimizer")


class StopCapture(BaseException):
    """Raised out of a hook to end the user's script once the requested steps are captured."""


@dataclass
class CaptureConfig:
    run_dir: Path
    project_dir: Path
    steps: int = 3
    max_batches: int = 64
    max_elems: int = 2_000_000
    trace_functions: bool = True
    record_ops: bool = True
    script: str | None = None
    argv: list[str] = field(default_factory=list)


@dataclass
class Batch:
    index: int
    step: int
    start: Mark
    phase: str = "data"
    phase_marks: list[tuple[str, Mark]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)
    has_forward: bool = False
    has_backward: bool = False
    indices: list[Any] | None = None
    batch_size: int | None = None
    param_arrivals: dict[int, Mark] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    forward_grad: bool | None = None
    forward_models: set[int] = field(default_factory=set)


@dataclass
class Step:
    index: int
    batches: list[int] = field(default_factory=list)
    optimizer_steps: int = 0
    params: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)


class Patcher:
    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any]] = []

    def set(self, owner: Any, name: str, value: Any) -> None:
        self._patches.append((owner, name, owner.__dict__.get(name, getattr(owner, name))))
        setattr(owner, name, value)

    def undo(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)


class Session:
    MAX_OPS_PER_BATCH = 50_000

    def __init__(self, config: CaptureConfig, writer: RunWriter | None = None):
        self.config = config
        self.writer = writer or RunWriter(config.run_dir)
        self.timer = Timer(record_cuda=torch.cuda.is_available())
        self.project = ProjectFiles(config.project_dir, on_new_file=self._snapshot_source)
        self.registry = TensorRegistry(
            max_elems=config.max_elems, mark=self.timer.mark, internal=self.internal, guard=self.guard
        )
        self.patcher = Patcher()
        self._tl = threading.local()
        self._call_ids = itertools.count()
        self._handles: list[Any] = []
        self.stack: list[Call] = []
        self.batch: Batch | None = None
        self.step = Step(index=0)
        self.steps: list[dict[str, Any]] = []
        self.batches_written = 0
        self.active = False
        self.done = False
        self.notices: dict[str, dict[str, Any]] = {}
        self.sources: list[str] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        self.models: list[nn.Module] = []
        self._module_names: dict[int, tuple[int, str]] = {}
        self._param_names: dict[int, str] = {}
        self._hooked_params: set[int] = set()
        self._module_class_src: dict[type, tuple[str | None, int | None]] = {}
        self.data_info: dict[str, Any] = {}
        self.tokenizer: Any = None
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._instrumentations: list[Any] = []
        self._last_batch_had_backward = False

    # ------------------------------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------------------------------
    def install(self) -> None:
        from tvis.capture import data, modules, ops, tracer

        self._write_meta(status=schema.STATUS_RUNNING)
        self._instrumentations = [modules.ModuleHooks(self), data.DataInstrumentation(self)]
        self._patch_backward()
        self._patch_optimizers()
        if self.config.record_ops:
            self._instrumentations.append(ops.OpRecorder(self))
        if self.config.trace_functions:
            self._instrumentations.append(tracer.FunctionTracer(self))
        for inst in self._instrumentations:
            inst.install()
        self.active = True

    def uninstall(self) -> None:
        self.active = False
        for inst in reversed(self._instrumentations):
            with contextlib.suppress(Exception):
                inst.uninstall()
        self._instrumentations = []
        for handle in self._handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self._handles = []
        self.patcher.undo()

    def finish(self, status: str = schema.STATUS_COMPLETE, error: str | None = None) -> None:
        """Flush everything and remove all instrumentation. Idempotent."""
        if self.done:
            return
        self.done = True
        self.uninstall()
        try:
            if self.batch is not None:
                self._finalize_batch()
            if self.step.batches:
                self._close_step(check_stop=False)
            write_errors = self.writer.close()
            for err in write_errors:
                self.notice("write", err)
        finally:
            self._write_meta(status=status, error=error)

    # ------------------------------------------------------------------------------------------
    # internal-work bracketing and error isolation
    # ------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def internal(self) -> Iterator[None]:
        depth = getattr(self._tl, "depth", 0)
        self._tl.depth = depth + 1
        start = self.timer.mark() if depth == 0 else None
        try:
            yield
        finally:
            self._tl.depth = depth
            if start is not None:
                self.timer.add_overhead(start, self.timer.mark())

    def in_internal(self) -> bool:
        return getattr(self._tl, "depth", 0) > 0

    @contextlib.contextmanager
    def guard(self, where: str) -> Iterator[None]:
        try:
            yield
        except Exception as exc:  # StopCapture is a BaseException and passes through
            self.notice(where, f"{type(exc).__name__}: {exc}", detail=traceback.format_exc(limit=6))

    def notice(self, where: str, message: str, detail: str | None = None) -> None:
        key = f"{where}: {message}"
        entry = self.notices.setdefault(key, {"where": where, "message": message, "count": 0})
        entry["count"] += 1
        if detail and "detail" not in entry:
            entry["detail"] = detail

    @property
    def recording(self) -> bool:
        return self.active and not self.done and self.batch is not None and not self.in_internal()

    # ------------------------------------------------------------------------------------------
    # calls
    # ------------------------------------------------------------------------------------------
    def open_call(
        self,
        kind: str,
        name: str,
        *,
        inputs: list[tuple[str, Any]] | None = None,
        hook_grad: bool = True,
        call_site: dict[str, Any] | None = None,
        **attrs: Any,
    ) -> Call | None:
        """Push a call onto the stack and capture its inputs. Returns None when not recording."""
        if not self.recording:
            return None
        batch = self.batch
        assert batch is not None
        with self.internal():
            parent = self.stack[-1] if self.stack else None
            call = Call(
                id=next(self._call_ids),
                parent=parent.id if parent else None,
                kind=kind,
                name=name,
                phase=batch.phase,
                batch=batch.index,
                depth=len(self.stack),
                call_site=call_site,
                **attrs,
            )
            self.registry.phase = batch.phase
            call.inputs = self._capture_values(inputs, hook_grad)
            batch.calls.append(call)
            self.stack.append(call)
        call.start = self.timer.mark()
        return call

    def close_call(
        self, call: Call | None, outputs: list[tuple[str, Any]] | None = None, *, hook_grad: bool = True
    ) -> None:
        if call is None:
            return
        end = self.timer.mark()
        if call not in self.stack:
            return  # already force-closed at a batch boundary
        with self.internal():
            while self.stack and self.stack[-1] is not call:
                orphan = self.stack.pop()
                orphan.end = end
                orphan.truncated = True
            self.stack.pop()
            call.end = end
            self.registry.phase = call.phase
            call.outputs = self._capture_values(outputs, hook_grad)

    def _capture_values(self, values: list[tuple[str, Any]] | None, hook_grad: bool) -> list[dict[str, Any]]:
        """Capture named values; a value that fails to capture becomes an "error" ref plus a notice,
        so call wrappers never propagate capture failures into the user's code."""
        captured = []
        for name, value in values or []:
            try:
                ref = capture_value(value, self.registry, hook_grad=hook_grad)
            except Exception as exc:
                self.notice(
                    "capture value", f"{type(exc).__name__}: {exc}", detail=traceback.format_exc(limit=6)
                )
                ref = {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
            captured.append({"name": name, "value": ref})
        return captured

    def find_call_site(self, start_frame: Any = None, max_depth: int = 12) -> dict[str, Any] | None:
        """The innermost project-file frame above tvis/library frames."""
        frame = start_frame or sys._getframe(1)
        for _ in range(max_depth):
            if frame is None:
                return None
            filename = frame.f_code.co_filename
            if not is_tvis_file(filename):
                rel = self.project.relpath(filename)
                if rel is not None:
                    return {"file": rel, "line": frame.f_lineno}
            frame = frame.f_back
        return None

    def module_identity(self, module: nn.Module) -> tuple[str, str | None]:
        """(display name, module path within its root model)."""
        cls = type(module).__name__
        entry = self._module_names.get(id(module))
        if entry is None:
            return cls, None
        root_idx, path = entry
        prefix = f"{root_idx}:" if len(self.models) > 1 else ""
        return (f"{prefix}{path}" if path else f"{prefix}{cls}"), f"{prefix}{path}"

    def module_source(self, module: nn.Module) -> tuple[str | None, int | None]:
        cls = type(module)
        if cls not in self._module_class_src:
            file = line = None
            with contextlib.suppress(Exception):
                code = cls.forward.__code__
                file = self.project.relpath(code.co_filename)
                line = code.co_firstlineno if file else None
            self._module_class_src[cls] = (file, line)
        return self._module_class_src[cls]

    # ------------------------------------------------------------------------------------------
    # batch / step state machine
    # ------------------------------------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        batch = self.batch
        if batch is not None and batch.phase != phase:
            batch.phase = phase
            batch.phase_marks.append((phase, self.timer.mark()))
            self.registry.phase = phase

    def on_data_fetch(self) -> None:
        """Called before a DataLoader yields a batch."""
        batch = self.batch
        if batch is None or batch.has_forward or batch.has_backward or batch.phase != "data":
            self._begin_batch()
        else:
            self.set_phase("data")

    def on_root_forward(self, module: nn.Module) -> None:
        """A top-level forward starts a new batch after a backward, when grad mode changes (a no-grad
        sanity check before a training forward), or when a no-grad forward repeats a model (an
        inference loop). Repeated grad-enabled forwards stay in one batch (siamese/contrastive)."""
        grad = torch.is_grad_enabled()
        batch = self.batch
        if (
            batch is None
            or batch.has_backward
            or (batch.forward_grad is not None and batch.forward_grad != grad)
            or (not grad and id(module) in batch.forward_models)
        ):
            self._begin_batch()
        self._register_model(module)
        batch = self.batch
        assert batch is not None
        batch.has_forward = True
        batch.forward_grad = grad
        batch.forward_models.add(id(module))
        self.set_phase("forward")

    def on_root_forward_end(self) -> None:
        if self.batch is not None and self.batch.phase == "forward":
            self.set_phase("loss")

    def _begin_batch(self) -> None:
        if self.batch is not None:
            self._finalize_batch()
        if self.step.optimizer_steps > 0 or (not self.optimizers and self._step_has_backward()):
            self._close_step(check_stop=True)
        if self.batches_written >= self.config.max_batches:
            self.notice(
                "capture", f"stopped after max_batches={self.config.max_batches} without completing steps"
            )
            self._stop()
        start = self.timer.mark()
        self.batch = Batch(index=self.batches_written, step=self.step.index, start=start)
        self.batch.phase_marks.append(("data", start))
        self.registry.batch_index = self.batch.index
        self.registry.phase = "data"
        self.step.batches.append(self.batch.index)

    def _step_has_backward(self) -> bool:
        return bool(self.step.batches) and self._last_batch_had_backward

    def _stop(self) -> None:
        self.finish(schema.STATUS_COMPLETE)
        raise StopCapture

    # -- finalisation --------------------------------------------------------------------------
    def _finalize_batch(self) -> None:
        batch = self.batch
        assert batch is not None
        self.batch = None
        end = self.timer.mark()
        for call in self.stack:
            if call.end is None:
                call.end = end
                call.truncated = True
        self.stack = []
        with self.guard("finalize batch"), self.internal():
            self._last_batch_had_backward = batch.has_backward
            self.timer.synchronize()
            records = self.registry.take_batch(batch.index)
            by_tid = {r.tid: r for r in records}
            batch_size = batch.batch_size or self._infer_batch_size(batch, by_tid)
            if batch_size:
                collated = _collated_tids(batch)
                for r in records:
                    r.batch_dim = _batch_dim(r, batch_size, collated)
            from tvis.capture.decode import build_samples

            samples = build_samples(self, batch, by_tid, batch_size)
            tensor_arrivals = {r.tid: r.grad_arrival for r in records if r.grad_arrival is not None}
            param_marks = _param_marks_by_call(batch)
            calls_json = [
                call.to_json(self.timer, tensor_arrivals, param_marks.get(call.id, []))
                for call in batch.calls
            ]
            for call in batch.calls:
                call.module_ref = None
            for r in records:
                self._save_record(batch.index, r)
            self.writer.write_batch(
                batch.index,
                {
                    "index": batch.index,
                    "step": batch.step,
                    "batch_size": batch_size,
                    "indices": batch.indices,
                    "flags": batch.flags + ([] if batch.has_backward else ["no_backward"]),
                    "timing": self._batch_timing(batch, end),
                    "calls": calls_json,
                    "tensors": {r.tid: r.to_json(self.timer.ms) for r in records},
                    "ops": batch.ops,
                    "samples": samples,
                },
            )
        self.batches_written += 1

    def _save_record(self, batch_index: int, record: TensorRecord) -> None:
        if record.value is not None:
            self.writer.save_array(batch_index, schema.value_array_name(record.tid), to_numpy(record.value))
        if record.grad is not None:
            self.writer.save_array(batch_index, schema.grad_array_name(record.tid), to_numpy(record.grad))
        record.value = record.grad = None

    def _batch_timing(self, batch: Batch, end: Mark) -> dict[str, Any]:
        phases: dict[str, float] = {}
        wall: dict[str, float] = {}
        marks = [*batch.phase_marks, ("_end", end)]
        for (phase, start), (_, stop) in itertools.pairwise(marks):
            phases[phase] = phases.get(phase, 0.0) + self.timer.duration(start, stop)
            wall[phase] = wall.get(phase, 0.0) + (stop.wall - start.wall) * 1000.0
        return {
            "clock": self.timer.kind,
            "phases_ms": phases,
            "total_ms": sum(phases.values()),
            "wall_phases_ms": wall,
            "wall_total_ms": (end.wall - batch.start.wall) * 1000.0,
            "start_ms": self.timer.ms(batch.start),
        }

    def _infer_batch_size(self, batch: Batch, by_tid: dict[str, TensorRecord]) -> int | None:
        for call in batch.calls:
            if call.kind == K.DATA and call.outputs:
                for ref in iter_tensor_refs(call.outputs[0]["value"]):
                    rec = by_tid.get(ref["tid"])
                    if rec is not None and rec.shape:
                        return rec.shape[0]
        for call in batch.calls:
            if call.kind == K.MODULE and call.depth == 0 and call.inputs:
                for ref in iter_tensor_refs(call.inputs[0]["value"]):
                    rec = by_tid.get(ref["tid"])
                    if rec is not None and rec.shape:
                        return rec.shape[0]
        return None

    def _close_step(self, *, check_stop: bool) -> None:
        step = self.step
        doc = {
            "index": step.index,
            "batches": step.batches,
            "optimizer_steps": step.optimizer_steps,
            "params": step.params,
        }
        self.steps.append(doc)
        self.writer.write_steps(self.steps)
        self.step = Step(index=step.index + 1)
        self._write_meta(status=schema.STATUS_RUNNING)
        if check_stop and len(self.steps) >= self.config.steps:
            self._stop()

    # ------------------------------------------------------------------------------------------
    # models, parameters, optimizers, backward
    # ------------------------------------------------------------------------------------------
    def _register_model(self, model: nn.Module) -> None:
        if any(m is model for m in self.models):
            return
        root_idx = len(self.models)
        self.models.append(model)
        for path, mod in model.named_modules():
            self._module_names.setdefault(id(mod), (root_idx, path))
        params = list(model.named_parameters())
        for opt in _optimizers_owning([p for _, p in params]):
            if not any(o is opt for o in self.optimizers):
                self.optimizers.append(opt)
        if params and self.timer.record_cuda and len(self.models) == 1:
            self.timer.use_cuda = params[0][1].is_cuda
        for name, p in params:
            self._param_names.setdefault(id(p), name if root_idx == 0 else f"{root_idx}:{name}")
            if p.requires_grad and id(p) not in self._hooked_params:
                self._hooked_params.add(id(p))
                self._handles.append(p.register_post_accumulate_grad_hook(self._make_param_hook()))

    def _make_param_hook(self) -> Callable[[torch.Tensor], None]:
        def hook(param: torch.Tensor) -> None:
            batch = self.batch
            if batch is not None and id(param) not in batch.param_arrivals:
                batch.param_arrivals[id(param)] = self.timer.mark()

        return hook

    def _patch_backward(self) -> None:
        original = torch.autograd.backward
        session = self

        def backward(tensors: Any, *args: Any, **kwargs: Any) -> Any:
            if not session.active or session.done or session.in_internal():
                return original(tensors, *args, **kwargs)
            call = None
            with session.guard("backward start"):
                if session.batch is None:
                    session._begin_batch()
                session.set_phase("backward")
                call = session.open_call(
                    K.BACKWARD,
                    "backward",
                    inputs=[("loss", tensors)],
                    hook_grad=False,
                    call_site=session.find_call_site(sys._getframe(1)),
                )
            try:
                return original(tensors, *args, **kwargs)
            finally:
                with session.guard("backward end"):
                    session.close_call(call)
                    if session.batch is not None:
                        session.batch.has_backward = True
                        session.set_phase("post_backward")

        functools_wraps(backward, original)
        self.patcher.set(torch.autograd, "backward", backward)

    def _patch_optimizers(self) -> None:
        session = self
        original_init = torch.optim.Optimizer.__init__

        def __init__(opt: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> None:
            original_init(opt, *args, **kwargs)
            if not any(o is opt for o in session.optimizers):
                session.optimizers.append(opt)

        functools_wraps(__init__, original_init)
        self.patcher.set(torch.optim.Optimizer, "__init__", __init__)
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

        self._handles.append(register_optimizer_step_pre_hook(self._on_optimizer_pre))
        self._handles.append(register_optimizer_step_post_hook(self._on_optimizer_post))

    def _on_optimizer_pre(self, opt: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        if not self.active or self.done or self.in_internal():
            return
        if not any(o is opt for o in self.optimizers):
            self.optimizers.append(opt)
        with self.guard("optimizer pre"):
            if self.batch is None:
                self._begin_batch()
                assert self.batch is not None
                self.batch.flags.append("optimizer_only")
            self.set_phase("optimizer")
            with self.internal():
                snapshot = [
                    (p, p.detach().clone())
                    for group in opt.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            call = self.open_call(
                K.OPTIMIZER,
                type(opt).__name__,
                call_site=self.find_call_site(sys._getframe(1)),
                extra={"hyper": _optimizer_hyper(opt)},
            )
            if call is not None:
                call.extra["_snapshot"] = snapshot
                self._tl.optimizer_call = call

    def _on_optimizer_post(self, opt: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        if not self.active or self.done or self.in_internal():
            return
        call = getattr(self._tl, "optimizer_call", None)
        self._tl.optimizer_call = None
        with self.guard("optimizer post"):
            self.close_call(call)
            if call is not None:
                snapshot = call.extra.pop("_snapshot", [])
                with self.internal():
                    for p, before in snapshot:
                        self.step.params.append(self._param_update(p, before, call.id))
            self.step.optimizer_steps += 1
        if len(self.optimizers) <= 1 and self.batch is not None:
            self._finalize_batch()
            self._close_step(check_stop=True)

    def _param_update(self, p: torch.Tensor, before: torch.Tensor, call_id: int) -> dict[str, Any]:
        after = p.detach()
        update = after - before
        weight_norm = float(torch.linalg.vector_norm(before.float()))
        update_norm = float(torch.linalg.vector_norm(update.float()))
        grad = p.grad.detach() if p.grad is not None else None
        refs = {
            "weight": capture_value(before, self.registry, hook_grad=False),
            "update": capture_value(update, self.registry, hook_grad=False),
        }
        if grad is not None:
            refs["grad"] = capture_value(grad, self.registry, hook_grad=False)
        return {
            "name": self._param_names.get(id(p), f"param@{id(p):x}"),
            "shape": list(p.shape),
            "optimizer_call": call_id,
            "weight_norm": weight_norm,
            "update_norm": update_norm,
            "grad_norm": float(torch.linalg.vector_norm(grad.float())) if grad is not None else None,
            "update_ratio": (update_norm / weight_norm) if weight_norm > 0 else None,
            **refs,
        }

    # ------------------------------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------------------------------
    def _snapshot_source(self, relpath: str, text: str) -> None:
        self.sources.append(relpath)
        with contextlib.suppress(Exception):
            self.writer.write_source(relpath, text)

    def _write_meta(self, *, status: str, error: str | None = None) -> None:
        cfg = self.config
        meta = {
            "run_id": cfg.run_dir.name,
            "status": status,
            "error": error,
            "script": cfg.script,
            "argv": cfg.argv,
            "project_dir": str(cfg.project_dir),
            "host": socket.gethostname(),
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps_requested": cfg.steps,
            "steps_captured": len(self.steps),
            "batches_captured": self.batches_written,
            "config": {
                "max_elems": cfg.max_elems,
                "max_batches": cfg.max_batches,
                "trace_functions": cfg.trace_functions,
                "record_ops": cfg.record_ops,
            },
            "env": _environment(),
            "clock": self.timer.kind,
            "models": [_describe_model(i, m, self) for i, m in enumerate(self.models)],
            "optimizers": [{"cls": type(o).__name__, "hyper": _optimizer_hyper(o)} for o in self.optimizers],
            "data": self.data_info,
            "sources": sorted(set(self.sources)),
            "notices": list(self.notices.values()),
        }
        with contextlib.suppress(Exception):
            self.writer.write_meta(meta)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def functools_wraps(wrapper: Any, wrapped: Any) -> None:
    import functools

    functools.update_wrapper(wrapper, wrapped)
    wrapper.__tvis_original__ = wrapped


def _param_marks_by_call(batch: Batch) -> dict[int, list[Mark]]:
    """Gradient-accumulation marks of the parameters each call used: a module's own parameters,
    or for a function, the parameters of every module called beneath it."""
    own: dict[int, list[Mark]] = {}
    for call in batch.calls:
        if call.kind == K.MODULE and call.module_ref is not None:
            own[call.id] = [
                batch.param_arrivals[id(p)]
                for p in call.module_ref.parameters()
                if id(p) in batch.param_arrivals
            ]
    result = dict(own)
    by_id = {c.id: c for c in batch.calls}
    for call in batch.calls:
        if call.kind != K.MODULE or call.id not in own:
            continue
        parent = by_id.get(call.parent) if call.parent is not None else None
        while parent is not None and parent.kind == K.FUNCTION:
            result.setdefault(parent.id, []).extend(own[call.id])
            parent = by_id.get(parent.parent) if parent.parent is not None else None
    return result


def _collated_tids(batch: Batch) -> set[str]:
    return {
        ref["tid"]
        for call in batch.calls
        if call.kind == K.DATA
        for out in call.outputs
        for ref in iter_tensor_refs(out["value"])
    }


def _batch_dim(record: TensorRecord, batch_size: int, collated: set[str]) -> dict[str, Any] | None:
    """Where the batch dimension is: exact when dim 0 equals the batch size; "inferred" when dim 0
    is a multiple of it (batch-major merges such as [B*T, V] or [B*H, T, D]). Data-pipeline tensors
    before collation are per-sample and have no batch dimension."""
    if record.kind != "tensor" or not record.shape or record.phase in (None, "optimizer"):
        return None
    if record.phase == "data" and record.tid not in collated:
        return None
    lead = record.shape[0]
    if lead == batch_size:
        return {"dim": 0, "block": 1, "inferred": False}
    if batch_size > 1 and lead > batch_size and lead % batch_size == 0:
        return {"dim": 0, "block": lead // batch_size, "inferred": True}
    return None


def _optimizer_hyper(opt: torch.optim.Optimizer) -> list[dict[str, Any]]:
    groups = []
    for group in opt.param_groups:
        hyper = {}
        for key, value in group.items():
            if key == "params":
                hyper["n_params"] = len(value)
            elif isinstance(value, (int, float, bool, str)) or value is None:
                hyper[key] = value
            elif isinstance(value, (tuple, list)) and all(isinstance(v, (int, float)) for v in value):
                hyper[key] = list(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                hyper[key] = float(value)
        groups.append(hyper)
    return groups


def _describe_model(index: int, model: nn.Module, session: Session) -> dict[str, Any]:
    modules = []
    for path, mod in model.named_modules():
        file, line = session.module_source(mod)
        own = sum(p.numel() for p in mod.parameters(recurse=False))
        modules.append({"path": path, "cls": type(mod).__name__, "params": own, "file": file, "line": line})
    params = list(model.parameters())
    first = params[0] if params else None
    return {
        "index": index,
        "cls": type(model).__name__,
        "n_params": sum(p.numel() for p in params),
        "trainable": sum(p.numel() for p in params if p.requires_grad),
        "device": str(first.device) if first is not None else None,
        "dtype": str(first.dtype).removeprefix("torch.") if first is not None else None,
        "modules": modules,
    }


def _environment() -> dict[str, Any]:
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": None,
    }
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            env["device"] = torch.cuda.get_device_name()
    return env


def _optimizers_owning(params: list[torch.Tensor]) -> list[torch.optim.Optimizer]:
    """Live optimizers that update any of ``params``: catches optimizers constructed before the
    session was installed, without counting unrelated ones still alive elsewhere."""
    import gc

    wanted = {id(p) for p in params}
    return [
        obj
        for obj in gc.get_objects()
        if issubclass(type(obj), torch.optim.Optimizer)
        and any(id(p) in wanted for group in obj.param_groups for p in group["params"])
    ]


def signature_names(func: Any) -> list[str]:
    try:
        return list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return []
