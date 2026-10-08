"""Global nn.Module forward hooks: one call record per module call, for every model in the script."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from torch import nn
from torch.nn.modules import module as torch_module

from tvis.capture import calls as K

if TYPE_CHECKING:
    from tvis.capture.session import Session

_SAMPLE_MIXING = (
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
)


class ModuleHooks:
    def __init__(self, session: Session):
        self.session = session
        self._handles: list[Any] = []
        self._arg_names: dict[type, list[str]] = {}

    def install(self) -> None:
        self._handles = [
            torch_module.register_module_forward_pre_hook(self._pre),
            torch_module.register_module_forward_hook(self._post, always_call=True),
        ]

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _skip(self) -> bool:
        s = self.session
        return not s.active or s.done or s.in_internal() or getattr(s._tl, "suppress_modules", 0) > 0

    def _pre(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        if self._skip():
            return
        s = self.session
        in_module = any(c.kind == K.MODULE for c in s.stack)
        is_root = not in_module and _has_parameters(module)
        if is_root:
            with s.guard("batch boundary"):
                s.on_root_forward(module)  # may start a batch, or raise StopCapture (passes the guard)
        if not s.recording:
            return
        with s.guard("module pre-hook"):
            with s.internal():
                name, path = s.module_identity(module)
                file, line = s.module_source(module)
                names = self._names(module, len(args))
            call = s.open_call(
                K.MODULE,
                name,
                inputs=list(zip(names, args, strict=False)),
                cls=type(module).__name__,
                module_path=path,
                file=file,
                line=line,
                site_frame=sys._getframe(1),
            )
            if call is None:
                return
            call.module_ref = module
            if is_root:
                call.extra["root"] = True
            if module.training and type(module).__name__ in _SAMPLE_MIXING:
                call.flags.append("mixes_samples")

    def _post(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        if self._skip():
            return
        s = self.session
        call = next(
            (c for c in reversed(s.stack) if c.kind == K.MODULE and c.module_ref is module),
            None,
        )
        if call is None:
            return
        with s.guard("module post-hook"):
            s.close_call(call, outputs=[("output", output)])
            if call.extra.get("root"):
                s.on_root_forward_end()

    def _names(self, module: nn.Module, n_args: int) -> list[str]:
        cls = type(module)
        names = self._arg_names.get(cls)
        if names is None:
            from tvis.capture.session import signature_names

            names = [n for n in signature_names(module.forward) if n not in ("args", "kwargs")]
            self._arg_names[cls] = names
        return [names[i] if i < len(names) else f"arg{i}" for i in range(n_args)]


def _has_parameters(module: nn.Module) -> bool:
    return next(module.parameters(), None) is not None
