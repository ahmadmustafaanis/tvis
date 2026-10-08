"""Project-function tracer.

Uses ``sys.setprofile`` (call/return events only, no per-line cost) and records calls to functions
*defined in the user's project directory*. Library code is never recorded. A user module's
``forward`` is not recorded twice: it is merged into the module call opened by the module hook.
"""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

from torch import nn

from tvis.capture import calls as K

if TYPE_CHECKING:
    from tvis.capture.session import Session

_SKIP_FLAGS = (
    inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_ITERABLE_COROUTINE
)
_MISSING = object()
_DUNDER_ALLOWED = {"__call__", "__getitem__", "__getitems__"}


class FunctionTracer:
    def __init__(self, session: Session):
        self.session = session
        self._code_info: dict[Any, tuple[str, str] | None] = {}
        self._previous: Any = None

    def install(self) -> None:
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)

    def uninstall(self) -> None:
        if sys.getprofile() is self._profile:
            sys.setprofile(self._previous)

    def _profile(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            if self.session.recording:
                self._on_call(frame)
        elif event == "return" and self.session.stack:
            self._on_return(frame, arg)

    def _info(self, code: Any) -> tuple[str, str] | None:
        info = self._code_info.get(code, _MISSING)
        if info is _MISSING:
            info = None
            name = code.co_name
            dunder = name.startswith("__") and name.endswith("__") and name not in _DUNDER_ALLOWED
            if not (code.co_flags & _SKIP_FLAGS) and not name.startswith("<") and not dunder:
                rel = self.session.project.relpath(code.co_filename)
                if rel is not None:
                    info = (rel, getattr(code, "co_qualname", code.co_name))
            self._code_info[code] = info
        return info  # type: ignore[return-value]

    def _on_call(self, frame: Any) -> None:
        code = frame.f_code
        info = self._info(code)
        if info is None:
            return
        s = self.session
        with s.guard("tracer call"):
            prepared = self._prepare(frame, code)
            if prepared is None:
                return
            inputs, call_site = prepared
            call = s.open_call(
                K.FUNCTION,
                code.co_name,
                inputs=inputs,
                qualname=info[1],
                file=info[0],
                line=code.co_firstlineno,
                call_site=call_site,
            )
            if call is not None:
                call.frame_id = id(frame)

    def _prepare(self, frame: Any, code: Any) -> tuple[list, dict | None] | None:
        """Inspect the new frame, bracketed as tvis overhead. None for a module's ``forward``, which
        is merged into the module call rather than recorded twice."""
        s = self.session
        with s.internal():
            local_vars = frame.f_locals
            names = _argument_names(code)
            first = local_vars.get(names[0]) if names else None
            if code.co_name == "forward" and isinstance(first, nn.Module):
                top = s.stack[-1] if s.stack else None
                if (
                    top is not None
                    and top.kind == K.MODULE
                    and top.module_ref is first
                    and top.frame_id is None
                ):
                    top.frame_id = id(frame)
                    return None
            inputs = [(n, local_vars[n]) for n in names if n in local_vars and n not in ("self", "cls")]
            caller = frame.f_back
            call_site = None
            if caller is not None:
                caller_rel = s.project.relpath(caller.f_code.co_filename)
                if caller_rel is not None:
                    call_site = {"file": caller_rel, "line": caller.f_lineno}
            return inputs, call_site

    def _on_return(self, frame: Any, value: Any) -> None:
        s = self.session
        frame_id = id(frame)
        call = next((c for c in reversed(s.stack) if c.frame_id == frame_id), None)
        if call is None or call.kind != K.FUNCTION:
            return  # module calls are closed by the module post-hook
        with s.guard("tracer return"):
            s.close_call(call, outputs=[("return", value)])


def _argument_names(code: Any) -> list[str]:
    n = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & inspect.CO_VARARGS:
        n += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        n += 1
    return list(code.co_varnames[:n])
