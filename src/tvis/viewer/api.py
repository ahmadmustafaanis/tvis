"""Request dispatch shared by the local HTTP server and the remote stdio agent.

Every request is ``(method, params) -> JSON-serialisable result``. Errors are raised as
:class:`ApiError` with a message safe to show in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tvis.store.reader import RunNotFound, RunsRoot
from tvis.viewer import views


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Api:
    def __init__(self, root: Path):
        self.root = RunsRoot(root)
        self._batch_cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        handler = getattr(self, f"_{method}", None) if method.isidentifier() else None
        if handler is None:
            raise ApiError(f"unknown method: {method}", status=404)
        try:
            return handler(**params)
        except TypeError as exc:
            raise ApiError(f"bad parameters for {method}: {exc}") from exc
        except RunNotFound as exc:
            raise ApiError(f"not found: {exc}", status=404) from exc
        except (views.ViewError, ValueError) as exc:
            raise ApiError(str(exc)) from exc

    # -- methods -------------------------------------------------------------------------------
    def _ping(self) -> dict[str, Any]:
        from tvis import __version__

        return {"ok": True, "version": __version__, "root": str(self.root.root)}

    def _runs(self) -> list[dict[str, Any]]:
        return self.root.list_runs()

    def _run(self, run: str) -> dict[str, Any]:
        r = self.root.run(run)
        return {"meta": r.meta(), "steps": r.steps()}

    def _batch(self, run: str, batch: int) -> dict[str, Any]:
        return self._load_batch(self.root.run(run), int(batch))

    def _source(self, run: str, path: str) -> dict[str, Any]:
        return {"path": path, "text": self.root.run(run).source(path)}

    def _tensor(
        self,
        run: str,
        batch: int,
        tid: str,
        grad: bool = False,
        sample: int | None = None,
        fixed: list[int] | None = None,
    ) -> dict[str, Any]:
        r = self.root.run(run)
        meta = self._tensor_meta(r, int(batch), tid)
        array = r.array(int(batch), tid, grad=bool(grad))
        if sample is not None:
            array = views.select_sample(array, meta, int(sample), grad=bool(grad))
        return {"tid": tid, "grad": bool(grad), "sample": sample, **views.tensor_view(array, fixed=fixed)}

    def _image(
        self, run: str, batch: int, tid: str, grad: bool = False, sample: int | None = None
    ) -> dict[str, Any]:
        r = self.root.run(run)
        meta = self._tensor_meta(r, int(batch), tid)
        array = r.array(int(batch), tid, grad=bool(grad))
        if sample is not None:
            array = views.select_sample(array, meta, int(sample), grad=bool(grad))
        normalize = r.meta().get("data", {}).get("normalize")
        return views.image_view(array, kind=meta.get("kind", "tensor"), normalize=normalize, grad=bool(grad))

    def _tensor_meta(self, run: Any, batch: int, tid: str) -> dict[str, Any]:
        meta = self._load_batch(run, batch)["tensors"].get(tid)
        if meta is None:
            raise RunNotFound(f"tensor {tid} in batch {batch}")
        return meta

    def _load_batch(self, run: Any, batch: int) -> dict[str, Any]:
        """batch.json can be several MB; tensor requests reuse a parsed copy until the file changes."""
        key = (str(run.run_dir), batch)
        mtime = run.batch_mtime(batch)
        cached = self._batch_cache.get(key)
        if cached is None or cached[0] != mtime:
            cached = (mtime, run.batch(batch))
            self._batch_cache[key] = cached
        return cached[1]
