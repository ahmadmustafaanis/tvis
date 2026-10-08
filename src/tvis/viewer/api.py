"""Request dispatch shared by the local HTTP server and the remote stdio agent.

Every request is ``(method, params) -> JSON-serialisable result``. Errors are raised as
:class:`ApiError` with a message safe to show in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        self,
        run: str,
        batch: int,
        tid: str,
        grad: bool = False,
        sample: int | None = None,
        unnormalize: bool | None = None,
    ) -> dict[str, Any]:
        """``unnormalize``: apply the inverse of the pipeline's Normalize. By default only to tensors
        that were normalised: a Normalize output, the collated batch, or anything in the model."""
        r = self.root.run(run)
        meta = self._tensor_meta(r, int(batch), tid)
        array = r.array(int(batch), tid, grad=bool(grad))
        if sample is not None:
            array = views.select_sample(array, meta, int(sample), grad=bool(grad))
        if unnormalize is None:
            unnormalize = bool(
                meta.get("normalized") or meta.get("batch_dim") or meta.get("phase") not in ("data", None)
            )
        normalize = r.meta().get("data", {}).get("normalize") if unnormalize else None
        return views.image_view(array, kind=meta.get("kind", "tensor"), normalize=normalize, grad=bool(grad))

    def _array(
        self, run: str, batch: int, tid: str, grad: bool = False, sample: int | None = None
    ) -> dict[str, Any]:
        r = self.root.run(run)
        meta = self._tensor_meta(r, int(batch), tid)
        array = r.array(int(batch), tid, grad=bool(grad))
        if sample is not None:
            array = views.select_sample(array, meta, int(sample), grad=bool(grad))
        return {"tid": tid, **views.full_array(array)}

    def _thumb(
        self, run: str, batch: int, tid: str, sample: int | None = None, grad: bool = False, size: int = 48
    ) -> dict[str, Any]:
        r = self.root.run(run)
        meta = self._tensor_meta(r, int(batch), tid)
        array = r.array(int(batch), tid, grad=bool(grad))
        if sample is not None and meta.get("batch_dim"):
            array = views.select_sample(array, meta, int(sample), grad=bool(grad))
        return {"tid": tid, **views.thumbnail(array, size=max(4, min(int(size), 256)))}

    def _loss_detail(self, run: str, batch: int) -> dict[str, Any]:
        """How the batch loss is built: per-element loss, targets, predictions and p(target)."""
        r = self.root.run(run)
        doc = self._load_batch(r, int(batch))
        call = next(
            (
                c
                for c in reversed(doc["calls"])
                if c["kind"] == "loss" and "per_element" in c.get("extra", {})
            ),
            None,
        )
        if call is None:
            raise ApiError("this batch has no loss with per-element values", status=404)
        extra = call["extra"]
        per = np.asarray(r.array(int(batch), extra["per_element"]["tid"]), dtype=np.float64)
        rows = doc.get("batch_size") or (per.shape[0] if per.ndim else 1)
        if per.size % rows:
            rows = 1
        per = per.reshape(rows, -1)
        out: dict[str, Any] = {
            "call": call["id"],
            "name": call["name"],
            "reduction": extra.get("reduction"),
            "ignore_index": extra.get("ignore_index"),
            "rows": rows,
            "cols": int(per.shape[1]),
            "per_element": _finite_lists(per),
        }
        loss_ref = _input_ref(call, "loss", outputs=True)
        if loss_ref:
            out["loss"] = doc["tensors"][loss_ref["tid"]]["stats"]["mean"]
        target = self._maybe_array(r, doc, int(batch), _input_ref(call, "target"))
        valid = np.ones_like(per, dtype=bool)
        if target is not None and target.dtype.kind in "iu" and target.size == per.size:
            target = target.reshape(rows, -1)
            if extra.get("ignore_index") is not None:
                valid = target != extra["ignore_index"]
            out["target_ids"] = target.tolist()
        out["valid"] = valid.tolist()
        logits = self._maybe_array(r, doc, int(batch), _input_ref(call, "input"))
        if (
            call["name"] in ("cross_entropy", "nll_loss")
            and logits is not None
            and logits.ndim == 2
            and logits.shape[0] == per.size
            and "target_ids" in out
        ):
            logits = logits.astype(np.float64)
            if call["name"] == "cross_entropy":
                shifted = logits - logits.max(axis=1, keepdims=True)
                logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            else:
                logp = logits
            flat_target = np.clip(np.asarray(out["target_ids"]).reshape(-1), 0, logits.shape[1] - 1)
            p_target = np.exp(logp[np.arange(logits.shape[0]), flat_target])
            out["pred_ids"] = logp.argmax(axis=1).reshape(rows, -1).tolist()
            out["p_target"] = _finite_lists(p_target.reshape(rows, -1))
        return out

    def _projection(self, run: str, path: str, max_points: int = 5000) -> dict[str, Any]:
        """PCA of one layer's per-sample outputs across every captured batch."""
        r = self.root.run(run)
        meta = r.meta()
        features, points = [], []
        for b in range(meta.get("batches_captured", 0)):
            doc = self._load_batch(r, b)
            call = next(
                (
                    c
                    for c in doc["calls"]
                    if c["phase"] == "forward"
                    and (c.get("module_path") or c["name"]) == path
                    and c["outputs"]
                ),
                None,
            )
            ref = _first_tensor(call["outputs"][0]["value"]) if call else None
            tmeta = doc["tensors"].get(ref["tid"]) if ref else None
            if not tmeta or (tmeta.get("batch_dim") or {}).get("block") != 1 or tmeta.get("stored") == "none":
                continue
            pooled = views.pool_features(r.array(b, ref["tid"]))
            features.append(pooled)
            samples = doc.get("samples") or []
            for i in range(pooled.shape[0]):
                s = samples[i] if i < len(samples) else {}
                target = s.get("target") or {}
                points.append(
                    {
                        "batch": b,
                        "position": i,
                        "label": target.get("name") if target.get("name") is not None else target.get("id"),
                        "correct": (s.get("prediction") or {}).get("correct"),
                        "loss": s.get("loss"),
                    }
                )
        if not features:
            raise ApiError(f"no per-sample outputs recorded for {path!r}", status=404)
        x = np.concatenate(features)[:max_points]
        coords, ratio = views.pca2(x)
        for point, (px, py) in zip(points, coords, strict=False):
            point["x"], point["y"] = float(px), float(py)
        return {"path": path, "dim": int(x.shape[1]), "explained": ratio, "points": points[: len(x)]}

    def _weight_projection(self, run: str, step: int, param: str, max_points: int = 5000) -> dict[str, Any]:
        """PCA of the rows of a 2-D weight (e.g. an embedding table), labelled by token if known."""
        r = self.root.run(run)
        steps = r.steps()
        if not 0 <= int(step) < len(steps):
            raise ApiError(f"no step {step}", status=404)
        record = next((p for p in steps[int(step)]["params"] if p["name"] == param), None)
        if record is None:
            raise ApiError(f"no parameter {param!r} in step {step}", status=404)
        weight = np.asarray(r.array(record["weight"]["batch"], record["weight"]["tid"]))
        if weight.ndim != 2:
            raise ApiError(f"{param} is not a matrix (shape {list(weight.shape)})")
        rows = weight[:max_points]
        coords, ratio = views.pca2(rows)
        vocab = r.meta().get("data", {}).get("vocab")
        labels = vocab if vocab and len(vocab) == weight.shape[0] else None
        return {
            "param": param,
            "rows": int(weight.shape[0]),
            "dim": int(weight.shape[1]),
            "explained": ratio,
            "points": [
                {"id": i, "label": labels[i] if labels else str(i), "x": float(x), "y": float(y)}
                for i, (x, y) in enumerate(coords)
            ],
        }

    def _maybe_array(self, run: Any, doc: dict[str, Any], batch: int, ref: dict | None) -> np.ndarray | None:
        if not ref or ref.get("kind") != "tensor":
            return None
        meta = doc["tensors"].get(ref["tid"])
        if not meta or meta.get("stored") != "full":
            return None
        return np.asarray(run.array(ref.get("batch", batch), ref["tid"]))

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


def _input_ref(call: dict[str, Any], name: str, outputs: bool = False) -> dict[str, Any] | None:
    for entry in call["outputs" if outputs else "inputs"]:
        if entry["name"] == name:
            return entry["value"]
    return None


def _first_tensor(value: dict[str, Any]) -> dict[str, Any] | None:
    if value.get("kind") == "tensor":
        return value
    items = value.get("items")
    children = items.values() if isinstance(items, dict) else items or []
    for child in children:
        found = _first_tensor(child)
        if found:
            return found
    return None


def _finite_lists(array: np.ndarray) -> list:
    return [[float(v) if np.isfinite(v) else None for v in row] for row in np.asarray(array)]
