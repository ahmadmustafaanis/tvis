#!/usr/bin/env python3
"""Read a tvis recording as text, so an agent (or a human in a terminal) can debug training.

Reads runs through tvis's own viewer API (numpy + stdlib only, no torch), locally or over SSH
exactly like `tvis open`:

    tvis_debug.py runs    TARGET                      list recordings, newest first
    tvis_debug.py summary TARGET [--run ID]           what ran, loss/accuracy per batch, health
    tvis_debug.py health  TARGET [--run ID]           automated checks for common training bugs
    tvis_debug.py layers  TARGET [--batch K]          forward calls: output + gradient stats, time
    tvis_debug.py params  TARGET [--step S]           per parameter: |w| |grad| |Δw| Δw/w per step
    tvis_debug.py samples TARGET [--batch K]          per sample: target, prediction, loss (worst first)
    tvis_debug.py data    TARGET [--batch K]          data pipeline: raw item → transforms → batch
    tvis_debug.py call    TARGET ID [--batch K]       one call: inputs/outputs, gradients, source
    tvis_debug.py tensor  TARGET TID [--batch K]      one tensor: stats + a window of its values
    tvis_debug.py source  TARGET FILE [--line N]      source snapshot with op shapes per line
    tvis_debug.py timing  TARGET [--batch K]          phase times and the slowest calls
    tvis_debug.py compare TARGET --run A --against B  two recordings side by side (before/after a fix)
    tvis_debug.py link    TARGET [--page P] ...       viewer URL for a run/batch/call/sample

TARGET is what `tvis open` takes: a runs directory, a single run directory, HOST:PATH, or a saved
target name. `--run` takes a run id, a unique prefix/suffix of one, or an index (0 = newest).

Health checks are heuristics; each finding says how it was detected and what to run next.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tvis.viewer import targets
    from tvis.viewer.api import ApiError
    from tvis.viewer.transport import LocalBackend, StdioBackend
except ImportError:  # pragma: no cover
    sys.exit("tvis_debug: `tvis` is not importable; run with the Python that has tvis installed")

PARAM_LAYERS = {
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Embedding",
    "EmbeddingBag",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "MultiheadAttention",
    "LSTM",
    "GRU",
    "RNN",
    "Bilinear",
}
SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


# ---------------------------------------------------------------------------------------------
# access
# ---------------------------------------------------------------------------------------------
class Reader:
    """Thin cache over the viewer API (`tvis.viewer.api.Api` methods, via a tvis backend)."""

    def __init__(self, target: str, via: list[str], remote_tvis: str | None):
        try:
            resolved = targets.resolve(target, via=via, remote_tvis=remote_tvis)
        except targets.TargetError as exc:
            sys.exit(f"tvis_debug: {exc}")
        if resolved.is_local:
            self.backend: Any = LocalBackend(Path(resolved.path))
        else:
            self.backend = StdioBackend(targets.ssh_command(resolved), description=resolved.describe())
        self.target = resolved.describe()
        self._runs: list[dict[str, Any]] | None = None
        self._run: dict[str, dict[str, Any]] = {}
        self._batches: dict[tuple[str, int], dict[str, Any]] = {}

    def request(self, method: str, **params: Any) -> Any:
        return self.backend.request(method, params)

    def runs(self) -> list[dict[str, Any]]:
        if self._runs is None:
            self._runs = self.request("runs")
        return self._runs

    def pick(self, spec: str | None) -> str:
        runs = self.runs()
        if not runs:
            sys.exit(f"tvis_debug: no runs under {self.target}")
        if spec is None:
            return runs[0]["run_id"]
        if spec.isdigit() and int(spec) < len(runs) and not any(r["run_id"] == spec for r in runs):
            return runs[int(spec)]["run_id"]
        exact = [r["run_id"] for r in runs if r["run_id"] == spec]
        partial = [r["run_id"] for r in runs if spec in r["run_id"]]
        found = exact or partial
        if len(found) != 1:
            ids = "\n  ".join(r["run_id"] for r in runs)
            sys.exit(f"tvis_debug: --run {spec!r} matches {len(found)} runs; available:\n  {ids}")
        return found[0]

    def run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._run:
            self._run[run_id] = self.request("run", run=run_id)
        return self._run[run_id]

    def batch(self, run_id: str, index: int) -> dict[str, Any]:
        key = (run_id, index)
        if key not in self._batches:
            self._batches[key] = self.request("batch", run=run_id, batch=index)
        return self._batches[key]

    def batches(self, run_id: str) -> list[dict[str, Any]]:
        out = []
        for index in range(self.run(run_id)["meta"].get("batches_captured", 0)):
            try:
                out.append(self.batch(run_id, index))
            except ApiError:
                break  # a crashed run may stop mid-batch
        return out

    def array(self, run_id: str, batch: int, tid: str, grad: bool = False) -> np.ndarray | None:
        try:
            result = self.request("array", run=run_id, batch=batch, tid=tid, grad=grad)
        except ApiError:
            return None
        data = np.frombuffer(base64.b64decode(result["data"]), dtype="<f4")
        return data.reshape(result["shape"]) if result["shape"] else data.reshape(())

    def close(self) -> None:
        self.backend.close()


# ---------------------------------------------------------------------------------------------
# helpers over the run format
# ---------------------------------------------------------------------------------------------
def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def table(rows: list[list[Any]], header: list[str]) -> str:
    cells = [[str(c) for c in header]] + [[c if isinstance(c, str) else num(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def tensors_in(value: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    if value.get("kind") == "tensor":
        return [value]
    items = value.get("items")
    children = items.values() if isinstance(items, dict) else items or []
    return [t for child in children for t in tensors_in(child)]


def first_tensor(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in entries:
        found = tensors_in(entry["value"])
        if found:
            return found[0]
    return None


def tmeta(doc: dict[str, Any], ref: dict[str, Any] | None) -> dict[str, Any] | None:
    return doc["tensors"].get(ref["tid"]) if ref and ref.get("batch", doc["index"]) == doc["index"] else None


def is_float(meta: dict[str, Any] | None) -> bool:
    return bool(meta) and str(meta.get("dtype", "")).startswith(("float", "bfloat", "complex"))


def label(call: dict[str, Any]) -> str:
    return call.get("module_path") or call.get("qualname") or call["name"]


def loss_call(doc: dict[str, Any]) -> dict[str, Any] | None:
    losses = [c for c in doc["calls"] if c["kind"] == "loss"]
    return losses[-1] if losses else None


def batch_loss(doc: dict[str, Any]) -> float | None:
    call = loss_call(doc)
    if call is None:
        return None
    meta = tmeta(doc, first_tensor(call["outputs"]))
    return meta["stats"].get("mean") if meta and meta.get("stats") else None


def batch_accuracy(doc: dict[str, Any]) -> float | None:
    marks = [(s.get("prediction") or {}).get("correct") for s in doc.get("samples") or []]
    marks = [m for m in marks if m is not None]
    if marks:
        return sum(marks) / len(marks)
    tokens = [(s.get("prediction") or {}).get("token_accuracy") for s in doc.get("samples") or []]
    tokens = [t for t in tokens if t is not None]
    return sum(tokens) / len(tokens) if tokens else None


def site(call: dict[str, Any]) -> str:
    s = call.get("call_site") or {}
    return f"{s['file']}:{s['line']}" if s.get("file") else ""


def stats_line(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "no stats"
    bad = f"  NaN={stats['nan']} Inf={stats['inf']}" if stats.get("nan") or stats.get("inf") else ""
    return (
        f"mean={num(stats.get('mean'))} std={num(stats.get('std'))} min={num(stats.get('min'))} "
        f"max={num(stats.get('max'))} |x|={num(stats.get('abs_mean'))} norm={num(stats.get('norm'))} "
        f"zeros={pct(stats.get('zero_frac'))}{bad}"
    )


def tensor_line(meta: dict[str, Any] | None, grad: bool = False) -> str:
    if meta is None:
        return "(not in this batch)"
    stats = meta.get("grad_stats" if grad else "stats")
    head = f"{meta['tid']} {meta.get('dtype')} {meta.get('shape')}"
    stored = meta.get("grad_stored" if grad else "stored")
    return f"{head} stored={stored}  {stats_line(stats)}"


# ---------------------------------------------------------------------------------------------
# health checks
# ---------------------------------------------------------------------------------------------
def finding(severity: str, check: str, message: str, nxt: str | None = None, **where: Any) -> dict:
    return {"severity": severity, "check": check, "message": message, "next": nxt, "where": where}


def health(reader: Reader, run_id: str) -> list[dict[str, Any]]:
    info = reader.run(run_id)
    meta, steps = info["meta"], info["steps"]
    docs = reader.batches(run_id)
    out: list[dict[str, Any]] = []

    if meta.get("status") == "error":
        tail = "\n".join((meta.get("error") or "").strip().splitlines()[-6:])
        out.append(finding("error", "run-failed", f"the script crashed during capture:\n{tail}"))
    elif meta.get("status") == "running":
        out.append(
            finding("warn", "run-incomplete", "status is 'running': capture still going or was killed")
        )
    for notice in meta.get("notices") or []:
        sev = "info" if notice.get("where") == "data" else "warn"
        out.append(finding(sev, "notice", f"[{notice.get('where')}] {notice.get('message')}"))
    if not docs:
        out.append(finding("error", "no-batches", "no batch was recorded"))
        return out

    out += _check_nonfinite(docs)
    out += _check_loss(docs, steps)
    out += _check_inputs(docs)
    out += _check_activations(docs)
    out += _check_params(meta, steps, docs)
    out += _check_stale_grads(reader, run_id, steps, docs)
    out += _check_labels(docs)
    out.sort(key=lambda f: SEVERITY_ORDER[f["severity"]])
    return out


def _producers(doc: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
    found: dict[str, tuple[dict[str, Any], str]] = {}
    for call in doc["calls"]:
        for role in ("outputs", "inputs"):
            for entry in call[role]:
                for ref in tensors_in(entry["value"]):
                    found.setdefault(ref["tid"], (call, f"{role[:-1]} '{entry['name']}'"))
    return found


def _check_nonfinite(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for doc in docs:
        producers = _producers(doc)
        order = {c["id"]: i for i, c in enumerate(doc["calls"])}

        def position(tid: str, producers: dict = producers, order: dict = order) -> int:
            call = producers.get(tid, (None,))[0]
            return order.get(call["id"], 10**9) if call else 10**9

        bad_values = [
            t
            for t in doc["tensors"].values()
            if (t.get("stats") or {}).get("nan") or (t.get("stats") or {}).get("inf")
        ]
        bad_grads = [
            t
            for t in doc["tensors"].values()
            if (t.get("grad_stats") or {}).get("nan") or (t.get("grad_stats") or {}).get("inf")
        ]
        found = []
        if bad_values:
            first = min(bad_values, key=lambda t: position(t["tid"]))
            call, role = producers.get(first["tid"], ({"id": None, "name": "?"}, "?"))
            found.append(
                finding(
                    "error",
                    "nonfinite-value",
                    f"batch {doc['index']}: first NaN/Inf value is the {role} of call {call['id']} "
                    f"({label(call)}, {site(call)}): {stats_line(first['stats'])}",
                    f"call {call['id']} --batch {doc['index']}",
                    batch=doc["index"],
                    call=call["id"],
                    tid=first["tid"],
                )
            )
        if bad_grads:  # backward runs in reverse: the origin is the latest call with a bad gradient
            origin = max(bad_grads, key=lambda t: -1 if position(t["tid"]) == 10**9 else position(t["tid"]))
            call, role = producers.get(origin["tid"], ({"id": None, "name": "?"}, "?"))
            found.append(
                finding(
                    "error",
                    "nonfinite-grad",
                    f"batch {doc['index']}: NaN/Inf gradients start at the {role} of call {call['id']} "
                    f"({label(call)}, {site(call)}) and flow back from there",
                    f"call {call['id']} --batch {doc['index']}",
                    batch=doc["index"],
                    call=call["id"],
                    tid=origin["tid"],
                )
            )
        if found:
            return found  # the first bad batch is the one to debug
    return []


def _check_loss(docs: list[dict[str, Any]], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    first = docs[0]
    call = loss_call(first)
    if call is None:
        return [
            finding(
                "warn", "no-loss", "no loss call was recognised (custom loss: per-sample views unavailable)"
            )
        ]
    inp = tmeta(first, first_tensor([e for e in call["inputs"] if e["name"] == "input"] or call["inputs"]))
    stats = (inp or {}).get("stats") or {}
    name = call["name"]
    where = f"{name} at {site(call)}"
    if name == "cross_entropy" and inp and len(inp.get("shape") or []) >= 2 and stats:
        if stats.get("min", -1) >= 0 and stats.get("max", 2) <= 1:
            out.append(
                finding(
                    "error",
                    "loss-input-probabilities",
                    f"{where} receives values in [0, 1] (min {num(stats['min'])}, max {num(stats['max'])}): "
                    "looks like probabilities. cross_entropy applies log_softmax itself; pass raw logits "
                    "(softmax applied twice flattens gradients).",
                    f"call {call['id']} --batch 0",
                    call=call["id"],
                )
            )
        classes = inp["shape"][1] if len(inp["shape"]) == 2 else inp["shape"][-1]
        loss0 = batch_loss(first)
        chance = math.log(classes) if classes and classes > 1 else None
        if loss0 is not None and chance and math.isfinite(loss0) and loss0 > 3 * chance:
            out.append(
                finding(
                    "warn",
                    "initial-loss-high",
                    f"first loss {num(loss0)} is far above chance ln({classes})={num(chance)}: the model "
                    "starts confidently wrong (input scale, init, or label/output mismatch).",
                    "layers --batch 0",
                )
            )
    if name == "nll_loss" and stats.get("max", -1) > 0:
        out.append(
            finding(
                "error",
                "nll-loss-on-logits",
                f"{where} receives positive values (max {num(stats['max'])}); nll_loss expects "
                "log-probabilities: apply log_softmax first or use cross_entropy.",
                f"call {call['id']} --batch 0",
                call=call["id"],
            )
        )
    if name == "binary_cross_entropy_with_logits" and stats.get("min", -1) >= 0 and stats.get("max", 2) <= 1:
        out.append(
            finding(
                "warn",
                "bce-logits-on-probabilities",
                f"{where} receives values in [0, 1]: sigmoid may be applied twice.",
                f"call {call['id']} --batch 0",
            )
        )
    per_step = []
    for step in steps:
        values = [batch_loss(d) for d in docs if d["index"] in step.get("batches", [])]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if values:
            per_step.append(sum(values) / len(values))
    if len(per_step) >= 3 and per_step[-1] > 1.5 * per_step[0]:
        out.append(
            finding(
                "warn",
                "loss-rising",
                f"loss rises across steps ({' → '.join(num(v) for v in per_step)}): learning rate too high, "
                "wrong sign, or stale gradients.",
                "params",
            )
        )
    if len(per_step) >= 2 and max(per_step) - min(per_step) <= 1e-7 * max(1.0, abs(per_step[0])):
        out.append(
            finding("warn", "loss-frozen", f"loss is identical every step ({num(per_step[0])})", "params")
        )
    return out


def _check_inputs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc = docs[0]
    roots = [
        c for c in doc["calls"] if c["kind"] == "module" and c["phase"] == "forward" and c["parent"] is None
    ]
    out = []
    for call in roots[:1]:
        for entry in call["inputs"]:
            for ref in tensors_in(entry["value"]):
                meta = tmeta(doc, ref)
                stats = (meta or {}).get("stats") or {}
                if not is_float(meta) or not stats:
                    continue
                if stats.get("max", 0) > 20 or stats.get("min", 0) < -20 or stats.get("std", 0) > 10:
                    out.append(
                        finding(
                            "warn",
                            "input-not-normalized",
                            f"model input '{entry['name']}' of {label(call)} spans "
                            f"[{num(stats['min'])}, {num(stats['max'])}] (std {num(stats['std'])}): looks "
                            "unnormalised (e.g. raw 0-255 pixels). Normalise to roughly zero mean, unit std.",
                            "data --batch 0",
                            call=call["id"],
                            tid=ref["tid"],
                        )
                    )
    return out


def _check_activations(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc = docs[0]
    out, seen = [], set()
    blocked = None  # latest forward call whose output gets an all-zero gradient
    for call in doc["calls"]:
        if call["phase"] != "forward" or call["kind"] not in ("module", "function"):
            continue
        meta = tmeta(doc, first_tensor(call["outputs"]))
        stats = (meta or {}).get("stats") or {}
        if not is_float(meta) or not stats or stats.get("numel", 0) < 16:
            continue
        key = label(call)
        grad = meta.get("grad_stats") or {}
        if grad.get("zero_frac", 0) >= 0.95 and stats.get("zero_frac", 0) < 0.95 and call["kind"] == "module":
            blocked = (call, stats, grad)
        if stats.get("zero_frac", 0) >= 0.95 and key not in seen:
            seen.add(key)
            out.append(
                finding(
                    "warn",
                    "dead-activations",
                    f"{pct(stats['zero_frac'])} of the outputs of {key} ({call.get('cls') or call['kind']}, "
                    f"{site(call)}) are exactly 0: dead ReLUs / everything clipped. Check its pre-activation "
                    "(bias, init, input scale).",
                    f"call {call['id']} --batch 0",
                    call=call["id"],
                )
            )
        if max(abs(stats.get("max") or 0), abs(stats.get("min") or 0)) > 1e4 and key not in seen:
            seen.add(key)
            out.append(
                finding(
                    "warn",
                    "activation-explodes",
                    f"outputs of {key} reach {num(max(abs(stats['max']), abs(stats['min'])))}",
                    f"call {call['id']} --batch 0",
                    call=call["id"],
                )
            )
    if blocked is not None:
        call, stats, grad = blocked
        out.append(
            finding(
                "warn",
                "gradient-blocked",
                f"dL/d(output) of {label(call)} ({call.get('cls')}, {site(call)}) is 0 for {pct(grad['zero_frac'])} "
                f"of elements although its outputs are not (mean {num(stats.get('mean'))}, min {num(stats.get('min'))}, "
                f"max {num(stats.get('max'))}): the op right after it kills the gradient, typically a ReLU/clamp "
                "whose input is all negative (dead units: bias/init/input scale). Layers before it get no gradient.",
                f"call {call['id']} --batch 0",
                call=call["id"],
            )
        )
    return out


def _check_params(
    meta: dict[str, Any], steps: list[dict[str, Any]], docs: list[dict[str, Any]]
) -> list[dict]:
    out = []
    if not steps or not steps[0].get("params"):
        if steps:
            out.append(finding("warn", "no-param-updates", "no optimizer parameter updates were recorded"))
        return out
    names = {p["name"] for s in steps for p in s["params"]}
    # layers that ran forward with parameters none of which an optimizer updated
    skipped = {}
    for call in docs[0]["calls"]:
        path = call.get("module_path")
        if (
            call["kind"] != "module"
            or call["phase"] != "forward"
            or call.get("cls") not in PARAM_LAYERS
            or path is None
        ):
            continue
        if not any(_owns(path, n) for n in names):
            skipped.setdefault(path, call)
    for path, call in skipped.items():
        out.append(
            finding(
                "warn",
                "layer-not-optimized",
                f"{path} ({call['cls']}, {site(call)}) runs in forward but none of its parameters are "
                "updated by any optimizer: not passed to the optimizer, or frozen (requires_grad=False). "
                "Fine if intentional (frozen backbone).",
                "params",
                call=call["id"],
            )
        )
    last = steps[-1]["params"]
    no_grad = [p["name"] for p in last if p.get("grad_norm") is None]
    if no_grad:
        out.append(
            finding(
                "warn",
                "param-without-grad",
                f"in the optimizer but got no gradient: {', '.join(no_grad[:12])}"
                + (" …" if len(no_grad) > 12 else "")
                + " (unused in forward, detached, or requires_grad=False).",
                "params",
            )
        )
    zero = [p["name"] for p in last if p.get("grad_norm") == 0]
    if zero:
        out.append(
            finding("warn", "param-zero-grad", f"gradient exactly 0: {', '.join(zero[:12])}", "params")
        )
    ratios: dict[str, list[float]] = {}
    for step in steps:
        for p in step["params"]:
            r = p.get("update_ratio")
            if r is not None and math.isfinite(r):
                ratios.setdefault(p["name"], []).append(r)
    medians = {n: float(np.median(v)) for n, v in ratios.items() if v}
    ndims = {p["name"]: len(p.get("shape") or []) for s in steps for p in s["params"]}
    # biases and norm gains start near 0, so their ratio is noisy: only flag them when extreme
    big = sorted(
        ((r, n) for n, r in medians.items() if r > (0.1 if ndims.get(n, 2) >= 2 else 1.0)), reverse=True
    )
    if big:
        out.append(
            finding(
                "warn",
                "update-too-large",
                "‖Δw‖/‖w‖ above 0.1 (rule of thumb ~1e-3): "
                + ", ".join(f"{n} {num(r)}" for r, n in big[:8])
                + ". The learning rate is likely too high for these.",
                "params",
            )
        )
    tiny = sorted((r, n) for n, r in medians.items() if 0 < r < 1e-6)
    if tiny:
        out.append(
            finding(
                "info",
                "update-tiny",
                "‖Δw‖/‖w‖ below 1e-6: "
                + ", ".join(f"{n} {num(r)}" for r, n in tiny[:8])
                + " (barely learning).",
                "params",
            )
        )
    # weight matrices/kernels only: a bias feeding a BatchNorm legitimately gets a ~0 gradient
    grads = {
        p["name"]: p["grad_norm"]
        for p in steps[0]["params"]
        if p.get("grad_norm") and len(p.get("shape") or []) >= 2
    }
    if len(grads) >= 2:
        hi, lo = max(grads.items(), key=lambda kv: kv[1]), min(grads.items(), key=lambda kv: kv[1])
        if lo[1] > 0 and hi[1] / lo[1] > 1e4:
            out.append(
                finding(
                    "warn",
                    "grad-scale-spread",
                    f"gradient norms span {num(hi[1] / lo[1])}x across parameters ({hi[0]} {num(hi[1])} vs "
                    f"{lo[0]} {num(lo[1])}): vanishing/exploding gradients through depth.",
                    "params --step 0",
                )
            )
    return out


def _owns(path: str, name: str) -> bool:
    """Does module ``path`` own parameter ``name`` directly? Model 0's params are unprefixed even when
    its paths carry ``0:``; other models' params are ``k:name`` and their root path is ``k:``."""
    path = path[2:] if path.startswith("0:") else path
    if path == "":
        return "." not in name and ":" not in name
    if path.endswith(":"):
        return name.startswith(path) and "." not in name[len(path) :]
    return name.startswith(f"{path}.") and "." not in name[len(path) + 1 :]


def _check_stale_grads(
    reader: Reader, run_id: str, steps: list[dict[str, Any]], docs: list[dict[str, Any]]
) -> list:
    """Exact check: for the last nn.Linear, recompute dL/dW from this step's captured input and output
    gradient and compare with the .grad the optimizer saw. A mismatch means .grad held more than this
    step's batches (missing zero_grad) or was rescaled (clipping, loss scaling)."""
    by_index = {d["index"]: d for d in docs}
    for step in steps[1:]:  # step 0 cannot carry anything over
        params = {p["name"]: p for p in step.get("params", [])}
        batch_docs = [by_index[b] for b in step.get("batches", []) if b in by_index]
        if not batch_docs:
            continue
        linears = [
            c
            for c in batch_docs[-1]["calls"]
            if c["kind"] == "module"
            and c.get("cls") == "Linear"
            and f"{c.get('module_path')}.weight" in params
        ]
        if not linears:
            return []
        path = linears[-1]["module_path"]
        record = params[f"{path}.weight"]
        if not record.get("grad"):
            return []
        expected = None
        for doc in batch_docs:
            for call in doc["calls"]:
                if call["kind"] != "module" or call.get("module_path") != path or call["phase"] != "forward":
                    continue
                x_ref, y_ref = first_tensor(call["inputs"]), first_tensor(call["outputs"])
                x_meta, y_meta = tmeta(doc, x_ref), tmeta(doc, y_ref)
                if (
                    not x_meta
                    or not y_meta
                    or x_meta.get("stored") != "full"
                    or y_meta.get("grad_stored") != "full"
                ):
                    return []
                x = reader.array(run_id, doc["index"], x_ref["tid"])
                g = reader.array(run_id, doc["index"], y_ref["tid"], grad=True)
                if x is None or g is None:
                    return []
                part = g.reshape(-1, g.shape[-1]).astype(np.float64).T @ x.reshape(-1, x.shape[-1]).astype(
                    np.float64
                )
                expected = part if expected is None else expected + part
        actual = reader.array(run_id, record["grad"]["batch"], record["grad"]["tid"])
        if expected is None or actual is None or actual.shape != expected.shape:
            return []
        actual = actual.astype(np.float64)
        denom = np.linalg.norm(expected) or 1.0
        rel = float(np.linalg.norm(actual - expected) / denom)
        if rel < 1e-2:
            continue
        cos = float((actual * expected).sum() / ((np.linalg.norm(actual) * denom) or 1.0))
        ratio = float(np.linalg.norm(actual) / denom)
        if cos > 0.999:
            return [
                finding(
                    "info",
                    "grad-rescaled",
                    f"step {step['index']}: {path}.weight.grad is this step's gradient scaled by {num(ratio)} "
                    "(gradient clipping or loss scaling between backward and optimizer.step).",
                    f"params --step {step['index']}",
                )
            ]
        return [
            finding(
                "error",
                "stale-gradients",
                f"step {step['index']}: {path}.weight.grad does not match the gradient of this step's batches "
                f"(relative error {num(rel)}, ‖grad‖ is {num(ratio)}x the expected). Gradients from earlier "
                "steps are still in .grad: optimizer.zero_grad() is missing or in the wrong place.",
                f"params --step {step['index']}",
            )
        ]
    return []


def _check_labels(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    targets_seen, preds = set(), set()
    for doc in docs:
        samples = doc.get("samples") or []
        ids = [(s.get("target") or {}).get("id") for s in samples]
        ids = [i for i in ids if i is not None]
        if len(ids) >= 4 and len(set(ids)) == 1:
            out.append(
                finding(
                    "warn",
                    "single-label-batch",
                    f"batch {doc['index']}: every target is {ids[0]}",
                    f"samples --batch {doc['index']}",
                )
            )
        targets_seen.update(ids)
        preds.update(
            (s.get("prediction") or {}).get("id")
            for s in samples
            if (s.get("prediction") or {}).get("id") is not None
        )
        indices = doc.get("indices") or []
        if len(indices) != len(set(indices)):
            out.append(
                finding(
                    "warn",
                    "duplicate-indices",
                    f"batch {doc['index']}: the same dataset index appears twice",
                    f"data --batch {doc['index']}",
                )
            )
    if len(preds) == 1 and len(targets_seen) > 1:
        out.append(
            finding(
                "info",
                "single-prediction",
                f"the model predicts class {next(iter(preds))} for every captured sample "
                f"({len(targets_seen)} classes present): normal right after init, a red flag later.",
                "samples",
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def cmd_runs(reader: Reader, args: argparse.Namespace) -> Any:
    runs = reader.runs()
    if args.json:
        return runs
    rows = [
        [
            str(i),
            r["run_id"],
            r.get("status") or "-",
            str(r.get("steps_captured")),
            str(r.get("batches_captured")),
            r.get("script") or "",
            r.get("host") or "",
        ]
        for i, r in enumerate(runs)
    ]
    print(f"runs under {reader.target}")
    print(table(rows, ["#", "run", "status", "steps", "batches", "script", "host"]))
    return None


def summarize(reader: Reader, run_id: str) -> dict[str, Any]:
    info = reader.run(run_id)
    meta = info["meta"]
    docs = reader.batches(run_id)
    return {
        "run": run_id,
        "status": meta.get("status"),
        "script": meta.get("script"),
        "argv": meta.get("argv"),
        "host": meta.get("host"),
        "started_at": meta.get("started_at"),
        "env": meta.get("env"),
        "steps": f"{meta.get('steps_captured')}/{meta.get('steps_requested')}",
        "models": [
            {k: m.get(k) for k in ("index", "cls", "n_params", "trainable", "device", "dtype")}
            for m in meta.get("models") or []
        ],
        "optimizers": [
            {
                "cls": o["cls"],
                "groups": [
                    {k: g.get(k) for k in ("n_params", "lr", "weight_decay", "momentum", "betas") if k in g}
                    for g in o.get("hyper", [])
                ],
            }
            for o in meta.get("optimizers") or []
        ],
        "data": meta.get("data"),
        "batches": [
            {
                "batch": d["index"],
                "step": d.get("step"),
                "size": d.get("batch_size"),
                "loss": batch_loss(d),
                "accuracy": batch_accuracy(d),
                "flags": d.get("flags") or [],
                "device_ms": d.get("timing", {}).get("phases_ms"),
            }
            for d in docs
        ],
        "health": health(reader, run_id),
    }


def cmd_summary(reader: Reader, args: argparse.Namespace) -> Any:
    run_id = reader.pick(args.run)
    s = summarize(reader, run_id)
    if args.json:
        return s
    print(f"run      {s['run']}  [{s['status']}]  steps {s['steps']}  on {s['host']}  {s['started_at']}")
    print(f"script   {s['script']} {' '.join(s['argv'] or [])}")
    env = s["env"] or {}
    print(
        f"env      python {env.get('python')}  torch {env.get('torch')}  cuda {env.get('cuda')}  {env.get('device') or ''}"
    )
    for m in s["models"]:
        print(
            f"model {m['index']}  {m['cls']}  {num(m['n_params'])} params ({num(m['trainable'])} trainable)  {m['device']} {m['dtype']}"
        )
    for o in s["optimizers"]:
        groups = "; ".join(", ".join(f"{k}={v}" for k, v in g.items()) for g in o["groups"])
        print(f"optim    {o['cls']}  {groups}")
    data = s["data"] or {}
    if data:
        ds, loader = data.get("dataset") or {}, data.get("loader") or {}
        print(
            f"data     {ds.get('cls')} len={ds.get('len')}  batch_size={loader.get('batch_size')} sampler={loader.get('sampler')}"
        )
        if ds.get("classes"):
            print(f"         classes {ds['classes'][:20]}")
        if data.get("normalize"):
            print(f"         normalize {data['normalize']}")
        if data.get("vocab"):
            print(f"         vocab {len(data['vocab'])} tokens")
    print()
    rows = [
        [
            str(b["batch"]),
            str(b["step"]),
            str(b["size"]),
            b["loss"],
            pct(b["accuracy"]),
            " ".join(b["flags"]),
            " ".join(f"{k}={v:.1f}" for k, v in (b["device_ms"] or {}).items()),
        ]
        for b in s["batches"]
    ]
    print(table(rows, ["batch", "step", "size", "loss", "acc", "flags", "time ms (during capture)"]))
    print()
    print_findings(s["health"], target_args(args), run_id)
    return None


def target_args(args: argparse.Namespace) -> str:
    """The TARGET + connection flags to repeat in suggested follow-up commands."""
    import shlex

    extra = "".join(f" --via {shlex.quote(h)}" for h in args.via)
    if args.remote_tvis:
        extra += f" --remote-tvis {shlex.quote(args.remote_tvis)}"
    return shlex.quote(args.target) + extra


def print_findings(findings: list[dict[str, Any]], target: str, run_id: str) -> None:
    if not findings:
        print("health   no findings")
        return
    print(f"health   {len(findings)} finding(s)")
    for f in findings:
        print(f"  [{f['severity']}] {f['check']}: {f['message']}")
        if f.get("next"):
            command, _, rest = f["next"].partition(" ")
            print(f"      next: tvis_debug.py {command} {target} --run {run_id} {rest}".rstrip())


def cmd_health(reader: Reader, args: argparse.Namespace) -> Any:
    run_id = reader.pick(args.run)
    findings = health(reader, run_id)
    if args.json:
        return findings
    print_findings(findings, target_args(args), run_id)
    return None


def cmd_layers(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    doc = reader.batch(run_id, args.batch)
    kinds = set(args.kinds.split(","))
    rows = []
    for call in doc["calls"]:
        if call["phase"] not in ("forward", "loss") or call["kind"] not in kinds:
            continue
        if args.grep and args.grep not in label(call):
            continue
        if args.depth is not None and call.get("depth", 0) > args.depth:
            continue
        meta = tmeta(doc, first_tensor(call["outputs"]))
        st = (meta or {}).get("stats") or {}
        gs = (meta or {}).get("grad_stats") or {}
        bwd = call.get("bwd_ms")
        rows.append(
            [
                str(call["id"]),
                "  " * call.get("depth", 0) + label(call),
                call.get("cls") or call["kind"],
                str((meta or {}).get("shape", "")),
                st.get("mean"),
                st.get("std"),
                pct(st.get("zero_frac")),
                ("NaN/Inf" if st.get("nan") or st.get("inf") else ""),
                gs.get("norm"),
                ("NaN/Inf" if gs.get("nan") or gs.get("inf") else ""),
                call.get("ms"),
                (num(bwd) + ("~" if call.get("bwd_approx") else "")) if bwd is not None else "-",
                " ".join(call.get("flags") or []),
            ]
        )
    print(
        f"batch {doc['index']} (step {doc.get('step')}): forward calls, output value stats and ‖dL/d(output)‖"
    )
    print(
        table(
            rows,
            [
                "id",
                "call",
                "type",
                "out shape",
                "mean",
                "std",
                "zeros",
                "bad",
                "‖grad‖",
                "bad grad",
                "fwd ms",
                "bwd ms",
                "flags",
            ],
        )
    )
    print("(~ = backward time inferred from gradient arrival; times are measured during capture)")


def cmd_params(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    steps = reader.run(run_id)["steps"]
    rows = []
    for step in steps:
        if args.step is not None and step["index"] != args.step:
            continue
        for p in step["params"]:
            if args.grep and args.grep not in p["name"]:
                continue
            rows.append(
                [
                    str(step["index"]),
                    p["name"],
                    str(p["shape"]),
                    p.get("weight_norm"),
                    p.get("grad_norm"),
                    p.get("update_norm"),
                    p.get("update_ratio"),
                ]
            )
    print("parameter updates per optimizer step (‖Δw‖/‖w‖ rule of thumb: ~1e-3)")
    print(table(rows, ["step", "param", "shape", "‖w‖", "‖grad‖", "‖Δw‖", "‖Δw‖/‖w‖"]))


def cmd_samples(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    docs = [reader.batch(run_id, args.batch)] if args.batch is not None else reader.batches(run_id)
    rows = []
    for doc in docs:
        for s in doc.get("samples") or []:
            target, pred = s.get("target") or {}, s.get("prediction") or {}
            t = target.get("name") if target.get("name") is not None else target.get("text", target.get("id"))
            p = pred.get("name") if pred.get("name") is not None else pred.get("text", pred.get("id"))
            correct = pred.get("correct")
            if correct is None and pred.get("token_accuracy") is not None:
                correct = pct(pred["token_accuracy"])
            raw = (s.get("raw") or {}).get("value") or {}
            raw_desc = raw.get("text") if raw.get("kind") == "text" else raw.get("kind", "")
            rows.append(
                [
                    str(doc["index"]),
                    str(s["position"]),
                    str(s.get("index")),
                    str(t),
                    str(p),
                    pred.get("p_target"),
                    num(correct),
                    s.get("loss"),
                    str(raw_desc or "")[:40],
                ]
            )
    rows.sort(key=lambda r: -(r[7] if isinstance(r[7], float) else -1))
    print("per-sample results, worst loss first (p(target) and correct come from the decoded output)")
    print(
        table(
            rows[: args.top],
            ["batch", "pos", "index", "target", "prediction", "p(target)", "correct", "loss", "raw"],
        )
    )


def cmd_data(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    meta = reader.run(run_id)["meta"]
    doc = reader.batch(run_id, args.batch)
    print(f"data config: {json.dumps({k: v for k, v in (meta.get('data') or {}).items() if k != 'vocab'})}")
    print(f"batch {doc['index']}: size {doc.get('batch_size')}  dataset indices {doc.get('indices')}")
    calls = {c["id"]: c for c in doc["calls"]}
    children: dict[int, list[dict[str, Any]]] = {}
    for c in doc["calls"]:
        children.setdefault(c["parent"], []).append(c)
    samples = doc.get("samples") or []
    for s in samples[: args.samples]:
        print(f"\nsample pos {s['position']} (dataset index {s.get('index')}) target={s.get('target')}")
        raw = (s.get("raw") or {}).get("value") or {}
        if raw.get("kind") == "text":
            print(f"  raw text: {raw.get('text')!r}")
        elif raw:
            print(f"  raw: {raw.get('kind')} {raw.get('size') or ''}")
        if s.get("tokens"):
            print(f"  tokens: {s['tokens'].get('tokens')}")

        def walk(call_id: int, depth: int) -> None:
            for child in children.get(call_id, []):
                out = tmeta(doc, first_tensor(child["outputs"]))
                value = child["outputs"][0]["value"] if child["outputs"] else {}
                desc = tensor_line(out) if out else _value_brief(value)
                print(f"  {'  ' * depth}{child['kind']:<9} {label(child):<24} {desc}")
                walk(child["id"], depth + 1)

        if s.get("sample_call") in calls:
            walk(s["sample_call"], 0)
        inp = doc["tensors"].get((s.get("input") or {}).get("tid", ""))
        if inp:
            print(f"  collated batch input: {tensor_line(inp)}")


def _value_brief(value: dict[str, Any]) -> str:
    kind = value.get("kind")
    if kind == "image":
        return f"image {value.get('size')} {value.get('mode') or ''}".rstrip()
    if kind == "text":
        return repr(value.get("text"))[:80]
    return json.dumps(value, default=str)[:80] if value else ""


def cmd_call(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    doc = reader.batch(run_id, args.batch)
    call = next((c for c in doc["calls"] if c["id"] == args.id), None)
    if call is None:
        sys.exit(f"tvis_debug: no call {args.id} in batch {args.batch}")
    print(
        f"call {call['id']}  {label(call)}  kind={call['kind']} phase={call['phase']} cls={call.get('cls')}  at {site(call)}"
    )
    defined = f"{call.get('file')}:{call.get('line')}" if call.get("file") else ""
    bwd = num(call.get("bwd_ms")) + (" (approx)" if call.get("bwd_approx") else "")
    print(
        f"  {'defined ' + defined + '  ' if defined else ''}fwd {num(call.get('ms'))} ms  bwd {bwd} ms  flags {call.get('flags') or []}"
    )
    extra = {
        k: v
        for k, v in (call.get("extra") or {}).items()
        if not isinstance(v, dict) or v.get("kind") != "tensor"
    }
    if extra:
        print(f"  extra {json.dumps(extra, default=str)[:400]}")
    for role in ("inputs", "outputs"):
        for entry in call[role]:
            value = entry["value"]
            refs = tensors_in(value)
            if not refs:
                print(f"  {role[:-1]:<6} {entry['name']}: {json.dumps(value, default=str)[:200]}")
            for ref in refs:
                meta = tmeta(doc, ref)
                print(f"  {role[:-1]:<6} {entry['name']}: value {tensor_line(meta)}")
                if meta and meta.get("grad_stats"):
                    print(f"  {'':<6} {'':<{len(entry['name'])}}  grad  {stats_line(meta['grad_stats'])}")
    kids = [c for c in doc["calls"] if c["parent"] == call["id"]]
    if kids:
        print("  children: " + ", ".join(f"{c['id']} {label(c)}" for c in kids[:30]))
    ops = [o for o in doc.get("ops") or [] if o.get("call") == call["id"]]
    for o in ops[:30]:
        print(f"  op  {o['file']}:{o['line']}  {o['op']} → {o.get('shapes')} {o.get('dtype')}")
    s = call.get("call_site") or {}
    file, line = call.get("file") or s.get("file"), call.get("line") or s.get("line")
    if file and line:
        try:
            text = reader.request("source", run=run_id, path=file)["text"].splitlines()
        except ApiError:
            return
        print(f"  source {file}:")
        for n in range(max(1, line - 2), min(len(text), line + 8) + 1):
            print(f"  {n:>5}{'>' if n == line else ' '} {text[n - 1]}")


def cmd_tensor(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    doc = reader.batch(run_id, args.batch)
    meta = doc["tensors"].get(args.tid)
    if meta is None:
        sys.exit(f"tvis_debug: no tensor {args.tid} in batch {args.batch}")
    print(("gradient of " if args.grad else "") + tensor_line(meta, grad=args.grad))
    if meta.get("batch_dim"):
        print(f"batch_dim {meta['batch_dim']}")
    fixed = [int(v) for v in args.fixed.split(",")] if args.fixed else None
    try:
        view = reader.request(
            "tensor",
            run=run_id,
            batch=args.batch,
            tid=args.tid,
            grad=args.grad,
            sample=args.sample,
            fixed=fixed,
        )
    except ApiError as exc:
        sys.exit(f"tvis_debug: {exc}")
    values = np.frombuffer(base64.b64decode(view["data"]), dtype="<f4").reshape(view["rows"], view["cols"])
    print(
        f"window: leading dims {view['leading']} fixed at {view['fixed']}, stride {view['stride']}, showing ≤{args.max}x{args.max}"
    )
    with np.printoptions(precision=4, suppress=False, linewidth=160):
        print(values[: args.max, : args.max])


def cmd_source(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    try:
        text = reader.request("source", run=run_id, path=args.file)["text"].splitlines()
    except ApiError:
        sys.exit(
            f"tvis_debug: no source snapshot {args.file!r}; snapshots: {reader.run(run_id)['meta'].get('sources')}"
        )
    shapes: dict[int, list[str]] = {}
    for o in reader.batch(run_id, args.batch).get("ops") or []:
        if o["file"] == args.file:
            shapes.setdefault(o["line"], []).append(f"{o['op']}→{o.get('shapes')}")
    lo, hi = 1, len(text)
    if args.line:
        lo, hi = max(1, args.line - args.context), min(len(text), args.line + args.context)
    for n in range(lo, hi + 1):
        note = f"   # {'; '.join(dict.fromkeys(shapes[n]))}" if n in shapes else ""
        print(f"{n:>5}{'>' if n == args.line else ' '} {text[n - 1]}{note}")


def cmd_timing(reader: Reader, args: argparse.Namespace) -> None:
    run_id = reader.pick(args.run)
    meta = reader.run(run_id)["meta"]
    docs = [reader.batch(run_id, args.batch)] if args.batch is not None else reader.batches(run_id)
    print(f"clock: {meta.get('clock')} (cuda events on GPU; tvis overhead removed; wall = during capture)")
    rows = []
    for d in docs:
        t = d.get("timing") or {}
        rows.append(
            [
                str(d["index"]),
                t.get("total_ms"),
                t.get("wall_total_ms"),
                " ".join(f"{k}={v:.2f}" for k, v in (t.get("phases_ms") or {}).items()),
            ]
        )
    print(table(rows, ["batch", "device ms", "wall ms", "phases (device ms)"]))
    calls = [
        (c, d["index"])
        for d in docs
        for c in d["calls"]
        if c["kind"] in ("module", "function", "loss") and c["parent"] is not None
    ]
    for key, title in (("ms", "forward"), ("bwd_ms", "backward")):
        top = sorted((x for x in calls if x[0].get(key) is not None), key=lambda x: -x[0][key])[: args.top]
        print(f"\nslowest {title} calls")
        print(
            table(
                [
                    [
                        str(b),
                        str(c["id"]),
                        label(c),
                        c.get("cls") or c["kind"],
                        c[key],
                        "approx" if key == "bwd_ms" and c.get("bwd_approx") else "",
                    ]
                    for c, b in top
                ],
                ["batch", "id", "call", "type", "ms", ""],
            )
        )


def cmd_compare(reader: Reader, args: argparse.Namespace) -> Any:
    a, b = reader.pick(args.run), reader.pick(args.against)
    sa, sb = summarize(reader, a), summarize(reader, b)
    if args.json:
        return {"a": sa, "b": sb}
    print(f"A = {a} [{sa['status']}]  {sa['script']} {' '.join(sa['argv'] or [])}")
    print(f"B = {b} [{sb['status']}]  {sb['script']} {' '.join(sb['argv'] or [])}")
    n = max(len(sa["batches"]), len(sb["batches"]))
    rows = []
    for i in range(n):
        ra = sa["batches"][i] if i < len(sa["batches"]) else {}
        rb = sb["batches"][i] if i < len(sb["batches"]) else {}
        rows.append(
            [str(i), ra.get("loss"), rb.get("loss"), pct(ra.get("accuracy")), pct(rb.get("accuracy"))]
        )
    print(table(rows, ["batch", "loss A", "loss B", "acc A", "acc B"]))
    pa = {p["name"]: p for p in (reader.run(a)["steps"] or [{}])[-1].get("params", [])}
    pb = {p["name"]: p for p in (reader.run(b)["steps"] or [{}])[-1].get("params", [])}
    rows = [
        [
            n,
            (pa.get(n) or {}).get("grad_norm"),
            (pb.get(n) or {}).get("grad_norm"),
            (pa.get(n) or {}).get("update_ratio"),
            (pb.get(n) or {}).get("update_ratio"),
        ]
        for n in sorted(set(pa) | set(pb))
    ]
    print("\nlast step parameters")
    print(table(rows, ["param", "‖grad‖ A", "‖grad‖ B", "Δw/w A", "Δw/w B"]))
    ca = {f["check"] for f in sa["health"]}
    cb = {f["check"] for f in sb["health"]}
    print(
        f"\nhealth  fixed in B: {sorted(ca - cb) or '-'}   new in B: {sorted(cb - ca) or '-'}   both: {sorted(ca & cb) or '-'}"
    )
    return None


def viewer_fragment(reader: Reader, args: argparse.Namespace) -> str:
    from urllib.parse import urlencode

    params: dict[str, Any] = {
        "run": reader.pick(args.run),
        "page": args.page,
        "mode": "batch",
        "b": args.batch,
    }
    if args.page == "explore":
        params["tab"] = args.tab
    if args.page == "insights" and args.insight:
        params["insight"] = args.insight
    if args.call is not None:
        params["call"] = args.call
    if args.sample is not None:
        params["sample"] = args.sample
    return f"#{urlencode(params)}"


def cmd_link(reader: Reader, args: argparse.Namespace) -> None:
    print(f"{args.viewer.rstrip('/')}/{viewer_fragment(reader, args)}")


CHROME_CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def find_chrome() -> str | None:
    import os
    import shutil

    for candidate in [os.environ.get("TVIS_CHROME"), *CHROME_CANDIDATES]:
        if candidate and (shutil.which(candidate) or Path(candidate).is_file()):
            return shutil.which(candidate) or candidate
    return None


def cmd_shot(reader: Reader, args: argparse.Namespace) -> None:
    """Serve the viewer in-process on a free 127.0.0.1 port, screenshot one view with headless Chrome."""
    import subprocess
    import tempfile
    import time

    from tvis.viewer.server import ViewerServer

    chrome = find_chrome()
    if chrome is None:
        sys.exit("tvis_debug: no Chrome/Chromium found; set TVIS_CHROME=/path/to/chrome")
    url = None
    server = ViewerServer(reader.backend, port=0)
    server.serve_in_background()
    out = Path(args.out).expanduser().resolve()
    out.unlink(missing_ok=True)
    try:
        url = f"{server.url}{viewer_fragment(reader, args)}"
        with tempfile.TemporaryDirectory(prefix="tvis-chrome-") as profile:
            command = [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--user-data-dir={profile}",
                f"--window-size={args.size}",
                f"--virtual-time-budget={args.wait_ms}",
                f"--screenshot={out}",
                url,
            ]
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # headless Chrome often keeps running after writing the file: wait for the PNG, then stop it
            deadline = time.monotonic() + args.timeout
            size = -1
            while time.monotonic() < deadline and proc.poll() is None:
                if out.is_file() and out.stat().st_size == size and size > 0:
                    break
                size = out.stat().st_size if out.is_file() else -1
                time.sleep(0.5)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    finally:
        server.shutdown()
        server.server_close()
    if not out.is_file():
        sys.exit(f"tvis_debug: Chrome wrote no screenshot for {url}")
    print(f"{out}  ({url})")


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvis_debug.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Any, help: str, batch: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("target", help="runs dir, run dir, HOST:PATH, or saved target (as for `tvis open`)")
        p.add_argument("--run", help="run id, unique part of one, or index (0 = newest; default newest)")
        p.add_argument("--via", action="append", default=[], help="ssh jump host (repeatable)")
        p.add_argument("--remote-tvis", default=None, help="tvis executable on the remote host")
        p.add_argument(
            "--json", action="store_true", help="machine-readable output (runs/summary/health/compare)"
        )
        if batch:
            p.add_argument("--batch", type=int, default=0, help="batch index (default 0)")
        p.set_defaults(fn=fn)
        return p

    add("runs", cmd_runs, "list recordings")
    add("summary", cmd_summary, "overview + health findings")
    add("health", cmd_health, "automated checks for common training bugs")
    p = add("layers", cmd_layers, "forward calls with output/gradient stats", batch=True)
    p.add_argument("--kinds", default="module,function,loss", help="call kinds to list")
    p.add_argument("--depth", type=int, default=None, help="max tree depth")
    p.add_argument("--grep", default=None, help="only calls whose name contains this")
    p = add("params", cmd_params, "parameter weight/grad/update per step")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--grep", default=None)
    p = add("samples", cmd_samples, "per-sample target/prediction/loss")
    p.add_argument("--batch", type=int, default=None, help="one batch (default: all)")
    p.add_argument("--top", type=int, default=20)
    p = add("data", cmd_data, "data pipeline per sample", batch=True)
    p.add_argument("--samples", type=int, default=2, help="how many samples to walk")
    p = add("call", cmd_call, "one call in detail", batch=True)
    p.add_argument("id", type=int)
    p = add("tensor", cmd_tensor, "one tensor's stats and values", batch=True)
    p.add_argument("tid")
    p.add_argument("--grad", action="store_true", help="the gradient instead of the value")
    p.add_argument("--sample", type=int, default=None, help="slice to one sample")
    p.add_argument("--fixed", default=None, help="indices for leading dims, e.g. 0,3")
    p.add_argument("--max", type=int, default=12, help="max rows/cols to print")
    p = add("source", cmd_source, "source snapshot with op shapes", batch=True)
    p.add_argument("file", help="path relative to the project, as in `summary` sources / call sites")
    p.add_argument("--line", type=int, default=None)
    p.add_argument("--context", type=int, default=12)
    p = add("timing", cmd_timing, "phase times and slowest calls")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    p = add("compare", cmd_compare, "two runs side by side")
    p.add_argument("--against", required=True, help="the other run (id, part, or index)")
    for name, fn, help in (
        ("link", cmd_link, "viewer URL for a run/page/batch/call/sample"),
        ("shot", cmd_shot, "PNG screenshot of a viewer page (serves the viewer itself, headless Chrome)"),
    ):
        p = add(name, fn, help, batch=True)
        p.add_argument(
            "--page", default="replay", choices=["replay", "insights", "loss", "weights", "explore"]
        )
        p.add_argument(
            "--tab", default="overview", help="Trace tab: overview|samples|timeline|source|layers|step"
        )
        p.add_argument("--insight", default=None, help="Insights view: attention|gradcam|embeddings")
        p.add_argument("--call", type=int, default=None, help="select this call (Trace inspector)")
        p.add_argument("--sample", type=int, default=None, help="follow this sample position")
        if name == "link":
            p.add_argument(
                "--viewer", default="http://127.0.0.1:8765/", help="base URL printed by `tvis open`"
            )
        else:
            p.add_argument("--out", default="tvis_shot.png", help="PNG path (default ./tvis_shot.png)")
            p.add_argument("--size", default="1400,900", help="window size W,H")
            p.add_argument("--wait-ms", type=int, default=8000, help="virtual time for the page to load")
            p.add_argument("--timeout", type=float, default=60, help="seconds before giving up")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reader = Reader(args.target, args.via, args.remote_tvis)
    try:
        result = args.fn(reader, args)
        if result is not None:
            print(json.dumps(result, indent=1, default=str))
    except ApiError as exc:
        print(f"tvis_debug: {exc}", file=sys.stderr)
        return 1
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
