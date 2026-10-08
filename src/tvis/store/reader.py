"""Viewer-side reader for run directories. Depends on numpy + stdlib only (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tvis.store import schema


class RunNotFound(LookupError):
    pass


class RunsRoot:
    """A directory containing run directories (or a single run directory itself)."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for run_dir in self._run_dirs():
            try:
                meta = Run(run_dir).meta()
            except (OSError, ValueError):
                continue
            runs.append(
                {
                    "run_id": run_dir.name,
                    "script": meta.get("script"),
                    "started_at": meta.get("started_at"),
                    "status": meta.get("status"),
                    "host": meta.get("host"),
                    "steps_captured": meta.get("steps_captured", 0),
                    "batches_captured": meta.get("batches_captured", 0),
                }
            )
        runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
        return runs

    def run(self, run_id: str) -> Run:
        for run_dir in self._run_dirs():
            if run_dir.name == run_id:
                return Run(run_dir)
        raise RunNotFound(run_id)

    def _run_dirs(self) -> list[Path]:
        if (self.root / schema.META_FILE).is_file():
            return [self.root]
        if not self.root.is_dir():
            return []
        return [p for p in self.root.iterdir() if (p / schema.META_FILE).is_file()]


class Run:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def meta(self) -> dict[str, Any]:
        return _read_json(self.run_dir / schema.META_FILE)

    def steps(self) -> list[dict[str, Any]]:
        path = self.run_dir / schema.STEPS_FILE
        return _read_json(path) if path.is_file() else []

    def batch(self, index: int) -> dict[str, Any]:
        path = self._batch_dir(index) / schema.BATCH_FILE
        if not path.is_file():
            raise RunNotFound(f"batch {index}")
        return _read_json(path)

    def array(self, batch_index: int, tid: str, grad: bool = False) -> np.ndarray:
        name = schema.grad_array_name(tid) if grad else schema.value_array_name(tid)
        path = self._batch_dir(batch_index) / schema.ARRAYS_DIR / schema.safe_relpath(name)
        if not path.is_file():
            raise RunNotFound(f"array {name} in batch {batch_index}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def source(self, relpath: str) -> str:
        path = self.run_dir / schema.SOURCE_DIR / schema.safe_relpath(relpath)
        if not path.is_file():
            raise RunNotFound(f"source {relpath}")
        return path.read_text(encoding="utf-8")

    def _batch_dir(self, index: int) -> Path:
        return self.run_dir / schema.BATCHES_DIR / schema.batch_dir_name(int(index))


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
