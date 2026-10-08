"""Capture-side writer for a run directory.

JSON documents are written atomically (temp file + rename) so a viewer polling a running capture
never reads a half-written file. Arrays are written on a background thread so disk latency (slow
shared filesystems on clusters) does not stall the training process.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np

from tvis.store import schema


def write_json_atomic(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(document, f, separators=(",", ":"), allow_nan=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


class RunWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._arrays: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue()
        self._errors: list[str] = []
        self._thread = threading.Thread(target=self._array_worker, name="tvis-writer", daemon=True)
        self._thread.start()

    # -- documents -----------------------------------------------------------------------------
    def write_meta(self, meta: dict[str, Any]) -> None:
        write_json_atomic(self.run_dir / schema.META_FILE, {"format_version": schema.FORMAT_VERSION, **meta})

    def write_steps(self, steps: list[dict[str, Any]]) -> None:
        write_json_atomic(self.run_dir / schema.STEPS_FILE, steps)

    def write_batch(self, index: int, batch: dict[str, Any]) -> None:
        write_json_atomic(self._batch_dir(index) / schema.BATCH_FILE, batch)

    def write_source(self, relpath: str, text: str) -> None:
        path = self.run_dir / schema.SOURCE_DIR / schema.safe_relpath(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # -- arrays --------------------------------------------------------------------------------
    def save_array(self, batch_index: int, name: str, array: np.ndarray) -> None:
        self._arrays.put((self._batch_dir(batch_index) / schema.ARRAYS_DIR / name, array))

    def flush(self) -> list[str]:
        """Block until all queued arrays are on disk. Returns write errors encountered so far."""
        self._arrays.join()
        return list(self._errors)

    def close(self) -> list[str]:
        errors = self.flush()
        self._arrays.put(None)
        self._thread.join()
        return errors

    def _batch_dir(self, index: int) -> Path:
        return self.run_dir / schema.BATCHES_DIR / schema.batch_dir_name(index)

    def _array_worker(self) -> None:
        while True:
            item = self._arrays.get()
            try:
                if item is None:
                    return
                path, array = item
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, array, allow_pickle=False)
            except Exception as exc:  # recorded, surfaced by flush(); never crash training
                self._errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                self._arrays.task_done()
