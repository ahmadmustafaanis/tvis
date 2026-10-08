"""On-disk layout of a tvis run directory.

```
<run>/
  meta.json                 run-level info (script, env, models, data pipeline, notices, status)
  steps.json                per optimizer step: batches, timing, parameter updates
  batches/<k>/batch.json    calls, tensor metadata, ops, samples, timing for batch k
  batches/<k>/arrays/*.npy  captured tensor values ("<tid>.npy") and gradients ("<tid>.grad.npy")
  source/<relpath>          snapshot of every project file that executed during capture
```

Shared by the capture side (writer) and the viewer side (reader); stdlib only.
"""

from __future__ import annotations

from pathlib import PurePosixPath

FORMAT_VERSION = 1

META_FILE = "meta.json"
STEPS_FILE = "steps.json"
BATCHES_DIR = "batches"
BATCH_FILE = "batch.json"
ARRAYS_DIR = "arrays"
SOURCE_DIR = "source"

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"


def batch_dir_name(index: int) -> str:
    return f"{index:04d}"


def value_array_name(tid: str) -> str:
    return f"{tid}.npy"


def grad_array_name(tid: str) -> str:
    return f"{tid}.grad.npy"


def safe_relpath(relpath: str) -> PurePosixPath:
    """Validate a relative path taken from a request so it cannot escape its directory."""
    path = PurePosixPath(relpath)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts) or not path.parts:
        raise ValueError(f"invalid relative path: {relpath!r}")
    return path
