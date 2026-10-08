"""Which source files belong to the user's project, and snapshotting them as they execute."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

_EXCLUDED_PARTS = {"site-packages", "dist-packages", ".venv", "venv", ".tox", "__pypackages__", ".git"}
_TVIS_DIR = str(Path(__file__).resolve().parents[1]) + os.sep


class ProjectFiles:
    def __init__(self, project_dir: Path, on_new_file: Callable[[str, str], None] | None = None):
        self.root = Path(project_dir).resolve()
        self._root_prefix = str(self.root) + os.sep
        self._cache: dict[str, str | None] = {}
        self._on_new_file = on_new_file
        self._env_prefixes = tuple(
            {str(Path(p).resolve()) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
        )

    def relpath(self, filename: str) -> str | None:
        """Project-relative POSIX path for files in the project, else None."""
        try:
            return self._cache[filename]
        except KeyError:
            pass
        rel = self._classify(filename)
        self._cache[filename] = rel
        if rel is not None and self._on_new_file is not None:
            try:
                text = Path(filename).read_text(encoding="utf-8")
            except OSError:
                text = None
            if text is not None:
                self._on_new_file(rel, text)
        return rel

    def _classify(self, filename: str) -> str | None:
        if not filename or filename.startswith("<"):
            return None
        try:
            path = str(Path(filename).resolve())
        except OSError:
            return None
        if not path.startswith(self._root_prefix) or path.startswith(_TVIS_DIR):
            return None
        # a virtualenv inside the project dir is still library code
        if any(path.startswith(p) for p in self._env_prefixes if p.startswith(self._root_prefix)):
            return None
        rel = Path(path).relative_to(self.root)
        if any(part in _EXCLUDED_PARTS for part in rel.parts):
            return None
        return rel.as_posix()


def is_tvis_file(filename: str) -> bool:
    return filename.startswith(_TVIS_DIR)
