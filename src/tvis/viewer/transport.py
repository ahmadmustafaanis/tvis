"""Backends the HTTP server talks to: in-process (local runs) or a JSON-lines stdio agent (remote
runs, reached over SSH).

Protocol (one JSON object per line):
  request  {"id": 7, "method": "batch", "params": {"run": "...", "batch": 0}}
  response {"id": 7, "result": {...}}  or  {"id": 7, "error": "message", "status": 404}
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import IO, Any, Protocol

from tvis.viewer.api import Api, ApiError


class Backend(Protocol):
    def request(self, method: str, params: dict[str, Any]) -> Any: ...

    def close(self) -> None: ...

    def describe(self) -> str: ...


class LocalBackend:
    def __init__(self, root: Path):
        self.api = Api(root)
        self.root = Path(root)

    def request(self, method: str, params: dict[str, Any]) -> Any:
        return self.api.dispatch(method, params)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return str(self.root)


class StdioBackend:
    """Talks to `tvis agent --stdio` running as a subprocess (normally `ssh host tvis agent ...`).

    Requests may be issued concurrently from the HTTP server's threads; responses are matched by id.
    """

    def __init__(self, command: list[str], description: str, timeout: float = 120.0):
        self.command = command
        self._description = description
        self.timeout = timeout
        self._ids = itertools.count(1)
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # ssh prompts (password/2FA) and agent warnings go to the user's terminal
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, name="tvis-stdio-reader", daemon=True)
        self._reader.start()

    def describe(self) -> str:
        return self._description

    def request(self, method: str, params: dict[str, Any]) -> Any:
        if self._closed or self.process.poll() is not None:
            raise ApiError(f"connection to {self._description} is closed", status=502)
        request_id = next(self._ids)
        future: Future = Future()
        with self._lock:
            self._pending[request_id] = future
        line = json.dumps({"id": request_id, "method": method, "params": params}) + "\n"
        try:
            with self._write_lock:
                assert self.process.stdin is not None
                self.process.stdin.write(line)
                self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self._pending.pop(request_id, None)
            raise ApiError(f"connection to {self._description} lost: {exc}", status=502) from exc
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError as exc:
            raise ApiError(
                f"{self._description} did not answer within {self.timeout:.0f}s", status=504
            ) from exc
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def close(self) -> None:
        self._closed = True
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print(f"tvis: unexpected output from agent: {raw.rstrip()}", file=sys.stderr)
                continue
            with self._lock:
                future = self._pending.get(message.get("id"))
            if future is None:
                continue
            if "error" in message:
                future.set_exception(ApiError(message["error"], status=message.get("status", 500)))
            else:
                future.set_result(message.get("result"))
        error = ApiError(f"connection to {self._description} closed", status=502)
        with self._lock:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)


def serve_stdio(root: Path, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
    """The remote side: answer JSON-lines requests on stdin until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    api = Api(root)
    for raw in stdin:
        if not raw.strip():
            continue
        request_id = None
        try:
            request = json.loads(raw)
            request_id = request.get("id")
            result = api.dispatch(request["method"], request.get("params") or {})
            reply: dict[str, Any] = {"id": request_id, "result": result}
        except ApiError as exc:
            reply = {"id": request_id, "error": str(exc), "status": exc.status}
        except Exception as exc:  # never let one bad request kill the agent
            reply = {"id": request_id, "error": f"{type(exc).__name__}: {exc}", "status": 500}
        stdout.write(json.dumps(reply, allow_nan=False) + "\n")
        stdout.flush()
