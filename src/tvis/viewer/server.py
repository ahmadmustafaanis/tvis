"""The local web server behind `tvis open`. Binds to 127.0.0.1 only.

GET  /                 the UI
GET  /static/<file>    UI assets
POST /api              {"method": ..., "params": {...}} → {"result": ...} | {"error": ...}
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from tvis.viewer.api import ApiError
from tvis.viewer.transport import Backend

HOST = "127.0.0.1"
MAX_BODY = 1 << 20


def ui_dir() -> Path:
    return Path(str(resources.files("tvis") / "ui"))


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, backend: Backend, port: int = 0, static_dir: Path | None = None):
        self.backend = backend
        self.static_dir = (static_dir or ui_dir()).resolve()
        super().__init__((HOST, port), _Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    def serve_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="tvis-viewer", daemon=True)
        thread.start()
        return thread


class _Handler(BaseHTTPRequestHandler):
    server: ViewerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the user's terminal quiet

    # -- routing -------------------------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._host_allowed():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(self.server.static_dir / "index.html")
        elif path.startswith("/static/"):
            self._send_static(path.removeprefix("/static/"))
        elif path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_allowed():
            return
        if self.path != "/api":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._same_origin():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            method, params = body["method"], body.get("params", {})
            params = {} if params is None else params
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ValueError("method must be a string and params an object")
        except (ValueError, KeyError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"malformed request: {exc}"})
            return
        try:
            result = self.server.backend.request(method, params)
        except ApiError as exc:
            self._send_json(exc.status, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(HTTPStatus.OK, {"result": result})

    # -- security ------------------------------------------------------------------------------
    def _host_allowed(self) -> bool:
        """Reject DNS-rebinding: the Host header must name this loopback server."""
        host = (self.headers.get("Host") or "").lower()
        allowed = {f"{HOST}:{self.server.port}", f"localhost:{self.server.port}"}
        if host not in allowed:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid host"})
            return False
        return True

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin.lower() in {f"http://{HOST}:{self.server.port}", f"http://localhost:{self.server.port}"}

    # -- responses -----------------------------------------------------------------------------
    def _send_static(self, relpath: str) -> None:
        root = self.server.static_dir
        target = (root / relpath).resolve()
        if root not in target.parents or not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript"
        self._send(HTTPStatus.OK, path.read_bytes(), f"{content_type}; charset=utf-8")

    def _send_json(self, status: int, document: Any) -> None:
        self._send(status, json.dumps(document, allow_nan=False).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)
