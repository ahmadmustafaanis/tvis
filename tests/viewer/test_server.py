from __future__ import annotations

import http.client
import json
from pathlib import Path

import pytest

from tests.viewer.conftest import RUN_ID
from tvis.viewer.server import ViewerServer
from tvis.viewer.transport import LocalBackend


@pytest.fixture
def server(run_root: Path, tmp_path: Path):
    static = tmp_path / "ui"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>tvis</title>")
    (static / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("nope")
    srv = ViewerServer(LocalBackend(run_root), static_dir=static)
    srv.serve_in_background()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(
    server: ViewerServer, method: str, path: str, body: dict | None = None, headers: dict | None = None
):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, response.getheader("Content-Type"), data


def test_binds_to_loopback_only(server: ViewerServer):
    assert server.server_address[0] == "127.0.0.1"
    assert server.url == f"http://127.0.0.1:{server.port}/"


def test_serves_the_ui_and_static_assets(server: ViewerServer):
    status, ctype, body = request(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b"tvis" in body

    status, ctype, _ = request(server, "GET", "/static/app.js")
    assert status == 200 and ctype.startswith("text/javascript")


def test_api_round_trip(server: ViewerServer):
    status, _, body = request(server, "POST", "/api", {"method": "run", "params": {"run": RUN_ID}})

    assert status == 200
    assert json.loads(body)["result"]["meta"]["run_id"] == RUN_ID


def test_api_errors_carry_status_and_message(server: ViewerServer):
    status, _, body = request(server, "POST", "/api", {"method": "run", "params": {"run": "missing"}})

    assert status == 404 and "not found" in json.loads(body)["error"]


@pytest.mark.parametrize("body", [{"params": {}}, {"method": 3}, {"method": "runs", "params": []}])
def test_malformed_api_requests_are_rejected(server: ViewerServer, body: dict):
    status, _, _ = request(server, "POST", "/api", body)

    assert status == 400


def test_static_paths_cannot_escape_the_ui_directory(server: ViewerServer):
    status, _, _ = request(server, "GET", "/static/../secret.txt")

    assert status == 404


def test_foreign_host_header_is_refused(server: ViewerServer):
    # DNS rebinding: a page on evil.example resolving to 127.0.0.1 sends Host: evil.example
    status, _, _ = request(server, "POST", "/api", {"method": "runs"}, headers={"Host": "evil.example"})

    assert status == 403


def test_cross_origin_api_calls_are_refused(server: ViewerServer):
    status, _, _ = request(
        server, "POST", "/api", {"method": "runs"}, headers={"Origin": "http://evil.example"}
    )

    assert status == 403


def test_same_origin_api_calls_are_allowed(server: ViewerServer):
    origin = f"http://127.0.0.1:{server.port}"
    status, _, _ = request(server, "POST", "/api", {"method": "runs"}, headers={"Origin": origin})

    assert status == 200
