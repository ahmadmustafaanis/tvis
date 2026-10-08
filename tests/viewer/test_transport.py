from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest

from tests.viewer.conftest import RUN_ID
from tvis.viewer.api import ApiError
from tvis.viewer.transport import StdioBackend, serve_stdio

AGENT = [sys.executable, "-m", "tvis.cli", "agent", "--stdio", "--root"]


def ask(root: Path, *requests: dict) -> list[dict]:
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve_stdio(root, stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


class TestAgentProtocol:
    def test_answers_each_request_with_its_id(self, run_root: Path):
        replies = ask(run_root, {"id": 1, "method": "runs"}, {"id": 2, "method": "ping"})

        assert [r["id"] for r in replies] == [1, 2]
        assert replies[0]["result"][0]["run_id"] == RUN_ID
        assert replies[1]["result"]["ok"] is True

    def test_errors_are_replies_not_crashes(self, run_root: Path):
        replies = ask(
            run_root,
            {"id": 1, "method": "run", "params": {"run": "missing"}},
            {"id": 2, "method": "runs"},
        )

        assert replies[0]["status"] == 404 and "not found" in replies[0]["error"]
        assert "result" in replies[1]  # the agent kept serving

    def test_malformed_lines_get_an_error_reply(self, run_root: Path):
        stdout = io.StringIO()
        serve_stdio(run_root, stdin=io.StringIO("{not json\n\n"), stdout=stdout)

        reply = json.loads(stdout.getvalue())
        assert reply["id"] is None and reply["status"] == 500


class TestStdioBackend:
    def test_requests_round_trip_through_a_subprocess_agent(self, run_root: Path):
        backend = StdioBackend([*AGENT, str(run_root)], description="test agent")
        try:
            assert backend.request("run", {"run": RUN_ID})["meta"]["run_id"] == RUN_ID
        finally:
            backend.close()

    def test_concurrent_requests_get_their_own_responses(self, run_root: Path):
        backend = StdioBackend([*AGENT, str(run_root)], description="test agent")
        results: dict[int, float] = {}

        def fetch(sample: int) -> None:
            view = backend.request("tensor", {"run": RUN_ID, "batch": 0, "tid": "t2", "sample": sample})
            results[sample] = view["min"]

        try:
            threads = [threading.Thread(target=fetch, args=(s,)) for s in (0, 1) * 10]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            backend.close()

        assert results == {0: 0.0, 1: 12.0}

    def test_remote_errors_raise_api_errors_with_status(self, run_root: Path):
        backend = StdioBackend([*AGENT, str(run_root)], description="test agent")
        try:
            with pytest.raises(ApiError) as info:
                backend.request("run", {"run": "missing"})
            assert info.value.status == 404
        finally:
            backend.close()

    def test_agent_that_dies_fails_pending_and_future_requests(self, tmp_path: Path):
        backend = StdioBackend(
            [sys.executable, "-c", "import sys; sys.stdin.readline()"], description="dying agent"
        )
        try:
            with pytest.raises(ApiError) as info:
                backend.request("ping", {})
            assert info.value.status == 502
            with pytest.raises(ApiError):
                backend.request("ping", {})
        finally:
            backend.close()
