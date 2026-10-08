from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.viewer.conftest import RUN_ID
from tvis.viewer import targets
from tvis.viewer.transport import StdioBackend


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TVIS_CONFIG_DIR", str(tmp_path / "tvis-config"))


class TestResolve:
    def test_existing_local_directory_is_a_local_target(self, tmp_path: Path):
        target = targets.resolve(str(tmp_path))

        assert target.is_local and target.path == str(tmp_path.resolve())

    def test_host_and_path(self):
        target = targets.resolve("gpu-login:~/proj/tvis_runs", via=["bastion"])

        assert (target.host, target.path, target.via) == ("gpu-login", "~/proj/tvis_runs", ["bastion"])
        assert target.describe() == "bastion → gpu-login:~/proj/tvis_runs"

    def test_saved_target_by_name_with_overrides(self):
        targets.save_target("gpu", targets.Target(host="c", path="/runs", via=["a", "b"]))

        assert targets.resolve("gpu").via == ["a", "b"]
        assert targets.resolve("gpu", remote_tvis="~/venv/bin/tvis").remote_tvis == "~/venv/bin/tvis"

    @pytest.mark.parametrize("spec", ["no-such-name", "host:", ":path"])
    def test_unresolvable_specs_explain_the_accepted_forms(self, spec: str):
        with pytest.raises(targets.TargetError, match="HOST:PATH"):
            targets.resolve(spec)


class TestSavedTargets:
    def test_save_list_and_remove(self):
        targets.save_target("a", targets.Target(host="h1", path="/r1"))
        targets.save_target("b", targets.Target(host="h2", path="/r2", via=["j"]))
        targets.save_target("a", targets.Target(host="h3", path="/r3"))  # overwrite

        assert {n: t.host for n, t in targets.load_targets().items()} == {"a": "h3", "b": "h2"}
        assert targets.remove_target("a") is True
        assert targets.remove_target("a") is False
        assert list(targets.load_targets()) == ["b"]


class TestSshCommand:
    def test_hops_become_proxy_jump_and_agent_runs_on_the_final_host(self):
        target = targets.Target(host="c", path="/data/runs", via=["a", "b"])

        cmd = targets.ssh_command(target, reuse_connections=False)

        assert cmd[:3] == ["ssh", "-J", "a,b"]
        assert cmd[-2] == "c"
        assert cmd[-1] == "tvis agent --stdio --root /data/runs"

    def test_connection_reuse_options_are_added_when_requested(self):
        cmd = targets.ssh_command(targets.Target(host="c", path="/r"), reuse_connections=True)

        assert "ControlMaster=auto" in cmd and "ControlPersist=10m" in cmd

    def test_custom_remote_tvis_invocation(self):
        target = targets.Target(host="c", path="/r", remote_tvis="~/venv/bin/tvis")

        assert targets.ssh_command(target, reuse_connections=False)[-1].startswith("~/venv/bin/tvis agent")

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("~/proj/runs", "~/proj/runs"),
            ("~/my runs", "~/'my runs'"),
            ("/abs/path", "/abs/path"),
            ("/a b/$(rm -rf x)", "'/a b/$(rm -rf x)'"),
            ("~", "~"),
        ],
    )
    def test_remote_paths_are_quoted_but_tilde_still_expands(self, path: str, expected: str):
        assert targets.remote_path_arg(path) == expected


FAKE_SSH = """\
#!{python}
# Stand-in for ssh: drops options, then runs the remote command locally through a shell (like sshd).
import os, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_SSH_LOG")
if log:
    open(log, "a").write(" ".join(args) + "\\n")
while args and args[0].startswith("-"):
    flag = args.pop(0)
    if flag in ("-J", "-o", "-p", "-i", "-l"):
        args.pop(0)
host, command = args[0], " ".join(args[1:])
os.execvp("/bin/sh", ["/bin/sh", "-c", command])
"""


@pytest.fixture
def fake_ssh(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "ssh"
    path.parent.mkdir()
    path.write_text(FAKE_SSH.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_remote_runs_are_served_through_ssh_with_tilde_paths(
    run_root: Path, tmp_path: Path, fake_ssh: Path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    (home / "proj").mkdir()
    (home / "proj" / "runs").symlink_to(run_root)
    monkeypatch.setenv("HOME", str(home))
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    remote_tvis = f"{shlex.quote(sys.executable)} -m tvis.cli"
    target = targets.resolve("cluster:~/proj/runs", via=["jump1", "jump2"], remote_tvis=remote_tvis)

    backend = StdioBackend(
        targets.ssh_command(target, ssh=str(fake_ssh), reuse_connections=False), target.describe()
    )
    try:
        runs = backend.request("runs", {})
    finally:
        backend.close()

    assert [r["run_id"] for r in runs] == [RUN_ID]
    assert log.read_text().startswith("-J jump1,jump2 ")


def test_tvis_open_reports_a_missing_remote_tvis(tmp_path: Path, fake_ssh: Path):
    env = {**os.environ, "PATH": f"{fake_ssh.parent}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tvis.cli",
            "open",
            "cluster:/runs",
            "--no-browser",
            "--remote-tvis",
            "no-such-tvis",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert proc.returncode == 1
    assert "could not start the agent" in proc.stderr and "--remote-tvis" in proc.stderr


def test_tvis_target_cli_round_trip(tmp_path: Path):
    env = {**os.environ, "TVIS_CONFIG_DIR": str(tmp_path / "cfg")}

    def tvis(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "tvis.cli", *args], capture_output=True, text=True, env=env
        )

    assert tvis("target", "add", "gpu", "--host", "c", "--path", "~/runs", "--via", "a, b").returncode == 0
    assert tvis("target", "list").stdout.strip() == textwrap.dedent("gpu\ta → b → c:~/runs")
    assert tvis("target", "remove", "gpu").returncode == 0
    assert tvis("target", "remove", "gpu").returncode == 1
