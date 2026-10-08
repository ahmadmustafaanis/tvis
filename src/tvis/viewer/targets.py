"""Where runs live and how to reach them: local paths, `host:path`, and saved targets.

The laptop always starts the connection (remote machines can't reliably reach back through NAT
and jump hosts). Hops come from the user's ~/.ssh/config (ProxyJump) or explicit ``--via``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_REMOTE_TVIS = "tvis"


@dataclass
class Target:
    host: str | None  # None = local
    path: str
    via: list[str] = field(default_factory=list)
    remote_tvis: str = DEFAULT_REMOTE_TVIS
    name: str | None = None

    @property
    def is_local(self) -> bool:
        return self.host is None

    def describe(self) -> str:
        if self.is_local:
            return self.path
        hops = " → ".join([*self.via, self.host or ""])
        return f"{hops}:{self.path}"


class TargetError(ValueError):
    pass


def config_dir() -> Path:
    return Path(os.environ.get("TVIS_CONFIG_DIR", Path.home() / ".tvis"))


def targets_file() -> Path:
    return config_dir() / "targets.json"


def load_targets() -> dict[str, Target]:
    path = targets_file()
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {name: Target(**{**spec, "name": name}) for name, spec in raw.items()}


def save_target(name: str, target: Target) -> None:
    targets = {n: t for n, t in load_targets().items() if n != name}
    targets[name] = target
    _write_targets(targets)


def remove_target(name: str) -> bool:
    targets = load_targets()
    if name not in targets:
        return False
    del targets[name]
    _write_targets(targets)
    return True


def _write_targets(targets: dict[str, Target]) -> None:
    path = targets_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {n: {k: v for k, v in asdict(t).items() if k != "name"} for n, t in sorted(targets.items())}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def resolve(spec: str, *, via: list[str] | None = None, remote_tvis: str | None = None) -> Target:
    """Resolve what the user typed after `tvis open`.

    Order: an existing local path → a saved target name → ``host:path``.
    """
    via = list(via or [])
    if Path(spec).expanduser().exists() and not via:
        return Target(host=None, path=str(Path(spec).expanduser().resolve()))
    saved = load_targets().get(spec)
    if saved is not None:
        if via:
            saved.via = via
        if remote_tvis:
            saved.remote_tvis = remote_tvis
        return saved
    host, sep, path = spec.partition(":")
    if not sep or not host or not path:
        raise TargetError(
            f"{spec!r} is not a local directory, a saved target, or HOST:PATH "
            "(e.g. mycluster:~/proj/tvis_runs)"
        )
    return Target(host=host, path=path, via=via, remote_tvis=remote_tvis or DEFAULT_REMOTE_TVIS)


def remote_path_arg(path: str) -> str:
    """Quote a remote path for the remote shell while keeping a leading ``~`` expandable."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def agent_command(target: Target) -> str:
    return f"{target.remote_tvis} agent --stdio --root {remote_path_arg(target.path)}"


def ssh_command(target: Target, *, ssh: str = "ssh", reuse_connections: bool | None = None) -> list[str]:
    """The full ssh invocation that starts the remote agent.

    Connection reuse (ControlMaster) is added only when the user's ssh config doesn't already set a
    ControlPath, so an existing authenticated master (e.g. after 2FA) is reused rather than shadowed.
    """
    assert target.host is not None
    base = [ssh]
    if target.via:
        base += ["-J", ",".join(target.via)]
    if reuse_connections is None:
        reuse_connections = not _user_configured_control_path(base, target.host)
    options = ["-T", "-o", "ExitOnForwardFailure=no"]
    if reuse_connections:
        options += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={Path('~/.ssh/tvis-%C').expanduser()}",
            "-o",
            "ControlPersist=10m",
        ]
    return [*base, *options, target.host, agent_command(target)]


def _user_configured_control_path(base: list[str], host: str) -> bool:
    if shutil.which(base[0]) is None:
        return False
    try:
        out = subprocess.run([*base, "-G", host], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key.lower() == "controlpath":
            return value.strip().lower() != "none"
    return False
