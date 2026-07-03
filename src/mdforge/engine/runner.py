"""Command execution for file-based engines (Tinker), local or over SSH.

Separating *where* a command runs from *what* the engine does keeps
:class:`~mdforge.engine.tinker.TinkerEngine` identical for local CPU and remote
GPU runs. A runner executes an argv in a working directory that contains the
input files; ``LocalRunner`` runs in place, ``SSHRunner`` stages the directory
to a remote host, runs, and syncs results back.

No ``shell=True``, no ``os.chdir`` — commands are arg-lists run with ``cwd=``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int


@runtime_checkable
class Runner(Protocol):
    def run_in(self, workdir: str | Path, argv: list[str], *, stdin: str | None = None,
               timeout: float | None = None) -> RunResult: ...


@dataclass
class LocalRunner:
    """Run commands locally via subprocess (arg-list, ``cwd=``, no shell)."""

    def run_in(self, workdir, argv, *, stdin=None, timeout=None) -> RunResult:
        proc = subprocess.run(
            argv,
            cwd=str(workdir),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return RunResult(proc.stdout or "", proc.stderr or "", proc.returncode)


@dataclass
class SSHRunner:
    """Run commands on a remote host, staging the workdir with rsync.

    The local ``workdir`` is mirrored to ``{remote_dir}/{workdir.name}`` on
    ``host`` before the command and synced back after, so engine code that reads
    output files keeps working unchanged.

    .. note::
        Built for remote GPU (tinker9 / OpenMM on a cluster) but currently
        unvalidated — the target host was unreachable at authoring time. The
        local path (:class:`LocalRunner`) is the tested default.
    """

    host: str
    remote_dir: str
    ssh: str = "ssh"
    rsync: str = "rsync"
    conda_env: str | None = None  # `conda activate <env>` before running

    def _remote_path(self, workdir: Path) -> str:
        return f"{self.remote_dir.rstrip('/')}/{workdir.name}"

    def run_in(self, workdir, argv, *, stdin=None, timeout=None) -> RunResult:
        workdir = Path(workdir)
        remote = self._remote_path(workdir)

        mkdir = subprocess.run(
            [self.ssh, self.host, f"mkdir -p {remote}"],
            capture_output=True, text=True, check=False,
        )
        if mkdir.returncode != 0:
            return RunResult("", mkdir.stderr, mkdir.returncode)

        push = subprocess.run(
            [self.rsync, "-az", f"{workdir}/", f"{self.host}:{remote}/"],
            capture_output=True, text=True, check=False,
        )
        if push.returncode != 0:
            return RunResult("", push.stderr, push.returncode)

        prefix = f"source ~/.bashrc 2>/dev/null; conda activate {self.conda_env}; " if self.conda_env else ""
        quoted = " ".join(_shquote(a) for a in argv)
        remote_cmd = f"{prefix}cd {remote} && {quoted}"
        proc = subprocess.run(
            [self.ssh, self.host, remote_cmd],
            input=stdin, capture_output=True, text=True, timeout=timeout, check=False,
        )

        # Sync results back (best effort).
        subprocess.run(
            [self.rsync, "-az", f"{self.host}:{remote}/", f"{workdir}/"],
            capture_output=True, text=True, check=False,
        )
        return RunResult(proc.stdout or "", proc.stderr or "", proc.returncode)


def _shquote(arg: str) -> str:
    import shlex
    return shlex.quote(arg)


__all__ = ["Runner", "RunResult", "LocalRunner", "SSHRunner"]
