"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import shlex
from typing import Dict, List, Optional, Sequence

# Teardown reclaims strictly by this label (label-scoped ``docker rm -f``), never
# a global prune that could reap unrelated containers on a shared host.
RUN_ID_LABEL = "cvs_run_id"


class ContainerHandle:
    """Context manager for a single workload container.

    - Non-privileged by default: ``--privileged`` / ``seccomp=unconfined`` are
      opt-in.
    - Tagged ``cvs_run_id=<run_id>``; teardown removes by that label only, never
      a global prune.
    - On exit ``__exit__`` snapshots logs, ``dmesg`` and GPU state before
      removal, even on the error path.
    - ``command`` / ``extra_args`` are appended to the ``docker run`` line
      unquoted, so an operator can pass arbitrary shell; never build them from
      untrusted input.
    - Spec fields (``name``, ``network``, ``ipc``, ``shm_size``, ``devices``,
      ``volumes``, ``ports``, ``env``) must be ``str`` — dict keys *and* values.
      Each token is quoted with ``shlex.quote``, which does not coerce, so a
      non-``str`` (an ``int`` port, a ``pathlib.Path``) raises ``TypeError``
      instead of silently emitting a wrong arg. The G2 ``ContainerSpec``
      stringifies before passing.

    The runtime is reached through an injected ``runner.exec(cmd) -> str | dict``
    (duck-typed to :class:`cvs.lib.parallel.pssh.Pssh` or a local wrapper), so the
    class is unit-testable with a fake.
    """

    def __init__(
        self,
        image: str,
        run_id: str,
        runner,
        *,
        name: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        shm_size: Optional[str] = None,
        ports: Optional[Dict[str, str]] = None,
        network: Optional[str] = None,
        ipc: Optional[str] = None,
        privileged: bool = False,
        seccomp_unconfined: bool = False,
        command: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> None:
        self.image = image
        self.run_id = run_id
        self.runner = runner
        self.name = name or f"cvs_{run_id}"
        self.devices = list(devices or [])
        self.volumes = dict(volumes or {})
        self.env = dict(env or {})
        self.shm_size = shm_size
        self.ports = dict(ports or {})
        self.network = network
        self.ipc = ipc
        self.privileged = privileged
        self.seccomp_unconfined = seccomp_unconfined
        self.command = command
        self.extra_args = list(extra_args or [])
        self.started = False

    def build_run_command(self) -> str:
        """Return the ``docker run -d`` command string. Pure / side-effect free."""
        parts: List[str] = ["docker", "run", "-d"]
        parts += ["--name", shlex.quote(self.name)]
        parts += ["--label", f"{RUN_ID_LABEL}={shlex.quote(self.run_id)}"]
        if self.privileged:
            parts.append("--privileged")
        if self.seccomp_unconfined:
            parts += ["--security-opt", "seccomp=unconfined"]
        if self.network:
            parts += ["--network", shlex.quote(self.network)]
        if self.ipc:
            parts += ["--ipc", shlex.quote(self.ipc)]
        if self.shm_size:
            parts += ["--shm-size", shlex.quote(self.shm_size)]
        for dev in self.devices:
            parts += ["--device", shlex.quote(dev)]
        for host_path, container_path in self.volumes.items():
            parts += ["-v", f"{shlex.quote(host_path)}:{shlex.quote(container_path)}"]
        for host_port, container_port in self.ports.items():
            parts += ["-p", f"{shlex.quote(host_port)}:{shlex.quote(container_port)}"]
        for key, value in self.env.items():
            parts += ["-e", f"{shlex.quote(key)}={shlex.quote(value)}"]
        # Appended unquoted on purpose; never pass untrusted input.
        parts += list(self.extra_args)
        parts.append(shlex.quote(self.image))
        if self.command:
            # Also raw / operator-authored: passed through verbatim, not quoted.
            parts.append(self.command)
        return " ".join(parts)

    def _run(self, cmd: str, timeout: Optional[float] = None) -> str:
        out = self.runner.exec(cmd, timeout=timeout) if timeout is not None else self.runner.exec(cmd)
        # Pssh.exec returns a per-host dict; a local runner returns a plain
        # string. Flatten either form to one string for callers.
        if isinstance(out, dict):
            return "\n".join(str(v) for v in out.values())
        return str(out)

    def __enter__(self) -> "ContainerHandle":
        try:
            self._run(self.build_run_command())
            self.started = True
        except BaseException:
            # __exit__ does not run when __enter__ raises, so apply the same
            # teardown contract here: forensics (if launched) then NAME-scoped
            # removal. The label-scoped remove() would wipe sibling containers
            # already launched under the same run_id (e.g. a successful 'router'
            # when 'server' failed to start). capture()/remove are best-effort.
            if self.started:
                self.capture()
            self._remove_by_name()
            raise
        return self

    def _remove_by_name(self) -> None:
        """Remove only this handle's container (failed-launch path)."""
        try:
            self._run(f"docker rm -f {shlex.quote(self.name)} 2>/dev/null || true")
        except Exception:
            pass

    def capture(self) -> Dict[str, str]:
        """Snapshot logs + dmesg + GPU state. Best-effort; never raises."""
        artifacts: Dict[str, str] = {}
        probes = {
            "container.log": f"docker logs {shlex.quote(self.name)} 2>&1 || true",
            "dmesg.txt": "dmesg -T 2>&1 | tail -n 500 || true",
            "gpu_state.txt": "(rocm-smi || amd-smi monitor || true) 2>&1",
        }
        for artifact_name, cmd in probes.items():
            try:
                artifacts[artifact_name] = self._run(cmd)
            except Exception as exc:  # noqa: BLE001 - capture must never mask the real error
                artifacts[artifact_name] = f"<capture failed: {exc}>"
        return artifacts

    def remove(self) -> None:
        """Remove every container started under this run_id label."""
        cmd = (
            f"docker ps -aq --filter label={RUN_ID_LABEL}={shlex.quote(self.run_id)} "
            f"| xargs --no-run-if-empty docker rm -f"
        )
        try:
            self._run(cmd)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self.started:
                self.capture()
        finally:
            self.remove()
        # Do not suppress exceptions raised inside the with-block.
        return False
