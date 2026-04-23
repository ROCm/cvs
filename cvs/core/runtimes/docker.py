"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import getpass
import shlex
from typing import Optional

from cvs.core.scope import ExecScope
from cvs.core.transports.base import Transport


# Docker defaults aimed at ROCm + InfiniBand workloads. These are the DEFAULT
# settings; cluster.json runtime.config can override any of them. A dedicated
# RuntimeProfile abstraction (so an apptainer/podman runtime can share the
# same intent) is intentionally deferred to a follow-up PR -- see plan.
_DEFAULTS: dict = {
    "container_name": "cvs-runner",
    "env": {"GPUS": "8", "MULTINODE": "true"},
    "volumes": [],  # user-specific volumes are appended dynamically by parse_config
    "devices": ["/dev/kfd", "/dev/dri", "/dev/infiniband"],
    "cap_add": ["SYS_PTRACE", "IPC_LOCK", "SYS_ADMIN"],
    "security_opt": ["seccomp=unconfined", "apparmor=unconfined"],
    "group_add": ["video"],
    "ulimit": ["memlock=-1"],
    "network": "host",
    "ipc": "host",
    "privileged": True,
    # If True, the docker run command also expands /dev/infiniband/* via a
    # per-host shell glob so RDMA devices come through.
    "expand_infiniband_devices": True,
}


class DockerRuntime:
    """Runtime that runs commands inside a long-lived per-host docker container.

    Lifecycle (per-host):
      * setup    - docker run -d --name <container_name> ... sleep infinity
                   so the container stays up between exec calls.
      * teardown - docker rm -f <container_name> on every host.

    wrap_cmd produces sudo docker exec -e ENVS NAME bash -lc <quoted cmd>;
    workload_ssh_port is 2224 (in-container sshd that MultinodeSshPhase brings
    up); workload_hostfile_path is /tmp/mpi_hosts.txt INSIDE the container.

    capabilities = {"in_namespace_sshd"} so MultinodeSshPhase.applies_to
    returns True; a hostshell runtime would have an empty set and the phase
    skips itself.
    """

    name = "docker"

    def __init__(
        self,
        image: str,
        container_name: str = "cvs-runner",
        env: Optional[dict[str, str]] = None,
        volumes: Optional[list[str]] = None,
        devices: Optional[list[str]] = None,
        cap_add: Optional[list[str]] = None,
        security_opt: Optional[list[str]] = None,
        group_add: Optional[list[str]] = None,
        ulimit: Optional[list[str]] = None,
        network: str = "host",
        ipc: str = "host",
        privileged: bool = True,
        expand_infiniband_devices: bool = True,
        image_tar: Optional[str] = None,
        capabilities: Optional[set[str]] = None,
    ):
        self.image = image
        self.container_name = container_name
        self.env: dict[str, str] = dict(env or {})
        self.volumes: list[str] = list(volumes or [])
        self.devices: list[str] = list(devices or [])
        self.cap_add: list[str] = list(cap_add or [])
        self.security_opt: list[str] = list(security_opt or [])
        self.group_add: list[str] = list(group_add or [])
        self.ulimit: list[str] = list(ulimit or [])
        self.network = network
        self.ipc = ipc
        self.privileged = bool(privileged)
        self.expand_infiniband_devices = bool(expand_infiniband_devices)
        self.image_tar = image_tar
        self.capabilities: set[str] = (
            set(capabilities) if capabilities is not None else {"in_namespace_sshd"}
        )

    @classmethod
    def parse_config(cls, config: Optional[dict]) -> "DockerRuntime":
        """Build a DockerRuntime from the runtime.config dict in cluster.json."""
        from cvs.core.errors import OrchestratorConfigError

        config = config or {}
        image = config.get("image")
        if not image:
            raise OrchestratorConfigError(
                "runtime.config.image is required when runtime.name == 'docker'"
            )

        # Defaults + user overrides. Lists are replaced wholesale (not concatenated)
        # so users can opt out of e.g. SYS_ADMIN by setting cap_add=[] explicitly.
        merged = {**_DEFAULTS, **config}
        merged_env = {**_DEFAULTS["env"], **(config.get("env") or {})}

        # User-specific defaults: home and SSH key bind-mounts. Keep these in
        # the runtime (not in MultinodeSshPhase) because the bind mount has to
        # exist BEFORE setup_sshd runs and requires no config from the user.
        user = getpass.getuser()
        default_volumes = [f"/home/{user}:/workspace", f"/home/{user}/.ssh:/host_ssh"]
        volumes = default_volumes + list(config.get("volumes") or [])

        return cls(
            image=image,
            container_name=merged.get("container_name", "cvs-runner"),
            env=merged_env,
            volumes=volumes,
            devices=merged.get("devices") or [],
            cap_add=merged.get("cap_add") or [],
            security_opt=merged.get("security_opt") or [],
            group_add=merged.get("group_add") or [],
            ulimit=merged.get("ulimit") or [],
            network=merged.get("network", "host"),
            ipc=merged.get("ipc", "host"),
            privileged=bool(merged.get("privileged", True)),
            expand_infiniband_devices=bool(merged.get("expand_infiniband_devices", True)),
            image_tar=merged.get("image_tar"),
        )

    def _docker_run_args(self) -> str:
        """Build the argument string for `sudo docker run -d --name NAME ARGS IMAGE sleep infinity`."""
        parts: list[str] = []
        parts.append(f"--network {self.network}")
        parts.append(f"--ipc {self.ipc}")
        if self.privileged:
            parts.append("--privileged")
        for v in self.volumes:
            parts.append(f"-v {shlex.quote(v)}")
        for d in self.devices:
            parts.append(f"--device {shlex.quote(d)}")
        for c in self.cap_add:
            parts.append(f"--cap-add {shlex.quote(c)}")
        for s in self.security_opt:
            parts.append(f"--security-opt {shlex.quote(s)}")
        for g in self.group_add:
            parts.append(f"--group-add {shlex.quote(g)}")
        for u in self.ulimit:
            parts.append(f"--ulimit {shlex.quote(u)}")
        for k, v in self.env.items():
            parts.append(f"-e {shlex.quote(f'{k}={v}')}")
        if self.expand_infiniband_devices:
            # Per-host glob expansion happens at remote-shell evaluation time.
            parts.append('$(for dev in /dev/infiniband/*; do echo -n "--device $dev:$dev "; done)')
        return " ".join(parts)

    def setup(self, transport: Transport) -> None:
        from cvs.core.errors import OrchestratorConfigError

        if self.image_tar:
            # Load the image on every host before trying to run it.
            load = f"sudo docker load < {shlex.quote(self.image_tar)}"
            results = transport.exec(load, ExecScope.ALL, timeout=600)
            failed = [h for h, r in results.items() if r.exit_code != 0]
            if failed:
                raise OrchestratorConfigError(
                    f"docker load failed on hosts: {failed}"
                )

        # Idempotent: remove any leftover container before starting.
        transport.exec(
            f"sudo docker rm -f {shlex.quote(self.container_name)} 2>/dev/null || true",
            ExecScope.ALL,
            timeout=30,
        )

        run = (
            f"sudo docker run -d --name {shlex.quote(self.container_name)} "
            f"{self._docker_run_args()} {shlex.quote(self.image)} sleep infinity"
        )
        results = transport.exec(run, ExecScope.ALL, timeout=120)
        failed = {h: r for h, r in results.items() if r.exit_code != 0}
        if failed:
            # Best-effort cleanup of any partial successes.
            self.teardown(transport)
            raise OrchestratorConfigError(
                "docker run failed on hosts: "
                + ", ".join(f"{h} (exit={r.exit_code}): {r.output.strip()}" for h, r in failed.items())
            )

    def teardown(self, transport: Transport) -> None:
        transport.exec(
            f"sudo docker rm -f {shlex.quote(self.container_name)} 2>/dev/null || true",
            ExecScope.ALL,
            timeout=30,
        )

    def wrap_cmd(self, cmd: str) -> str:
        # -e ENV=VALUE flags so workload commands see the env even after the
        # bash -lc subshell is set up.
        env_flags = " ".join(
            f"-e {shlex.quote(f'{k}={v}')}" for k, v in self.env.items()
        )
        return (
            f"sudo docker exec {env_flags} {shlex.quote(self.container_name)} "
            f"bash -lc {shlex.quote(cmd)}"
        )

    def workload_ssh_port(self) -> int:
        return 2224

    def workload_hostfile_path(self) -> str:
        return "/tmp/mpi_hosts.txt"
