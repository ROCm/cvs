"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cvs.core.scope import ExecResult, ExecScope, ExecTarget

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


class MpiLauncher:
    """WorkloadLauncher that fans out via mpirun on the head node.

    This is the launcher that today's RCCL tests use. It asks the runtime for
    its workload-launch facts (ssh port, hostfile path) so the same launcher
    works against either HostShellRuntime (port 22, host /tmp/...) or
    DockerRuntime (port 2224, in-container /tmp/...) without isinstance checks.

    The container/baremetal MPI difference becomes one method on the runtime,
    not a subclass of the orchestrator.
    """

    name = "mpi"

    def __init__(
        self,
        install_dir: str = "/opt/openmpi",
        extra_args: Optional[list[str]] = None,
        default_timeout: int = 500,
    ):
        self.install_dir = install_dir
        self.extra_args: list[str] = list(extra_args or [])
        self.default_timeout = default_timeout

    @classmethod
    def parse_config(cls, config: Optional[dict]) -> "MpiLauncher":
        config = config or {}
        return cls(
            install_dir=config.get("install_dir", "/opt/openmpi"),
            extra_args=list(config.get("extra_args", []) or []),
            default_timeout=int(config.get("default_timeout", 500)),
        )

    def launch(
        self,
        orch: "Orchestrator",
        cmd: str,
        hosts: list[str],
        env: dict[str, str],
        ranks_per_host: int,
        extra_args: Optional[list[str]] = None,
    ) -> dict[str, ExecResult]:
        port = orch.runtime.workload_ssh_port()
        hostfile = orch.runtime.workload_hostfile_path()

        host_lines = "\n".join(f"{h} slots={ranks_per_host}" for h in hosts)
        write_hostfile = (
            f"sudo rm -f {hostfile}; "
            f'bash -c \'echo "{host_lines}" > {hostfile}\''
        )
        # Hostfile is written wherever mpirun will run, i.e. inside the runtime
        # (host shell for hostshell; in-container /tmp for docker).
        orch.exec(write_hostfile, scope=ExecScope.HEAD, target=ExecTarget.RUNTIME, timeout=30)

        no_of_global_ranks = len(hosts) * ranks_per_host

        runner_args = [
            "--np",
            str(no_of_global_ranks),
            "--allow-run-as-root",
            "--hostfile",
            hostfile,
        ]
        runner_args.extend(self.extra_args)
        if extra_args:
            runner_args.extend(extra_args)

        env_args = [f"-x {key}={value}" for key, value in env.items()]

        ssh_args = (
            f'--mca plm_rsh_agent ssh '
            f'--mca plm_rsh_args "-p {port} '
            f'-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"'
        )

        mpi_cmd = (
            f"{self.install_dir}/mpirun {ssh_args} "
            f"{' '.join(runner_args)} {' '.join(env_args)} {cmd}"
        )

        orch.log.info("Launching MPI job")
        orch.log.debug(f"MPI command: {mpi_cmd}")

        return orch.exec(
            mpi_cmd,
            scope=ExecScope.HEAD,
            target=ExecTarget.RUNTIME,
            timeout=self.default_timeout,
        )
