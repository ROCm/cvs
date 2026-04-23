"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import logging
import unittest

from cvs.core.launchers.mpi import MpiLauncher
from cvs.core.orchestrator import Orchestrator
from cvs.core.scope import ExecResult, ExecScope, ExecTarget


class _RecordingTransport:
    def __init__(self, hosts, head_node):
        self.hosts = list(hosts)
        self.head_node = head_node
        self.env_prefix = ""
        self.calls: list[dict] = []

    def exec(self, cmd, scope, *, subset=None, timeout=None):
        self.calls.append(
            {"cmd": cmd, "scope": scope, "subset": subset, "timeout": timeout}
        )
        targets = [self.head_node] if scope is ExecScope.HEAD else self.hosts
        return {h: ExecResult(host=h, output="", exit_code=0) for h in targets}

    def scp(self, *a, **kw):  # pragma: no cover
        pass


class _StubRuntime:
    name = "stub"
    capabilities: set = set()

    def __init__(self, ssh_port: int, hostfile: str):
        self._ssh_port = ssh_port
        self._hostfile = hostfile
        self.wrapped: list[str] = []

    def setup(self, transport): pass
    def teardown(self, transport): pass

    def wrap_cmd(self, cmd: str) -> str:
        # Identity wrap, but we record so the test can find the mpirun string.
        self.wrapped.append(cmd)
        return cmd

    def workload_ssh_port(self) -> int:
        return self._ssh_port

    def workload_hostfile_path(self) -> str:
        return self._hostfile


def _make_orch_with_mpi(ssh_port: int, hostfile: str):
    transport = _RecordingTransport(hosts=["h1", "h2"], head_node="h1")
    runtime = _StubRuntime(ssh_port=ssh_port, hostfile=hostfile)
    launcher = MpiLauncher(install_dir="/opt/openmpi")
    orch = Orchestrator(
        transport, runtime, {"mpi": launcher}, log=logging.getLogger("test_mpi")
    )
    return orch, transport, runtime, launcher


class TestMpiLauncherUsesRuntimeFacts(unittest.TestCase):
    """The whole point of MpiLauncher: container vs baremetal MPI difference
    is one method on the runtime, NOT a subclass of the orchestrator."""

    def test_hostshell_runtime_yields_port_22_in_mpirun_string(self):
        orch, transport, runtime, launcher = _make_orch_with_mpi(
            ssh_port=22, hostfile="/tmp/mpi_hosts.txt"
        )
        launcher.launch(
            orch, "RANK_CMD", ["h1", "h2"], {"NCCL_DEBUG": "INFO"}, ranks_per_host=4
        )
        # Exactly two transport calls: write hostfile then run mpirun.
        self.assertEqual(len(transport.calls), 2)
        write_cmd, mpi_cmd = transport.calls[0]["cmd"], transport.calls[1]["cmd"]
        self.assertIn("/tmp/mpi_hosts.txt", write_cmd)
        self.assertIn("h1 slots=4", write_cmd)
        self.assertIn("/opt/openmpi/mpirun", mpi_cmd)
        self.assertIn("-p 22 ", mpi_cmd)
        self.assertIn("--hostfile /tmp/mpi_hosts.txt", mpi_cmd)
        self.assertIn("-x NCCL_DEBUG=INFO", mpi_cmd)
        self.assertIn("--np 8", mpi_cmd)  # 2 hosts * 4 ranks_per_host
        self.assertIn("RANK_CMD", mpi_cmd)

    def test_docker_like_runtime_yields_port_2224_in_mpirun_string(self):
        orch, transport, runtime, launcher = _make_orch_with_mpi(
            ssh_port=2224, hostfile="/tmp/mpi_hosts.txt"
        )
        launcher.launch(
            orch, "RANK_CMD", ["h1", "h2"], {"NCCL_DEBUG": "INFO"}, ranks_per_host=4
        )
        mpi_cmd = transport.calls[1]["cmd"]
        self.assertIn("-p 2224 ", mpi_cmd)
        # Hostfile path is identical -- it's path INSIDE the runtime, which
        # for docker is the in-container /tmp.
        self.assertIn("--hostfile /tmp/mpi_hosts.txt", mpi_cmd)


class TestMpiLauncherCallsThroughRuntimeTarget(unittest.TestCase):
    """Both the hostfile write and the mpirun execution go through
    target=RUNTIME so they end up in the right namespace (host shell for
    hostshell, in-container for docker)."""

    def test_both_calls_route_through_runtime_target(self):
        orch, transport, runtime, launcher = _make_orch_with_mpi(
            ssh_port=22, hostfile="/tmp/mpi_hosts.txt"
        )
        launcher.launch(
            orch, "RANK_CMD", ["h1", "h2"], {}, ranks_per_host=4
        )
        # Both calls should have gone through the runtime wrap. Our stub
        # records every wrap_cmd call.
        self.assertEqual(len(runtime.wrapped), 2)
        # And both should be HEAD-scoped (mpirun on the head node).
        self.assertEqual(transport.calls[0]["scope"], ExecScope.HEAD)
        self.assertEqual(transport.calls[1]["scope"], ExecScope.HEAD)


class TestMpiLauncherExtraArgs(unittest.TestCase):
    def test_constructor_extra_args_and_call_extra_args_both_appear(self):
        launcher = MpiLauncher(
            install_dir="/opt/openmpi", extra_args=["--bind-to", "numa"]
        )
        orch, transport, _, _ = _make_orch_with_mpi(
            ssh_port=22, hostfile="/tmp/mpi_hosts.txt"
        )
        # Replace orch's launcher with the configured one
        orch.launchers["mpi"] = launcher
        launcher.launch(
            orch, "RANK_CMD", ["h1"], {}, ranks_per_host=2,
            extra_args=["--mca", "btl_tcp_if_include", "eth0"],
        )
        mpi_cmd = transport.calls[1]["cmd"]
        self.assertIn("--bind-to numa", mpi_cmd)
        self.assertIn("--mca btl_tcp_if_include eth0", mpi_cmd)


if __name__ == "__main__":
    unittest.main()
