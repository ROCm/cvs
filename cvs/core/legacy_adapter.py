"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cvs.core.scope import ExecScope, ExecTarget

if TYPE_CHECKING:
    from cvs.core.orchestrator import Orchestrator


class OrchestratorPsshAdapter:
    """Pssh-shaped adapter around the new Orchestrator.

    Several library helpers (cvs/lib/verify_lib.py, cvs/lib/linux_utils.py,
    ...) were written against the legacy Pssh contract:

        phdl.exec(cmd)               -> dict[host, str]
        phdl.exec_cmd_list(cmd_list) -> dict[host, str]

    They are called from the migrated RCCL fixtures, which now hand around
    the new Orchestrator. The new Orchestrator returns dict[host, ExecResult]
    and requires a (scope, target) pair on each call. This adapter translates.

    Routes everything through scope=ALL, target=HOST -- the legacy helpers
    are doing host-side work (dmesg, lspci, amd-smi, ethtool) that should
    NOT be wrapped through the runtime even when the runtime is docker.
    Using this adapter therefore also fixes the long-standing class of bug
    where, e.g., dmesg ran inside the container in container mode and
    silently scanned the wrong kernel log.

    Deletion criterion: when verify_lib + linux_utils get migrated to the
    new Orchestrator API directly (a future PR), this file goes away.
    """

    def __init__(self, orch: "Orchestrator"):
        self._orch = orch
        # Surface the few attributes legacy callers read directly off Pssh.
        self.env_prefix = orch.transport.env_prefix
        self.host_list = list(orch.hosts)
        self.reachable_hosts = list(orch.hosts)
        self.unreachable_hosts: list[str] = []

    def exec(self, cmd: str, timeout: Optional[int] = None, **_kwargs):
        results = self._orch.exec(
            cmd, scope=ExecScope.ALL, target=ExecTarget.HOST, timeout=timeout
        )
        return {host: r.output for host, r in results.items()}

    def exec_cmd_list(self, cmd_list, timeout: Optional[int] = None, **_kwargs):
        # exec_cmd_list runs DIFFERENT commands on DIFFERENT hosts (positional
        # pairing). The new Transport API doesn't expose this; route through
        # the underlying Pssh inside PsshTransport. Coupling to that internal
        # is acceptable inside this transitional adapter.
        pssh = self._orch.transport._all
        return pssh.exec_cmd_list(cmd_list, timeout=timeout)


def as_pssh_handle(handle):
    """Return a Pssh-shaped handle.

    If `handle` is the new Orchestrator, wrap it in OrchestratorPsshAdapter.
    Otherwise (legacy Pssh, mocks, anything else), return as-is.
    """
    # Late import to avoid pulling in the Orchestrator type for callers that
    # don't care.
    from cvs.core.orchestrator import Orchestrator

    if isinstance(handle, Orchestrator):
        return OrchestratorPsshAdapter(handle)
    return handle


def as_string_dict(d):
    """Accept either dict[host, str] or dict[host, ExecResult] and return
    dict[host, str].

    Use this for legacy helpers that receive an exec-output dict directly --
    e.g. verify_dmesg_for_errors(phdl, start_time, end_time) where the
    start/end_time dicts came from orch.exec(...) (= dict[host, ExecResult])
    on the migrated RCCL fixtures, but were dict[host, str] historically.
    """
    if not d:
        return d
    sample = next(iter(d.values()))
    # Duck-type ExecResult: it has .output and .exit_code.
    if hasattr(sample, "output") and hasattr(sample, "exit_code"):
        return {k: v.output for k, v in d.items()}
    return d
