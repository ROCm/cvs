'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""Helpers for the multi-orch RVS suite (cvs/tests/health/rvs_cvs.py).

Backend-blind by construction: this module never reads orch.orchestrator_type
and never branches on backend identity. All routing decisions are intent-based
and resolved through polymorphic methods on the orchestrator (privileged_prefix,
host_all, host_head, exec).

Public surface:
  - host_only(orch)      -> Pssh handle for host-namespace commands
  - sudo(orch, cmd)      -> command string with the orch's privileged prefix
  - sealed_tmp(name)     -> per-MULTIORCH_RUN_ID scratch path
  - stamp_run_id(orch, test_id)
                          -> echoes a sentinel into the orch log so the
                             validation harness can correlate cells
  - exec_detailed(orch, cmd, ...)
                          -> Dict[host, {output, exit_code}] variant; routes
                             through host_all so it works on both backends
                             without requiring the core ABC to grow a
                             `detailed` kwarg on exec()
  - require_run_id()     -> raise pytest.UsageError if MULTIORCH_RUN_ID unset
"""

import os

import pytest


# Plan §11 D2 default: hard-fail when MULTIORCH_RUN_ID is unset, with a clear
# pointer to run_cell.sh. Parity-safe; revisit if engineers complain.
_RUN_ID_HELP = (
    "MULTIORCH_RUN_ID is not set. The migrated rvs_cvs.py must be invoked via "
    "/tmp/cvs_iter/scripts/run_cell.sh (or run_matrix.sh) which sets a unique "
    "per-cell MULTIORCH_RUN_ID. Direct pytest invocation is intentionally "
    "rejected to keep per-iteration container/scratch state isolated. "
    "If you really need to run pytest by hand for a single-shot debug, set "
    "MULTIORCH_RUN_ID=manual_<descriptor> explicitly."
)


def require_run_id():
    """Raise pytest.UsageError if MULTIORCH_RUN_ID is unset."""
    if not os.environ.get("MULTIORCH_RUN_ID"):
        raise pytest.UsageError(_RUN_ID_HELP)


def get_run_id():
    """Return MULTIORCH_RUN_ID; raises if unset."""
    require_run_id()
    return os.environ["MULTIORCH_RUN_ID"]


def get_cell_id():
    """Return MULTIORCH_CELL or 'unknown' (CELL_ID is informational, not gating)."""
    return os.environ.get("MULTIORCH_CELL", "unknown")


def host_only(orch):
    """Return the Pssh handle that targets the physical host namespace,
    NOT the container namespace. Use for kernel/network/firewall commands
    (lsmod, dmesg, service ufw, rdma link, ibv_devinfo, cat /opt/rocm/.info/version)
    that must hit the host even when the orchestrator is container-backed."""
    return orch.host_all


def sudo(orch, cmd):
    """Prepend the orch's privileged prefix to a command string.

    On baremetal this returns 'sudo <cmd>'; on container it returns '<cmd>'
    (root inside container). The suite source never branches on backend type.
    """
    return f"{orch.privileged_prefix()}{cmd}"


def sealed_tmp(filename):
    """Return /tmp/multiorch_validate/${MULTIORCH_RUN_ID}/<filename>.

    The MULTIORCH_RUN_ID env var MUST be set (see require_run_id). All
    iteration scratch lives under this prefix so per-cell scrub
    (clean_remote.sh) and leakage_check.sh can scope to a single cell
    without disturbing concurrent peer cells in the same wave.
    """
    require_run_id()
    run_id = os.environ["MULTIORCH_RUN_ID"]
    base = f"/tmp/multiorch_validate/{run_id}"
    return f"{base}/{filename}"


def sealed_tmp_dir():
    """Return the per-cell sealed scratch directory itself."""
    return sealed_tmp("").rstrip("/")


def ensure_sealed_tmp_dir(orch, timeout=15):
    """Create the sealed scratch dir on every node so subsequent writes don't
    fail with 'no such file or directory'. Idempotent; safe to call repeatedly.
    """
    d = sealed_tmp_dir()
    orch.exec(f"mkdir -p {d}", timeout=timeout)


def stamp_run_id(orch, test_id):
    """Echo a sentinel into the orch's stdout stream so the validation harness
    can correlate per-cell sentinels back to their cell + test. The sentinel
    format is stable; do not change without updating the harness's parser.
    """
    rid = get_run_id()
    cell = get_cell_id()
    orch.exec(
        f"echo '__MULTIORCH_SENTINEL__ run_id={rid} cell={cell} test_id={test_id}'",
        timeout=10,
    )


def exec_detailed(orch, cmd, timeout=None):
    """Return Dict[host, {'output': str, 'exit_code': int}] for `cmd`.

    Routes through orch.host_all (i.e., the host namespace) so this helper
    works on both baremetal and container backends without requiring the core
    ABC to grow a `detailed` kwarg on exec(). For commands that MUST run in
    the container's namespace (workload commands like rvs/amd-smi), use the
    plain orch.exec(...) which preserves the legacy Dict[host, str] contract.
    """
    return orch.host_all.exec(cmd, timeout=timeout, detailed=True)
