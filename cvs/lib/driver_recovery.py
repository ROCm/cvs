"""amdgpu driver recovery helpers (CVS docker-mode P8).

Two real-world scenarios this addresses:

  1. **Conductor reservation refresh** drops the amdgpu module silently
     between sessions on `.dcgpu` hosts. P6 spike validation confirmed this
     happens. Without recovery, prepare_runtime preflight hard-fails on
     missing `/dev/kfd` and the user has to manually `sudo modprobe amdgpu`.

  2. **RVS LEVEL-4 stress tests** (`compute-fp8-trig` action in particular)
     can crash or unload `amdgpu` mid-test. After RVS, `/dev/kfd` may be
     gone and AGFHC / RCCL tests fail with confusing errors. This is
     documented in the cvs-config-gen skill.

The hooks here are called from:
  - `prepare_runtime` preflight (self-heal once before failing)
  - `verify_runtime` standalone test (between destructive tests)

Pure host-side commands. Always invoked via a `phdl_host` Pssh handle whose
wrapper is `NoOpWrapper` -- modprobe / cat /sys/module/amdgpu/initstate must
run on the *host*, not in the container (the kernel module lives on the host).
"""

from __future__ import annotations

import time
from typing import Dict

from cvs.lib import globals as cvs_globals

log = cvs_globals.log


def is_driver_live(phdl_host) -> Dict[str, bool]:
    """Return per-node {host: bool} of whether amdgpu is live + /dev/kfd present.

    A node is considered live iff `/dev/kfd` exists AND
    `/sys/module/amdgpu/initstate` reads `live`.
    """
    out = phdl_host.exec(
        "test -e /dev/kfd && cat /sys/module/amdgpu/initstate 2>&1 || echo NOT_LIVE",
        timeout=15,
    )
    return {node: ("live" in val and "NOT_LIVE" not in val) for node, val in out.items()}


def verify_or_recover_driver(phdl_host, max_wait_seconds: int = 20) -> Dict[str, dict]:
    """Verify amdgpu is live on every node; if not, modprobe and re-check.

    Returns a per-node dict with:
        before    -- bool, was the driver live before recovery?
        attempted -- bool, was modprobe attempted on this node?
        after     -- bool, is the driver live after recovery (or initially)?

    Raises RuntimeError if any node is still not live after modprobe + wait.
    Idempotent: a no-op when every node is already live.
    """
    state_before = is_driver_live(phdl_host)
    needs_recovery = [n for n, ok in state_before.items() if not ok]

    if not needs_recovery:
        log.info("[driver_recovery] all %d node(s) already live; no-op", len(state_before))
        return {
            n: {"before": True, "attempted": False, "after": True}
            for n in state_before
        }

    log.warning(
        "[driver_recovery] amdgpu NOT live on %d/%d node(s): %s; running modprobe",
        len(needs_recovery),
        len(state_before),
        needs_recovery,
    )
    # modprobe is idempotent; safe to send to every node (live nodes no-op).
    phdl_host.exec("sudo modprobe amdgpu 2>&1; true", timeout=60)

    log.info("[driver_recovery] sleeping %ds for amdgpu to enumerate devices", max_wait_seconds)
    time.sleep(max_wait_seconds)

    state_after = is_driver_live(phdl_host)
    result: Dict[str, dict] = {}
    for node in state_before:
        result[node] = {
            "before": state_before[node],
            "attempted": node in needs_recovery,
            "after": state_after.get(node, False),
        }

    still_dead = [n for n, ok in state_after.items() if not ok]
    if still_dead:
        raise RuntimeError(
            f"amdgpu still not live after `sudo modprobe amdgpu` + {max_wait_seconds}s "
            f"on: {still_dead}. Check `dmesg` for amdgpu errors."
        )

    log.info("[driver_recovery] recovery succeeded on %d node(s)", len(needs_recovery))
    return result


def restart_container(phdl_host, container_name: str, max_wait_seconds: int = 10) -> Dict[str, bool]:
    """`docker restart <container_name>` on every node + verify Up.

    Used after driver recovery so the container's stale `/dev/kfd` and
    `/dev/dri/*` file-descriptor mappings get refreshed. Without this, even
    after the host re-`modprobe`s amdgpu, the running container still holds
    references to the old (now-gone) device nodes and ROCm calls inside the
    container fail with confusing errors.

    Returns per-node {host: bool} of post-restart "running" status.
    """
    log.info("[driver_recovery] sudo docker restart %s on each node", container_name)
    phdl_host.exec(f"sudo docker restart {container_name} 2>&1; true", timeout=60)
    time.sleep(max_wait_seconds)

    out = phdl_host.exec(
        f"sudo docker ps --filter name=^{container_name}$ "
        f"--filter status=running --format '{{{{.Names}}}}'",
        timeout=15,
    )
    return {node: container_name in val for node, val in out.items()}
