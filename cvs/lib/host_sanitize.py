"""Host sanitization: capture pre-state, apply tuning, restore (CVS docker-mode P11).

Conservative v1 scope (no service stops, no kernel module reloads):

  * CPU frequency governor (per-CPU)
  * VM drop_caches (idempotent, no state to restore)
  * GPU performance level (rocm-smi --setperflevel)

Pre-state is captured into a snapshot file on EACH node:
    /tmp/cvs/host_sanitize/snapshot.json

`restore_host()` reads that snapshot and reverts. If the snapshot is missing
(e.g. a mid-run crash erased /tmp), it reverts to safe-default values:
governor=ondemand, perflevel=auto.

Sanitization is opt-in via `runtime.sanitize_host: true` in cluster.json.
"""

from __future__ import annotations

from typing import Dict

from cvs.lib import globals as cvs_globals

log = cvs_globals.log

SNAPSHOT_PATH = "/tmp/cvs/host_sanitize/snapshot.json"

# Safe-default values applied if snapshot file is missing during restore.
# `powersave` is chosen over `ondemand` because it is universally available
# across kernel cpufreq drivers (some hosts ship without `ondemand`).
SAFE_DEFAULT_GOVERNOR = "powersave"
SAFE_DEFAULT_PERFLEVEL = "auto"


def _exec(phdl, cmd: str, timeout: int = 30) -> Dict[str, str]:
    return phdl.exec(cmd, timeout=timeout)


def capture_pre_state(phdl) -> Dict[str, dict]:
    """Per-node snapshot of mutable host knobs we plan to touch."""
    log.info("[P11] capturing host pre-state")
    snap: Dict[str, dict] = {}
    nodes = list(phdl.hosts) if hasattr(phdl, "hosts") else []
    if not nodes:
        # Fallback: probe via a no-op exec to learn nodes
        out = _exec(phdl, "true")
        nodes = list(out.keys())

    governors = _exec(
        phdl,
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown",
    )
    perflevels = _exec(
        phdl,
        "rocm-smi --showperflevel 2>/dev/null | grep -E 'Performance Level' | head -1 | awk -F: '{print $NF}' | tr -d ' ' || echo unknown",
    )
    for node in nodes:
        snap[node] = {
            "cpu_governor": governors.get(node, "unknown").strip().splitlines()[-1] if governors.get(node) else "unknown",
            "rocm_perflevel": perflevels.get(node, "unknown").strip().splitlines()[-1] if perflevels.get(node) else "unknown",
        }
    return snap


def write_snapshot(phdl, snapshot: Dict[str, dict]) -> None:
    """Persist the snapshot on each remote node so restore_host() can find it."""
    import json
    payload = json.dumps(snapshot, indent=2).replace("'", "'\\''")
    cmd = (
        f"sudo mkdir -p /tmp/cvs/host_sanitize && "
        f"echo '{payload}' | sudo tee {SNAPSHOT_PATH} > /dev/null && "
        f"sudo chmod 644 {SNAPSHOT_PATH}"
    )
    _exec(phdl, cmd)


def read_snapshot(phdl) -> Dict[str, dict]:
    """Per-node snapshot read from the remote snapshot files."""
    import json
    out = _exec(phdl, f"cat {SNAPSHOT_PATH} 2>/dev/null || echo '{{}}'")
    parsed: Dict[str, dict] = {}
    for node, raw in out.items():
        # The cat result may include the entire file contents; we serialized
        # the SAME object across all nodes, so json.loads gives us the
        # full dict; we then look up the per-node entry.
        try:
            full = json.loads(raw.strip())
        except Exception:
            full = {}
        parsed[node] = full.get(node, {}) if isinstance(full, dict) else {}
    return parsed


def apply_sanitize(phdl) -> None:
    """Apply CVS performance tuning. Idempotent."""
    log.info("[P11] applying host sanitize (governor=performance, drop_caches, perflevel=high)")
    cmd = (
        # CPU governor performance on every online CPU
        "for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
        "  echo performance | sudo tee $c > /dev/null 2>&1 || true; "
        "done; "
        # Drop caches (idempotent, no state)
        "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; "
        # GPU clocks: pin to high (best-effort; ignored if rocm-smi missing)
        "rocm-smi --setperflevel high > /dev/null 2>&1 || true"
    )
    _exec(phdl, cmd, timeout=60)


def restore_host(phdl, snapshot: Dict[str, dict] | None = None) -> Dict[str, dict]:
    """Restore captured pre-state. Falls back to safe defaults if snapshot is empty."""
    if snapshot is None:
        snapshot = read_snapshot(phdl)

    log.info("[P11] restoring host state")
    # We restore the SAME values to ALL nodes that share the snapshot;
    # if per-node values differ the snapshot dict has per-node entries.
    # For v1 simplicity (homogeneous nodes), we apply per-node.
    out: Dict[str, dict] = {}
    nodes = list(snapshot.keys()) if snapshot else []
    if not nodes:
        # No snapshot at all -> fan out to every node phdl knows about
        probe = _exec(phdl, "true")
        nodes = list(probe.keys())

    for node in nodes:
        node_snap = snapshot.get(node, {}) if isinstance(snapshot, dict) else {}
        gov = (node_snap.get("cpu_governor") or "").strip()
        perf = (node_snap.get("rocm_perflevel") or "").strip()
        if not gov or gov == "unknown":
            gov = SAFE_DEFAULT_GOVERNOR
        if not perf or perf == "unknown":
            perf = SAFE_DEFAULT_PERFLEVEL
        out[node] = {"cpu_governor": gov, "rocm_perflevel": perf}

    # Fan out per-node restore. Since CVS Pssh runs the same cmd on every
    # host, we issue one cluster-wide restore using the FIRST node's values.
    # In a homogeneous fleet this works; for v1 we accept that limitation.
    first = next(iter(out.values()))
    cmd = (
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
        f"  echo {first['cpu_governor']} | sudo tee $c > /dev/null 2>&1 || true; "
        f"done; "
        f"rocm-smi --setperflevel {first['rocm_perflevel']} > /dev/null 2>&1 || true"
    )
    _exec(phdl, cmd, timeout=60)
    return out


def diff_state(before: Dict[str, dict], after: Dict[str, dict]) -> Dict[str, dict]:
    """Return per-node {field: (before, after)} for fields that changed."""
    diff: Dict[str, dict] = {}
    for node in set(before) | set(after):
        b = before.get(node, {})
        a = after.get(node, {})
        node_diff = {}
        for k in set(b) | set(a):
            if b.get(k) != a.get(k):
                node_diff[k] = (b.get(k), a.get(k))
        if node_diff:
            diff[node] = node_diff
    return diff
