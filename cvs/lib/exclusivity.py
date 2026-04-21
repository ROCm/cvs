"""Cluster exclusivity preflight checks (CVS docker-mode P10).

Converts the verbal "no other jobs on these nodes" guarantee into a
verifiable preflight invariant. Detects:

  * Stray containers (anything other than the cvs-runner + an explicit allowlist).
  * Stray GPU-using processes (PIDs holding /dev/kfd open, excluding our container).
  * Reserved port listeners (sshd-in-image port 2222 etc.).

Behavior is controlled by `runtime.exclusivity` from cluster.json:

    "warn"   (default) -- log violations to artifact, continue.
    "strict" -- fail prepare_runtime hard with a clear violation summary.

`runtime.allowed_containers` is a list of container-name patterns that are
expected on the cluster (e.g. Prometheus node-exporter on Conductor hosts).
"""

from __future__ import annotations

import re
from typing import Dict, List

from cvs.lib import globals as cvs_globals

log = cvs_globals.log


# Built-in allowlist for known infra containers commonly seen on
# AMD Conductor / shared-cluster hosts. Users can extend via cluster.json
# `runtime.allowed_containers`.
DEFAULT_ALLOWED_CONTAINERS = (
    "node-exporter",          # Prometheus node-exporter
    "node-exporter.service",  # systemd-style name
    "cadvisor",               # Google cAdvisor
)


def _is_allowed_container(name: str, allowlist) -> bool:
    if name in allowlist:
        return True
    # Substring match: e.g. "node-exporter" in "node-exporter.service"
    return any(p in name for p in allowlist)


def check_stray_containers(phdl_host, runtime_cfg) -> Dict[str, List[str]]:
    """Per-node list of running containers that aren't allowed."""
    allowlist = list(DEFAULT_ALLOWED_CONTAINERS)
    extra = getattr(runtime_cfg, "allowed_containers", None) or []
    allowlist.extend(extra)
    # Also allow our own cvs-runner container (it's allowed to be there).
    allowlist.append(runtime_cfg.container_name)

    out = phdl_host.exec(
        "sudo docker ps --format '{{.Names}}' 2>&1 || docker ps --format '{{.Names}}'",
        timeout=30,
    )
    violations: Dict[str, List[str]] = {}
    for node, raw in out.items():
        names = [
            ln.strip() for ln in raw.splitlines()
            if ln.strip() and not ln.strip().startswith(("==", "#", "Cannot"))
        ]
        node_violations = [
            n for n in names if not _is_allowed_container(n, allowlist)
        ]
        violations[node] = node_violations
    return violations


def check_kfd_holders(phdl_host) -> Dict[str, List[str]]:
    """Per-node list of PIDs holding /dev/kfd open (excluding cvs-runner)."""
    # `fuser /dev/kfd` lists PIDs; we just want PIDs that exist and aren't
    # ours. We can't easily filter our own container from the host side, so
    # for v1 we just report ALL holders -- the cvs-runner container PIDs
    # appear in the host's namespace too. The check is most useful BEFORE
    # we start cvs-runner, so this becomes a "no other workload" assertion.
    out = phdl_host.exec(
        "sudo fuser /dev/kfd 2>/dev/null | tr -s ' ' '\\n' | grep -E '^[0-9]+$' || true",
        timeout=15,
    )
    holders: Dict[str, List[str]] = {}
    for node, raw in out.items():
        pids = [ln.strip() for ln in raw.splitlines() if ln.strip().isdigit()]
        holders[node] = pids
    return holders


def check_reserved_ports(phdl_host, ports=(2222,)) -> Dict[str, List[int]]:
    """Per-node list of CVS-reserved ports already listening on."""
    port_pattern = "|".join(f":{p}\\b" for p in ports)
    out = phdl_host.exec(
        f"ss -tnlp 2>/dev/null | grep -E '({port_pattern})' || true",
        timeout=15,
    )
    busy: Dict[str, List[int]] = {}
    for node, raw in out.items():
        node_busy = []
        for p in ports:
            if re.search(rf":{p}\b", raw):
                node_busy.append(p)
        busy[node] = node_busy
    return busy


def check_exclusivity(phdl_host, runtime_cfg) -> dict:
    """Run all exclusivity checks; return a structured per-node summary.

    Caller decides what to do with violations based on `runtime.exclusivity`
    (warn vs strict). Always cheap and read-only.
    """
    log.info("[P10] exclusivity preflight (mode=%s)", runtime_cfg.exclusivity)
    summary = {
        "stray_containers": check_stray_containers(phdl_host, runtime_cfg),
        "kfd_holders": check_kfd_holders(phdl_host),
        "reserved_ports": check_reserved_ports(phdl_host),
    }
    return summary


def violation_count(summary: dict) -> int:
    """Total number of violations across all checks + nodes."""
    n = 0
    for node_dict in summary.values():
        for node_list in node_dict.values():
            n += len(node_list)
    return n


def render_violations(summary: dict) -> str:
    """Human-readable violations list for log/artifact messages."""
    lines = []
    for node, names in summary.get("stray_containers", {}).items():
        for name in names:
            lines.append(f"{node}: stray container '{name}'")
    for node, pids in summary.get("kfd_holders", {}).items():
        for pid in pids:
            lines.append(f"{node}: PID {pid} holds /dev/kfd")
    for node, ports in summary.get("reserved_ports", {}).items():
        for port in ports:
            lines.append(f"{node}: port {port} already listening")
    return "; ".join(lines) if lines else "(no violations)"
