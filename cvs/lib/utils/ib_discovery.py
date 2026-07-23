'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InfiniBand HCA discovery via ibv_devinfo or /sys/class/infiniband fallback.

Shared across suites (vllm, rccl, inferencemax). Infrastructure step, not
a benchmark step: topology is stable within a run and should be probed once.
'''

from __future__ import annotations

import re
import shlex

from cvs.lib import globals

log = globals.log

_HCA_RE = re.compile(r"hca_id:\s*(\S+)")
_IB_HCA_NETDEV_RE = re.compile(r"^mlx5_\d+$", re.I)

_SYSFS_CMD = "ls /sys/class/infiniband/ 2>/dev/null | tr '\\n' ' '"
_IBVDEVINFO_CMD = "ibv_devinfo -l 2>/dev/null"


def _parse_sysfs(output: str) -> list[str]:
    return [tok for tok in (output or "").split() if tok]


def discover_ib_hca_names(orch) -> dict[str, list[str]]:
    """Return {host: [hca_name, ...]} for all hosts in orch.

    Tries ``ibv_devinfo -l`` first; falls back to listing
    ``/sys/class/infiniband/`` when ibv_devinfo is absent from the image.
    Returns HCA names (e.g. ``rocep28s0``, ``mlx5_0``), correct for
    ``NCCL_IB_HCA``. These are NOT Linux netdev names (``ens51f1np1``) --
    those belong in ``ib_netdev`` in the suite config.

    Raises ``RuntimeError`` if:
    - any host returns an empty HCA list (indicates missing driver or no IB
      hardware on that node), or
    - the HCA lists are asymmetric across nodes (a hardware/driver mismatch
      must surface loudly, not be silently papered over by intersection).
    """
    # Try ibv_devinfo first.
    raw = orch.exec(_IBVDEVINFO_CMD)
    result: dict[str, list[str]] = {}
    use_sysfs = False
    for host, output in (raw or {}).items():
        hcas = _HCA_RE.findall(output or "")
        if not hcas:
            use_sysfs = True
            break
        result[host] = hcas

    if use_sysfs:
        log.info("ib_discovery: ibv_devinfo unavailable or empty; falling back to /sys/class/infiniband")
        raw = orch.exec(_SYSFS_CMD)
        result = {}
        for host, output in (raw or {}).items():
            hcas = _parse_sysfs(output)
            log.info("ib_discovery (sysfs): %s -> %s", host, hcas)
            result[host] = hcas
    else:
        for host, hcas in result.items():
            log.info("ib_discovery: %s -> %s", host, hcas)

    # Fail loudly on any empty node.
    empty = [h for h, devs in result.items() if not devs]
    if empty:
        raise RuntimeError(
            f"ib_discovery: no IB HCA devices found on {empty}. "
            "Check that ibv_devinfo is installed and IB drivers are loaded."
        )

    # Fail loudly on asymmetry — a validation suite must surface hardware
    # differences, not silently drop devices.
    lists = [tuple(sorted(devs)) for devs in result.values()]
    if len(set(lists)) > 1:
        detail = "; ".join(f"{h}={devs}" for h, devs in result.items())
        raise RuntimeError(
            f"ib_discovery: asymmetric HCA device lists across nodes ({detail}). "
            "Investigate hardware/driver mismatch before running."
        )

    return result


def validate_ib_hca_preflight(discovered: dict[str, list[str]], requested: list[str]) -> None:
    """Raise if any requested HCA name is absent from any node's discovered list.

    Called when the config provides an explicit ``ib_hca_devices`` list (not
    absent/``"auto"``). Fails loudly naming the missing devices and the node,
    so the operator knows exactly which device is wrong rather than getting a
    cryptic NCCL error later.
    """
    for host, devs in discovered.items():
        missing = [d for d in requested if d not in devs]
        if missing:
            raise RuntimeError(
                f"ib_discovery preflight: requested HCA devices {missing} not found on {host}. Available: {devs}"
            )


def _netdev_for_ip_cmd(ip: str) -> str:
    inner = (
        f"ip -4 -o addr show | awk -v ip={shlex.quote(ip)} "
        "'{split($4,a,\"/\"); if(a[1]==ip) {print $2; exit}}'"
    )
    return f"bash -c {shlex.quote(inner)}"


def _netdev_via_route_cmd(dest_ip: str) -> str:
    inner = (
        f"ip route get {shlex.quote(dest_ip)} 2>/dev/null | awk "
        "'{for(i=1;i<=NF;i++) if($i==\"dev\") {print $(i+1); exit}}'"
    )
    return f"bash -c {shlex.quote(inner)}"


def discover_socket_netdev_name(orch, master_addr: str | None = None) -> str:
    """Return the Linux netdev for NCCL/GLOO socket traffic on a homogeneous cluster.

    On each host, prefers the interface that owns that host's cluster IP (the key
    in ``orch.hosts``). Falls back to the egress interface toward ``master_addr``.
    Requires the same netdev **name** on every node because the suite broadcasts
    one env script to all ranks.

    These are IP netdevs (``ens51f1np1``), not IB HCA names (``mlx5_0``).
    """
    hosts = list(getattr(orch, "hosts", []) or [])
    if not hosts:
        raise RuntimeError("socket_netdev discovery: orchestrator has no hosts")

    master = (master_addr or "").strip() or hosts[0]
    per_host: dict[str, str] = {}
    for host in hosts:
        host_ip = str(host).strip()
        out = orch.exec(_netdev_for_ip_cmd(host_ip), hosts=[host])
        netdev = (out or {}).get(host, "").strip()
        if not netdev:
            out = orch.exec(_netdev_via_route_cmd(master), hosts=[host])
            netdev = (out or {}).get(host, "").strip()
        if not netdev:
            raise RuntimeError(
                f"socket_netdev discovery: no IPv4 netdev on {host} "
                f"(host_ip={host_ip!r}, master_addr={master!r})"
            )
        if _IB_HCA_NETDEV_RE.match(netdev):
            raise RuntimeError(
                f"socket_netdev discovery: {host} resolved {netdev!r}, which looks like an "
                "IB HCA name — set roles.server.ib_netdev to the IP-bearing Linux netdev "
                "(e.g. ens51f1np1), not mlx5_*"
            )
        per_host[host] = netdev
        log.info("socket_netdev discovery: %s -> %s", host, netdev)

    unique = set(per_host.values())
    if len(unique) > 1:
        detail = "; ".join(f"{h}={d}" for h, d in per_host.items())
        raise RuntimeError(
            f"socket_netdev discovery: asymmetric netdev names across nodes ({detail}). "
            "Set roles.server.ib_netdev explicitly when node interface names differ."
        )
    return next(iter(unique))


def resolve_multinode_fabric(
    orch,
    *,
    ib_hca_devices=None,
    ib_netdev=None,
    master_addr=None,
) -> tuple[list[str], str]:
    """Resolve ``NCCL_IB_HCA`` devices and the socket netdev for a multinode run.

    Used by ``test_discover_topology`` (once per lifecycle) and lazily by
    ``InferenceXAtomJob.build_server_cmd`` when a partial ``-k`` filter skips
    the topology test.
    """
    discovered = discover_ib_hca_names(orch)
    if ib_hca_devices and ib_hca_devices != "auto":
        validate_ib_hca_preflight(discovered, ib_hca_devices)
        hcas = list(ib_hca_devices)
    else:
        hcas = list(next(iter(discovered.values())))

    configured = (ib_netdev or "").strip()
    master = (master_addr or "").strip() or orch.hosts[0]
    if configured and configured.lower() != "auto":
        netdev = configured
    else:
        netdev = discover_socket_netdev_name(orch, master_addr=master)
    return hcas, netdev
