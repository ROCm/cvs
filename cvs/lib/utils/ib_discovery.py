'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InfiniBand HCA discovery via ibv_devinfo.

Shared across suites (vllm, rccl, inferencemax). Infrastructure step, not
a benchmark step: topology is stable within a run and should be probed once.
'''

from __future__ import annotations

import re

from cvs.lib import globals

log = globals.log

_HCA_RE = re.compile(r"hca_id:\s*(\S+)")


def discover_ib_hca_names(orch) -> dict[str, list[str]]:
    """Return {host: [hca_name, ...]} for all hosts in orch.

    Runs `ibv_devinfo -l` inside the container on every host and parses
    the `hca_id:` lines. Returns HCA names (e.g. ``rdma0``, ``mlx5_0``),
    which are correct for ``NCCL_IB_HCA``. These are NOT Linux netdev names
    (``ens51f1np1``) -- those belong in ``ib_netdev`` in the suite config.

    Raises ``RuntimeError`` if:
    - any host returns an empty HCA list (indicates missing driver or no IB
      hardware on that node), or
    - the HCA lists are asymmetric across nodes (a hardware/driver mismatch
      must surface loudly, not be silently papered over by intersection).
    """
    raw = orch.exec("ibv_devinfo -l")
    result: dict[str, list[str]] = {}
    for host, output in (raw or {}).items():
        hcas = _HCA_RE.findall(output or "")
        log.info("ib_discovery: %s -> %s", host, hcas)
        result[host] = hcas

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
                f"ib_discovery preflight: requested HCA devices {missing} not found on {host}. "
                f"Available: {devs}"
            )
