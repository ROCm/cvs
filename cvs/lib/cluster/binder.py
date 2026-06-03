"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from cvs.lib.cluster.pool import ClusterPool
from cvs.lib.cluster.topology import node_matches
from cvs.lib.config.base import Topology


class BindResult(BaseModel):
    """Outcome of binding a config's roles to a cluster pool for one cell.

    Downstream (G5b executor) reads ``bindings`` (role -> hostnames) to pick
    SSH targets when ``status == "bound"``. ``skipped`` cells carry a concrete
    ``reason`` so small dev clusters get useful partial coverage instead of
    hard failure.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["bound", "skipped"]
    bindings: Dict[str, List[str]] = Field(default_factory=dict)
    reason: Optional[str] = None


def bind(topology: Topology, pool: ClusterPool) -> BindResult:
    """Deterministic first-fit binder.

    Determinism is content-based, not insertion-order-based: callers may
    assemble the node dict in any order and the same pool contents must yield
    the same BindResult.bindings (this is what makes v2.A reuse-manifests
    sound). We achieve this by iterating candidate hostnames in lexicographic
    order. Within that order each role greedily claims the first ``count``
    nodes that (a) match the selector, (b) have at least ``gpus_per_node`` GPUs,
    and (c) are not already claimed by an earlier role.
    """
    claimed: Set[str] = set()
    bindings: Dict[str, List[str]] = {}
    candidate_order = sorted(pool.nodes.keys())

    for role_name, role in topology.roles.items():
        chosen: List[str] = []
        for hostname in candidate_order:
            if len(chosen) >= role.count:
                break
            if hostname in claimed:
                continue
            node = pool.nodes[hostname]
            if not node_matches(node, role.selector):
                continue
            if node.gpus < role.gpus_per_node:
                continue
            chosen.append(hostname)

        if len(chosen) < role.count:
            reason = (
                f"insufficient_nodes for role {role_name!r} "
                f"(need {role.count} matching selector={role.selector!r} "
                f"with >= {role.gpus_per_node} gpus, have {len(chosen)})"
            )
            return BindResult(status="skipped", bindings={}, reason=reason)

        claimed.update(chosen)
        bindings[role_name] = chosen

    return BindResult(status="bound", bindings=bindings)
