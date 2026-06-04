"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cvs.lib.cluster.pool import ClusterPool
from cvs.lib.config.base import Topology


class BindResult(BaseModel):
    """Outcome of binding a config's roles to a cluster pool for one cell.

    Downstream reads ``bindings`` (role -> hostnames) to pick SSH targets
    when ``status == "bound"``. ``skipped`` cells carry a concrete
    ``reason`` so small dev clusters get useful partial coverage instead of
    hard failure.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["bound", "skipped"]
    bindings: Dict[str, List[str]] = Field(default_factory=dict)
    reason: Optional[str] = None


def bind(topology: Topology, pool: ClusterPool) -> BindResult:
    """Deterministic first-N lexicographic claim, partitioned across roles.

    Pre-DTNI semantics: the cluster file is just "what hosts I can reach";
    role placement is "give me N hosts" with no GPU floor (runtime
    rocm-smi probe in ``prepare`` validates the actual GPU count) and no
    label selector (workload portability is the test author's problem,
    not the schema's). Determinism is content-based: callers may assemble
    the node dict in any order and the same pool contents yield the same
    BindResult.bindings -- this is what makes v2.A reuse-manifests sound.
    We achieve this by iterating candidate hostnames in lexicographic order
    and partitioning the first ``sum(role.count for role in roles)`` hosts
    into role buckets in declaration order.

    A binding is skipped (not raised) when the pool has fewer hosts than
    ``sum(role.count)``; downstream tier tests treat skipped as a
    skip-with-reason, mirroring the old behavior.
    """
    needed = sum(role.count for role in topology.roles.values())
    candidates = sorted(pool.nodes.keys())
    if len(candidates) < needed:
        return BindResult(
            status="skipped",
            bindings={},
            reason=(f"insufficient_nodes: need {needed} (sum of role counts), pool has {len(candidates)}"),
        )

    bindings: Dict[str, List[str]] = {}
    cursor = 0
    for role_name, role in topology.roles.items():
        chosen = candidates[cursor : cursor + role.count]
        bindings[role_name] = chosen
        cursor += role.count

    return BindResult(status="bound", bindings=bindings)
