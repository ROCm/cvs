"""Resolve roles -> host bindings from cluster.node_dict.

In v1, bindings are deterministic: take roles in declared order, allocate
`role.count` distinct hosts from the cluster's node_dict (in key order).
Fails if total demand exceeds the cluster's node count.
"""

from __future__ import annotations


def resolve_bindings(
    *,
    node_dict: dict[str, dict],
    roles: dict[str, dict],
) -> dict[str, list[str]]:
    """Return {role_name: [host_keys]}.

    Args:
        node_dict: cluster.node_dict, ordered. Keys are host identifiers
                   (as used everywhere else in CVS).
        roles: workload.topology.roles, ordered. Each value has `count`.
    """
    available = list(node_dict.keys())
    total_needed = sum(r["count"] for r in roles.values())
    if total_needed > len(available):
        raise ValueError(
            f"topology demands {total_needed} hosts across {len(roles)} role(s) "
            f"but cluster only has {len(available)}: {available}"
        )
    bindings: dict[str, list[str]] = {}
    cursor = 0
    for role_name, role_spec in roles.items():
        n = role_spec["count"]
        if n < 1:
            raise ValueError(f"role {role_name!r} has count={n} (must be >= 1)")
        bindings[role_name] = available[cursor : cursor + n]
        cursor += n
    return bindings
