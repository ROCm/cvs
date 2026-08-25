"""Derive Node Smoke Tier 1/2/3 test counts from Primus payloads and reports.

Tier 1 (per node, 8-GPU example): ``4`` subprocess checks/GPU + ``7`` node operational
checks = ``32 + 7 = 39``.

Tier 2 (per node, 8-GPU example): ``2`` perf checks/GPU (large GEMM + HBM) + ``1`` local
RCCL all-reduce = ``16 + 1 = 17``.  RCCL is omitted when ``gpus_per_node < 2``.

Tier 3 (cluster-wide): ``27`` individual collector checks from ``preflight --host --gpu
--network`` (validation-tracker catalog).  This is **not** multiplied by node count.

The Primus markdown report aggregates these into ``13`` summary sections (e.g. one
``## CPU`` table for all hosts).  CVS reports the **27** underlying checks, not the
``13`` report sections.

Multi-node Tier 1/2 summaries use ``N tests run per node``.  Tier 3 uses ``N tests run``
(cluster catalog).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TIER1_CHECKS_PER_GPU = 4
TIER1_NODE_OPERATIONAL_COLLECTORS = (
    "gpu_processes",
    "nics",
    "host_limits",
    "gpu_low_level",
    "xgmi",
    "tooling",
    "gpu_visibility",
)

TIER2_CHECKS_PER_GPU = 2
TIER2_RCCL_CHECK = 1

TIER3_TOP_LEVEL_GROUPS = ("host", "gpu", "network")

# Validation-tracker catalog for ``preflight --host --gpu --network`` (27 checks).
# Group prefix matches Primus CLI flags; label matches DTNI / collector finding names.
TIER3_CHECK_CATALOG: Tuple[Tuple[str, str], ...] = (
    ("host", "Host identity"),
    ("host", "Host identity"),
    ("host", "CPU"),
    ("host", "Memory"),
    ("host", "Memory"),
    ("host", "NUMA"),
    ("host", "PCIe inventory"),
    ("host", "PCIe link status"),
    ("host", "PCIe link status"),
    ("host", "PCIe link status"),
    ("gpu", "GPU enumeration"),
    ("gpu", "GPU identity"),
    ("gpu", "GPU occupancy"),
    ("gpu", "GPU / NUMA mapping"),
    ("gpu", "GPU topology"),
    ("gpu", "GPU topology"),
    ("gpu", "GPU perf sanity"),
    ("gpu", "GPU perf sanity"),
    ("network", "Network summary"),
    ("network", "Distributed intent"),
    ("network", "Distributed env"),
    ("network", "Network path"),
    ("network", "Network path"),
    ("network", "InfiniBand / RDMA"),
    ("network", "RCCL / NCCL config"),
    ("network", "RCCL / NCCL config"),
    ("network", "Runtime process group"),
)
TIER3_CATALOG_COUNT = len(TIER3_CHECK_CATALOG)


def tier3_check_catalog() -> List[Dict[str, str]]:
    """Return the Tier 3 validation-tracker check list (group + label per check)."""
    return [{"group": group, "label": label} for group, label in TIER3_CHECK_CATALOG]


def _collector_ran(key: str, value: Any) -> bool:
    if value is None:
        return False
    if key == "dmesg" and isinstance(value, dict) and value.get("error") == "skipped":
        return False
    return True


def _resolve_gpu_count(node_payload: Optional[Dict[str, Any]], gpus_per_node: Optional[int]) -> int:
    if isinstance(node_payload, dict):
        per_gpu = (node_payload.get("tier1") or {}).get("per_gpu") or []
        if per_gpu:
            return len(per_gpu)
    return int(gpus_per_node or 0)


def count_tier1_tests_from_payload(
    node_payload: Optional[Dict[str, Any]],
    *,
    gpus_per_node: Optional[int] = None,
) -> int:
    """Count Tier 1 checks for one node using the validation-tracker catalog."""
    n_gpus = _resolve_gpu_count(node_payload, gpus_per_node)
    if n_gpus <= 0:
        return 0

    count = TIER1_CHECKS_PER_GPU * n_gpus
    if isinstance(node_payload, dict):
        tier1 = node_payload.get("tier1") or {}
        for key in TIER1_NODE_OPERATIONAL_COLLECTORS:
            if _collector_ran(key, tier1.get(key)):
                count += 1
    else:
        count += len(TIER1_NODE_OPERATIONAL_COLLECTORS)
    return count


def count_tier2_tests_from_payload(
    node_payload: Optional[Dict[str, Any]],
    *,
    gpus_per_node: Optional[int] = None,
    tier2_enabled: bool = False,
) -> int:
    """Count Tier 2 checks for one node using the validation-tracker catalog."""
    if not tier2_enabled:
        if not isinstance(node_payload, dict) or not node_payload.get("tier2"):
            return 0

    n_gpus = _resolve_gpu_count(node_payload, gpus_per_node)
    if n_gpus <= 0:
        return 0

    count = TIER2_CHECKS_PER_GPU * n_gpus
    tier2 = node_payload.get("tier2") if isinstance(node_payload, dict) else {}
    if n_gpus > 1 and (tier2_enabled or (tier2 or {}).get("rccl")):
        count += TIER2_RCCL_CHECK
    return count


def _uniform_per_node_values(per_node: Dict[str, Dict[str, int]], key: str) -> Optional[int]:
    values = [entry.get(key, 0) for entry in per_node.values()]
    if not values:
        return None
    if len(set(values)) == 1:
        return values[0]
    return max(values)


def aggregate_node_smoke_test_counts(
    node_smoke_results: Optional[Dict[str, Any]],
    *,
    gpus_per_node: Optional[int] = None,
) -> Dict[str, Any]:
    """Derive per-node Tier 1/2 test counts from a Node Smoke Tier 1 run."""
    node_results = (node_smoke_results or {}).get("node_results") or {}
    tier2_enabled = bool((node_smoke_results or {}).get("tier2_perf"))
    per_node: Dict[str, Dict[str, int]] = {}
    tier1_total = 0
    tier2_total = 0

    for host, result in node_results.items():
        payload = result.get("node_payload") if isinstance(result, dict) else None
        tier1_count = count_tier1_tests_from_payload(payload, gpus_per_node=gpus_per_node)
        tier2_count = count_tier2_tests_from_payload(
            payload,
            gpus_per_node=gpus_per_node,
            tier2_enabled=tier2_enabled,
        )
        tier1_total += tier1_count
        tier2_total += tier2_count
        per_node[host] = {"tier1": tier1_count, "tier2": tier2_count}

    return {
        "tier1_tests_run": _uniform_per_node_values(per_node, "tier1") or 0,
        "tier2_tests_run": _uniform_per_node_values(per_node, "tier2") or 0,
        "tier1_tests_run_total": tier1_total,
        "tier2_tests_run_total": tier2_total,
        "per_node": per_node,
    }


def count_tier3_tests_from_results(tier3_results: Optional[Dict[str, Any]]) -> int:
    """Count Tier 3 collector checks for a Node Smoke Tier 3 cluster run."""
    if not tier3_results or tier3_results.get("skipped"):
        return 0

    node_results = tier3_results.get("node_results") or {}
    if not node_results and not tier3_results.get("report_markdown"):
        checks = tier3_results.get("checks") or []
        if checks:
            groups = [part.strip() for part in str(checks[0]).split(",") if part.strip()]
            if groups:
                return len(groups)
        return 0

    return TIER3_CATALOG_COUNT


def aggregate_tier3_test_counts(tier3_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive cluster-wide Tier 3 test count (not multiplied by node count)."""
    tests_run = count_tier3_tests_from_results(tier3_results)
    total_nodes = len((tier3_results or {}).get("node_results") or {})
    return {
        "tier3_tests_run": tests_run,
        "tier3_tests_run_total": tests_run,
        "tier3_check_catalog": tier3_check_catalog() if tests_run else [],
        "total_nodes": total_nodes,
    }


def format_tests_run_suffix(
    tests_run: Optional[int],
    *,
    per_node: bool = False,
    cluster_wide: bool = False,
    total_nodes: int = 1,
) -> str:
    """Append ``; N tests run`` with optional ``per node`` or ``cluster-wide`` scope."""
    if not tests_run:
        return ""
    label = "test" if tests_run == 1 else "tests"
    if per_node and total_nodes > 1:
        scope = " per node"
    elif cluster_wide and total_nodes > 1:
        scope = " cluster-wide"
    else:
        scope = ""
    return f"; {tests_run} {label} run{scope}"
