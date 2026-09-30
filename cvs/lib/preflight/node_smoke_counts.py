"""Derive Node Smoke Tier 1/2/3 test counts from Primus payloads and reports.

Counts include only checks Primus reported. A node-level PASS does not add the
configured GPU slots, the Tier 1 subprocess list, or the Tier 3 finding map.

Tier 1 counts one verdict per GPU entry in ``tier1.per_gpu`` plus each node
operational collector whose key is present.

Tier 2 counts a GEMM or HBM check only when that GPU entry carries the metric or
its own verdict, and counts local RCCL only when the payload includes it.

Tier 3 counts the host/GPU/network groups named in Primus ``checks=`` output.
``TIER3_CHECK_CATALOG`` maps a finding's text onto one of those groups; its length
is not a number of tests run.

Multi-node Tier 1/2 summaries use ``N tests run per node``.  Tier 3 uses ``N tests run``
for the groups that were reported.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Catalogs are built at pytest collection time, before any node is contacted, so an
# unspecified GPU count still yields the documented 8-GPU check list.
DEFAULT_GPUS_PER_NODE = 8

TIER1_NODE_OPERATIONAL_COLLECTORS = (
    "gpu_processes",
    "nics",
    "host_limits",
    "gpu_low_level",
    "xgmi",
    "tooling",
    "gpu_visibility",
)
TIER1_COLLECTOR_LABELS = {
    "gpu_processes": "GPU processes",
    "nics": "RDMA NICs",
    "host_limits": "Host limits",
    "gpu_low_level": "GPU low-level",
    "xgmi": "xGMI",
    "tooling": "Tooling",
    "gpu_visibility": "GPU visibility",
}

TIER2_PER_GPU_CHECKS = (
    "large_gemm",
    "hbm_d2d",
)
TIER2_CHECKS_PER_GPU = len(TIER2_PER_GPU_CHECKS)
TIER2_RCCL_CHECK = 1
TIER2_RCCL_METRIC = "local_rccl"
TIER2_CHECK_LABELS = {
    "large_gemm": "Large GEMM TFLOPS",
    "hbm_d2d": "HBM D2D bandwidth",
    "local_rccl": "Local RCCL all-reduce",
}

TIER3_TOP_LEVEL_GROUPS = ("host", "gpu", "network")

# Finding text -> host/gpu/network group. Used to attribute a Primus FAIL line to the
# group that produced it. This is not a list of checks CVS marks passed.
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
    """Return the finding-to-group map (not the list of checks a run executed)."""
    return [{"group": group, "label": label} for group, label in TIER3_CHECK_CATALOG]


def tier3_reported_groups(tier3_results: Optional[Dict[str, Any]]) -> List[str]:
    """Host/GPU/network groups named by Primus ``checks=`` output, in flag order."""
    if not isinstance(tier3_results, dict):
        return []
    sources: List[Any] = []
    if isinstance(tier3_results.get("checks"), list):
        sources.extend(tier3_results["checks"])
    for node_result in (tier3_results.get("node_results") or {}).values():
        if isinstance(node_result, dict) and isinstance(node_result.get("checks"), list):
            sources.extend(node_result["checks"])
    found = []
    for item in sources:
        for token in str(item).split(","):
            name = token.strip().lower()
            if name in TIER3_TOP_LEVEL_GROUPS and name not in found:
                found.append(name)
    return found


def _collector_ran(key: str, value: Any) -> bool:
    if value is None:
        return False
    if key == "dmesg" and isinstance(value, dict) and value.get("error") == "skipped":
        return False
    return True


def _has_own_verdict(value: Any) -> bool:
    """True when this value carries a pass/fail/skip, rather than inheriting one."""
    if isinstance(value, bool):
        return True
    if isinstance(value, dict):
        if any(key in value for key in ("status", "ok", "passed")):
            return True
        return value.get("error") == "skipped"
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in {
        "pass",
        "passed",
        "ok",
        "true",
        "success",
        "skip",
        "skipped",
        "fail",
        "failed",
        "error",
        "false",
        "unknown",
    }


def _has_number(entry: Any, *keys: str) -> bool:
    if not isinstance(entry, dict):
        return False
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, dict) and _has_number(value, "value", "actual", "tflops", "gbs", "gbps"):
            return True
    return False


def _tier2_point_measured(check_key: str, value: Any, entry: Any) -> bool:
    if _has_own_verdict(value):
        return True
    number_keys = (
        ("gemm_tflops", "tflops", "large_gemm") if check_key == "large_gemm" else ("hbm_gbs", "gbs", "hbm_d2d")
    )
    if isinstance(value, dict) and _has_number(value, *number_keys):
        return True
    return value is None and _has_number(entry, *number_keys)


def count_tier1_tests_from_payload(
    node_payload: Optional[Dict[str, Any]],
    *,
    gpus_per_node: Optional[int] = None,
) -> int:
    """Count Tier 1 GPU verdicts and collectors present in one node's payload.

    ``gpus_per_node`` is accepted for callers that know the configured width. It
    does not add rows the payload never reported.
    """
    del gpus_per_node
    if not isinstance(node_payload, dict):
        return 0
    tier1 = node_payload.get("tier1") or {}
    if not isinstance(tier1, dict):
        return 0

    count = sum(1 for entry in tier1.get("per_gpu") or [] if _has_own_verdict(entry))
    for key in TIER1_NODE_OPERATIONAL_COLLECTORS:
        if key in tier1 and _collector_ran(key, tier1.get(key)):
            count += 1
    return count


def count_tier2_tests_from_payload(
    node_payload: Optional[Dict[str, Any]],
    *,
    gpus_per_node: Optional[int] = None,
    tier2_enabled: bool = False,
) -> int:
    """Count Tier 2 metrics actually present on one node."""
    del gpus_per_node
    if not isinstance(node_payload, dict):
        return 0
    tier2 = node_payload.get("tier2") if isinstance(node_payload.get("tier2"), dict) else {}
    if not tier2_enabled and not tier2:
        return 0
    if not tier2:
        return 0

    per_gpu = [entry for entry in (tier2.get("per_gpu") or []) if isinstance(entry, dict)]
    width = len(per_gpu)
    if width == 0:
        series_lengths = [len(tier2[key]) for key in TIER2_PER_GPU_CHECKS if isinstance(tier2.get(key), list)]
        width = max(series_lengths, default=0)

    count = 0
    for idx in range(width):
        entry = per_gpu[idx] if idx < len(per_gpu) else {}
        for check_key in TIER2_PER_GPU_CHECKS:
            value = entry.get(check_key) if check_key in entry else None
            if value is None and isinstance(tier2.get(check_key), list) and idx < len(tier2[check_key]):
                value = tier2[check_key][idx]
            if _tier2_point_measured(check_key, value, entry):
                count += 1

    rccl = tier2.get("rccl") if tier2.get("rccl") is not None else tier2.get(TIER2_RCCL_METRIC)
    rccl_measured = _has_own_verdict(rccl) or _has_number(rccl if isinstance(rccl, dict) else {}, "rccl_gbs", "gbs")
    if width > 1 and rccl_measured:
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
    """Count Tier 3 groups Primus named, not the finding-map length."""
    if not tier3_results or tier3_results.get("skipped"):
        return 0
    return len(tier3_reported_groups(tier3_results))


def aggregate_tier3_test_counts(tier3_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive cluster-wide Tier 3 test count (not multiplied by node count)."""
    tests_run = count_tier3_tests_from_results(tier3_results)
    total_nodes = len((tier3_results or {}).get("node_results") or {})
    return {
        "tier3_tests_run": tests_run,
        "tier3_tests_run_total": tests_run,
        "tier3_check_catalog": [{"group": group} for group in tier3_reported_groups(tier3_results)]
        if tests_run
        else [],
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
