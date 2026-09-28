"""Node Smoke Tier 1/2/3 check catalogs and per-check pass/fail rows.

The catalog functions describe every check a tier *will* run using only the cluster
node list and GPU count, so pytest can parametrize one test row per check at
collection time. The row builders then resolve each catalog entry against the Primus
payloads captured during the run. Both sides key off the same ``metric`` string.
"""

import re

from cvs.lib.preflight.node_smoke_counts import (
    DEFAULT_GPUS_PER_NODE,
    TIER1_COLLECTOR_LABELS,
    TIER1_NODE_OPERATIONAL_COLLECTORS,
    TIER1_PER_GPU_CHECKS,
    TIER2_CHECK_LABELS,
    TIER2_PER_GPU_CHECKS,
    TIER2_RCCL_METRIC,
    TIER3_CHECK_CATALOG,
    _collector_ran,
    _resolve_gpu_count,
    aggregate_node_smoke_test_counts,
    count_tier3_tests_from_results,
)

_GPU_META_KEYS = {
    "gpu",
    "id",
    "index",
    "device",
    "status",
    "ok",
    "passed",
    "bdf",
    "pci",
    "uuid",
    "name",
    "message",
}


def _status_from_value(value, default="pass"):
    if value is None:
        return default
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, dict):
        if "status" in value:
            return _status_from_value(value.get("status"), default)
        if "ok" in value:
            return _status_from_value(value.get("ok"), default)
        if "passed" in value:
            return _status_from_value(value.get("passed"), default)
        if value.get("error") == "skipped":
            return "skip"
        return default
    token = str(value).strip().lower()
    if token in ("pass", "passed", "ok", "true", "success"):
        return "pass"
    if token in ("skip", "skipped"):
        return "skip"
    if token in ("fail", "failed", "error", "false", "unknown"):
        return "fail"
    return default


def _gpu_index(entry, fallback):
    if not isinstance(entry, dict):
        return fallback
    for key in ("gpu", "id", "index", "device"):
        if entry.get(key) is not None:
            return entry[key]
    return fallback


def _reasons_for_key(key, reasons):
    prefix = str(key).lower() + ":"
    hits = []
    for reason in reasons or []:
        text = str(reason)
        if text.lower().startswith(prefix):
            hits.append(text)
    return hits


def _row(node, metric, label, status, actual=None, unit=None, spec=None, reason=""):
    return {
        "node": node,
        "metric": metric,
        "label": label,
        "status": status,
        "actual": actual,
        "unit": unit,
        "spec": spec,
        "reason": reason,
        "enforced": status != "record",
    }


def _node_payload(result):
    if not isinstance(result, dict):
        return {}
    payload = result.get("node_payload")
    return payload if isinstance(payload, dict) else {}


def _node_fail_reasons(result):
    reasons = []
    if isinstance(result, dict):
        reasons.extend(result.get("fail_reasons") or [])
        payload = result.get("node_payload")
        if isinstance(payload, dict):
            reasons.extend(payload.get("fail_reasons") or [])
    seen = set()
    unique = []
    for reason in reasons:
        text = str(reason)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _unattributed_reasons(reasons, known_keys):
    """Node-level failures Primus did not prefix with a check name.

    Without these, a node that never produced a parseable payload fails every check
    row with no explanation of why.
    """
    attributed = set()
    for key in known_keys:
        attributed.update(_reasons_for_key(key, reasons))
    return [str(reason) for reason in reasons or [] if str(reason) not in attributed]


def _named_gpu_checks(entry, catalog):
    if not isinstance(entry, dict):
        return [(key, None) for key in catalog]
    checks = entry.get("checks") or entry.get("subprocess_checks")
    if isinstance(checks, list) and checks:
        rows = []
        for idx, check in enumerate(checks[: len(catalog)]):
            key = catalog[idx]
            rows.append((key, check))
        while len(rows) < len(catalog):
            rows.append((catalog[len(rows)], entry))
        return rows
    named = [(key, entry[key]) for key in entry if key not in _GPU_META_KEYS]
    if len(named) >= len(catalog):
        mapped = []
        used = set()
        for catalog_key in catalog:
            match = next((item for item in named if item[0] == catalog_key), None)
            if match:
                mapped.append(match)
                used.add(match[0])
            else:
                leftover = next((item for item in named if item[0] not in used), None)
                if leftover:
                    mapped.append((catalog_key, leftover[1]))
                    used.add(leftover[0])
                else:
                    mapped.append((catalog_key, entry))
        return mapped
    return [(key, entry) for key in catalog]


def _slug(text):
    return re.sub(r'_+', '_', re.sub(r'[^0-9a-zA-Z]+', '_', str(text))).strip('_').lower()


def _catalog_entry(node, metric, label, check_id):
    return {"node": node, "metric": metric, "label": label, "id": check_id}


def _gpu_slots(gpus_per_node):
    return range(max(0, int(gpus_per_node or 0)))


def tier1_check_catalog(hosts, gpus_per_node=DEFAULT_GPUS_PER_NODE):
    """Every Tier 1 check that will run, as one catalog entry per pytest row."""
    catalog = []
    for host in sorted(hosts or []):
        host_slug = _slug(host)
        for gpu_idx in _gpu_slots(gpus_per_node):
            for check_key in TIER1_PER_GPU_CHECKS:
                catalog.append(
                    _catalog_entry(
                        host,
                        f"{host}/gpu{gpu_idx}/{check_key}",
                        f"{host} GPU {gpu_idx} {check_key.replace('_', ' ')}",
                        f"{host_slug}-gpu{gpu_idx}-{_slug(check_key)}",
                    )
                )
        for collector in TIER1_NODE_OPERATIONAL_COLLECTORS:
            catalog.append(
                _catalog_entry(
                    host,
                    f"{host}/{collector}",
                    f"{host} {TIER1_COLLECTOR_LABELS.get(collector, collector)}",
                    f"{host_slug}-{_slug(collector)}",
                )
            )
    return catalog


def tier2_check_catalog(hosts, gpus_per_node=DEFAULT_GPUS_PER_NODE):
    """Every Tier 2 perf-sanity check that will run when ``tier2_perf`` is enabled."""
    catalog = []
    for host in sorted(hosts or []):
        host_slug = _slug(host)
        slots = list(_gpu_slots(gpus_per_node))
        for gpu_idx in slots:
            for check_key in TIER2_PER_GPU_CHECKS:
                catalog.append(
                    _catalog_entry(
                        host,
                        f"{host}/gpu{gpu_idx}/{check_key}",
                        f"{host} GPU {gpu_idx} {TIER2_CHECK_LABELS.get(check_key, check_key)}",
                        f"{host_slug}-gpu{gpu_idx}-{_slug(check_key)}",
                    )
                )
        # Primus only runs the local all-reduce when the node has peers to reduce across.
        if len(slots) > 1:
            catalog.append(
                _catalog_entry(
                    host,
                    f"{host}/{TIER2_RCCL_METRIC}",
                    f"{host} {TIER2_CHECK_LABELS[TIER2_RCCL_METRIC]}",
                    f"{host_slug}-{_slug(TIER2_RCCL_METRIC)}",
                )
            )
    return catalog


def tier3_check_catalog_entries():
    """Every cluster-wide Tier 3 collector check, as one catalog entry per pytest row."""
    catalog = []
    for group, label, occurrence in _catalog_occurrences():
        suffix = f" ({occurrence})" if occurrence > 1 else ""
        catalog.append(
            _catalog_entry(
                "cluster",
                f"cluster/{group}/{label}/{occurrence}",
                f"{group}: {label}{suffix}",
                f"{group}-{_slug(label)}-{occurrence}",
            )
        )
    return catalog


def _numeric_field(entry, *keys):
    if not isinstance(entry, dict):
        return None
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, dict):
            nested = _numeric_field(value, "value", "actual", "tflops", "gbs", "gbps")
            if nested is not None:
                return nested
    return None


def build_tier1_metric_rows(node_smoke_results):
    """One pytest-html subtest row per Tier 1 catalog check on each node."""
    if not node_smoke_results or node_smoke_results.get("skipped"):
        return []
    node_results = node_smoke_results.get("node_results") or {}
    gpus_per_node = node_smoke_results.get("gpus_per_node")
    rows = []
    for host, result in sorted(node_results.items()):
        payload = _node_payload(result)
        reasons = _node_fail_reasons(result)
        node_status = _status_from_value(
            result.get("status") if isinstance(result, dict) else None,
            default="fail",
        )
        tier1 = payload.get("tier1") if isinstance(payload.get("tier1"), dict) else {}
        per_gpu = list(tier1.get("per_gpu") or [])
        n_gpus = _resolve_gpu_count(payload, gpus_per_node)
        if n_gpus <= 0 and per_gpu:
            n_gpus = len(per_gpu)
        node_reasons = _unattributed_reasons(
            reasons,
            (
                *TIER1_PER_GPU_CHECKS,
                *TIER1_NODE_OPERATIONAL_COLLECTORS,
                *(f"gpu{idx}" for idx in range(max(n_gpus, len(per_gpu)))),
            ),
        )
        for gpu_idx in range(n_gpus):
            entry = per_gpu[gpu_idx] if gpu_idx < len(per_gpu) else {}
            gpu_id = _gpu_index(entry, gpu_idx)
            gpu_status = _status_from_value(entry, default=node_status)
            for check_key, check_value in _named_gpu_checks(entry, TIER1_PER_GPU_CHECKS):
                status = _status_from_value(check_value, default=gpu_status)
                hits = _reasons_for_key(check_key, reasons) or _reasons_for_key(f"gpu{gpu_id}", reasons)
                if hits and status == "pass":
                    status = "fail"
                label = f"GPU {gpu_id} {check_key.replace('_', ' ')}"
                rows.append(
                    _row(
                        host,
                        f"{host}/gpu{gpu_idx}/{check_key}",
                        label,
                        status,
                        reason="; ".join(hits or (node_reasons if status == "fail" else [])),
                    )
                )
        for collector in TIER1_NODE_OPERATIONAL_COLLECTORS:
            if n_gpus <= 0 and not _collector_ran(collector, tier1.get(collector)):
                continue
            value = tier1.get(collector)
            status = _status_from_value(value, default="pass" if node_status == "pass" else node_status)
            hits = _reasons_for_key(collector, reasons)
            if hits:
                status = "fail"
            label = TIER1_COLLECTOR_LABELS.get(collector, collector.replace("_", " "))
            rows.append(
                _row(
                    host,
                    f"{host}/{collector}",
                    label,
                    status,
                    reason="; ".join(hits or (node_reasons if status == "fail" else [])),
                )
            )
    return rows


def build_tier2_metric_rows(node_smoke_results):
    """One pytest-html subtest row per Tier 2 catalog check on each node."""
    if not node_smoke_results or node_smoke_results.get("skipped"):
        return []
    if not node_smoke_results.get("tier2_perf"):
        counts = aggregate_node_smoke_test_counts(node_smoke_results)
        if not counts.get("tier2_tests_run"):
            return []
    node_results = node_smoke_results.get("node_results") or {}
    thresholds = node_smoke_results.get("tier2_thresholds") or {}
    gpus_per_node = node_smoke_results.get("gpus_per_node")
    rows = []
    for host, result in sorted(node_results.items()):
        payload = _node_payload(result)
        reasons = _node_fail_reasons(result)
        node_status = _status_from_value(
            result.get("status") if isinstance(result, dict) else None,
            default="fail",
        )
        tier2 = payload.get("tier2") if isinstance(payload.get("tier2"), dict) else {}
        per_gpu = list(tier2.get("per_gpu") or (payload.get("tier1") or {}).get("per_gpu") or [])
        n_gpus = _resolve_gpu_count(payload, gpus_per_node)
        if n_gpus <= 0 and per_gpu:
            n_gpus = len(per_gpu)
        node_reasons = _unattributed_reasons(
            reasons,
            (*TIER2_PER_GPU_CHECKS, TIER2_RCCL_METRIC, "gemm", "hbm", "rccl"),
        )
        gemm_spec = (
            {"kind": ">=", "value": thresholds.get("gemm_tflops_min")} if thresholds.get("gemm_tflops_min") else None
        )
        hbm_spec = {"kind": ">=", "value": thresholds.get("hbm_gbs_min")} if thresholds.get("hbm_gbs_min") else None
        rccl_spec = {"kind": ">=", "value": thresholds.get("rccl_gbs_min")} if thresholds.get("rccl_gbs_min") else None
        for gpu_idx in range(n_gpus):
            entry = per_gpu[gpu_idx] if gpu_idx < len(per_gpu) else {}
            gpu_id = _gpu_index(entry, gpu_idx)
            gpu_status = _status_from_value(entry, default=node_status)
            for check_key in TIER2_PER_GPU_CHECKS:
                check_value = entry.get(check_key) if isinstance(entry, dict) else None
                if (
                    check_value is None
                    and isinstance(tier2.get(check_key), list)
                    and gpu_idx < len(tier2.get(check_key))
                ):
                    check_value = tier2[check_key][gpu_idx]
                status = _status_from_value(check_value if check_value is not None else entry, default=gpu_status)
                hits = _reasons_for_key(check_key, reasons) or _reasons_for_key(
                    "gemm" if check_key == "large_gemm" else "hbm",
                    reasons,
                )
                if hits:
                    status = "fail"
                if check_key == "large_gemm":
                    actual = _numeric_field(
                        check_value if isinstance(check_value, dict) else entry, "gemm_tflops", "tflops", "large_gemm"
                    )
                    unit = "TFLOPS"
                    spec = gemm_spec
                else:
                    actual = _numeric_field(
                        check_value if isinstance(check_value, dict) else entry, "hbm_gbs", "gbs", "hbm_d2d"
                    )
                    unit = "GB/s"
                    spec = hbm_spec
                label = f"GPU {gpu_id} {TIER2_CHECK_LABELS.get(check_key, check_key)}"
                rows.append(
                    _row(
                        host,
                        f"{host}/gpu{gpu_idx}/{check_key}",
                        label,
                        status,
                        actual=actual,
                        unit=unit,
                        spec=spec,
                        reason="; ".join(hits or (node_reasons if status == "fail" else [])),
                    )
                )
        if n_gpus > 1:
            rccl_value = tier2.get("rccl") or tier2.get(TIER2_RCCL_METRIC)
            status = _status_from_value(rccl_value, default=node_status)
            hits = _reasons_for_key("rccl", reasons) or _reasons_for_key(TIER2_RCCL_METRIC, reasons)
            if hits:
                status = "fail"
            rows.append(
                _row(
                    host,
                    f"{host}/{TIER2_RCCL_METRIC}",
                    TIER2_CHECK_LABELS[TIER2_RCCL_METRIC],
                    status,
                    actual=_numeric_field(rccl_value if isinstance(rccl_value, dict) else tier2, "rccl_gbs", "gbs"),
                    unit="GB/s",
                    spec=rccl_spec,
                    reason="; ".join(hits or (node_reasons if status == "fail" else [])),
                )
            )
    return rows


def _catalog_occurrences():
    counts = {}
    for group, label in TIER3_CHECK_CATALOG:
        counts[(group, label)] = counts.get((group, label), 0) + 1
        yield group, label, counts[(group, label)]


def _reason_matches_label(label, reasons):
    needle = label.lower()
    hits = []
    for reason in reasons or []:
        text = str(reason)
        if needle in text.lower():
            hits.append(text)
    return hits


def build_tier3_metric_rows(tier3_results):
    """One pytest-html subtest row per cluster-wide Tier 3 catalog check."""
    if not tier3_results or tier3_results.get("skipped"):
        return []
    if not count_tier3_tests_from_results(tier3_results):
        return []
    node_results = tier3_results.get("node_results") or {}
    reasons = []
    node_status = "pass"
    for host, result in sorted(node_results.items()):
        reasons.extend(_node_fail_reasons(result))
        status = _status_from_value(result.get("status") if isinstance(result, dict) else None, default="pass")
        if status == "fail":
            node_status = "fail"
    for host in tier3_results.get("failed_nodes") or []:
        node_status = "fail"
        result = node_results.get(host) or {}
        extra = _node_fail_reasons(result)
        reasons.extend(extra or [f"{host}: FAIL"])
    seen_reasons = set()
    unique_reasons = []
    for reason in reasons:
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        unique_reasons.append(reason)
    label_matched = any(_reason_matches_label(label, unique_reasons) for _, label in TIER3_CHECK_CATALOG)
    rows = []
    for group, label, occurrence in _catalog_occurrences():
        hits = _reason_matches_label(label, unique_reasons)
        if node_status == "pass":
            status = "pass"
        elif hits:
            status = "fail"
        elif label_matched:
            status = "pass"
        else:
            # Primus failed without naming a collector, so no check can be called clean.
            status = "fail"
            hits = unique_reasons
        suffix = f" ({occurrence})" if occurrence > 1 else ""
        rows.append(
            _row(
                "cluster",
                f"cluster/{group}/{label}/{occurrence}",
                f"{group}: {label}{suffix}",
                status,
                reason="; ".join(hits),
            )
        )
    return rows
