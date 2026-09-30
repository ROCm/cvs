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
    TIER2_CHECK_LABELS,
    TIER2_PER_GPU_CHECKS,
    TIER2_RCCL_METRIC,
    TIER3_CHECK_CATALOG,
    TIER3_TOP_LEVEL_GROUPS,
    _has_number,
    _has_own_verdict,
    _tier2_point_measured,
    aggregate_node_smoke_test_counts,
    tier3_reported_groups,
)

_NOT_REPORTED = "Primus did not report this check"
_TIER3_GROUP_LABELS = {"host": "Host", "gpu": "GPU", "network": "Network"}


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


def _own_status(value):
    """Verdict stored on this value, or None when the value does not state one."""
    if not _has_own_verdict(value):
        return None
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, dict):
        if "status" in value:
            return _status_from_value(value.get("status"), default=None)
        if "ok" in value:
            return _status_from_value(value.get("ok"), default=None)
        if "passed" in value:
            return _status_from_value(value.get("passed"), default=None)
        if value.get("error") == "skipped":
            return "skip"
        return None
    return _status_from_value(value, default=None)


def _slug(text):
    return re.sub(r'_+', '_', re.sub(r'[^0-9a-zA-Z]+', '_', str(text))).strip('_').lower()


def _catalog_entry(node, metric, label, check_id):
    return {"node": node, "metric": metric, "label": label, "id": check_id}


def _gpu_slots(gpus_per_node):
    return range(max(0, int(gpus_per_node or 0)))


def tier1_check_catalog(hosts, gpus_per_node=DEFAULT_GPUS_PER_NODE):
    """One pytest row per configured GPU plus one per node collector.

    The GPU row is the ``per_gpu`` verdict Primus reports. It is not expanded into
    subprocess checks the payload does not name.
    """
    catalog = []
    for host in sorted(hosts or []):
        host_slug = _slug(host)
        for gpu_idx in _gpu_slots(gpus_per_node):
            catalog.append(
                _catalog_entry(
                    host,
                    f"{host}/gpu{gpu_idx}",
                    f"{host} GPU {gpu_idx}",
                    f"{host_slug}-gpu{gpu_idx}",
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
    """One pytest row per Primus group (host, gpu, network).

    Individual collector names from the finding map are not rows: Primus reports
    one status for the group list, and a node-level pass is not a pass of each name.
    """
    return [
        _catalog_entry("cluster", f"cluster/{group}", _TIER3_GROUP_LABELS.get(group, group), group)
        for group in TIER3_TOP_LEVEL_GROUPS
    ]


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


def _configured_gpu_slots(results, reported):
    """Match the collection catalog: configured width, else however many were reported."""
    configured = results.get("gpus_per_node") if isinstance(results, dict) else None
    if configured:
        return int(configured)
    return len(reported)


def build_tier1_metric_rows(node_smoke_results):
    """One pytest-html row per configured GPU and node collector.

    A row passes or fails only when that GPU entry or collector is in the payload,
    or a fail reason names it. A node-level pass does not mark the rest passed.
    """
    if not node_smoke_results or node_smoke_results.get("skipped"):
        return []
    node_results = node_smoke_results.get("node_results") or {}
    rows = []
    for host, result in sorted(node_results.items()):
        payload = _node_payload(result)
        reasons = _node_fail_reasons(result)
        tier1 = payload.get("tier1") if isinstance(payload.get("tier1"), dict) else {}
        per_gpu = list(tier1.get("per_gpu") or [])
        for gpu_idx in range(_configured_gpu_slots(node_smoke_results, per_gpu)):
            entry = per_gpu[gpu_idx] if gpu_idx < len(per_gpu) else None
            gpu_id = _gpu_index(entry, gpu_idx) if isinstance(entry, dict) else gpu_idx
            hits = _reasons_for_key(f"gpu{gpu_id}", reasons) or _reasons_for_key(f"gpu{gpu_idx}", reasons)
            verdict = _own_status(entry)
            if hits:
                status = "fail"
                reason = "; ".join(hits)
            elif verdict:
                status = verdict
                reason = ""
            else:
                status = "skip"
                reason = _NOT_REPORTED
            rows.append(_row(host, f"{host}/gpu{gpu_idx}", f"GPU {gpu_id}", status, reason=reason))
        for collector in TIER1_NODE_OPERATIONAL_COLLECTORS:
            hits = _reasons_for_key(collector, reasons)
            verdict = _own_status(tier1.get(collector)) if collector in tier1 else None
            if hits:
                status = "fail"
                reason = "; ".join(hits)
            elif verdict:
                status = verdict
                reason = ""
            else:
                status = "skip"
                reason = _NOT_REPORTED
            label = TIER1_COLLECTOR_LABELS.get(collector, collector.replace("_", " "))
            rows.append(_row(host, f"{host}/{collector}", label, status, reason=reason))
    return rows


def build_tier2_metric_rows(node_smoke_results):
    """One pytest-html row per Tier 2 GEMM, HBM, and local RCCL check.

    A row passes only when that metric or an explicit verdict is in the payload.
    """
    if not node_smoke_results or node_smoke_results.get("skipped"):
        return []
    if not node_smoke_results.get("tier2_perf"):
        counts = aggregate_node_smoke_test_counts(node_smoke_results)
        if not counts.get("tier2_tests_run"):
            return []
    node_results = node_smoke_results.get("node_results") or {}
    thresholds = node_smoke_results.get("tier2_thresholds") or {}
    rows = []
    for host, result in sorted(node_results.items()):
        payload = _node_payload(result)
        reasons = _node_fail_reasons(result)
        tier2 = payload.get("tier2") if isinstance(payload.get("tier2"), dict) else {}
        per_gpu = [entry for entry in (tier2.get("per_gpu") or []) if isinstance(entry, dict)]
        n_gpus = _configured_gpu_slots(node_smoke_results, per_gpu)
        gemm_spec = (
            {"kind": ">=", "value": thresholds.get("gemm_tflops_min")} if thresholds.get("gemm_tflops_min") else None
        )
        hbm_spec = {"kind": ">=", "value": thresholds.get("hbm_gbs_min")} if thresholds.get("hbm_gbs_min") else None
        rccl_spec = {"kind": ">=", "value": thresholds.get("rccl_gbs_min")} if thresholds.get("rccl_gbs_min") else None
        for gpu_idx in range(n_gpus):
            entry = per_gpu[gpu_idx] if gpu_idx < len(per_gpu) else {}
            gpu_id = _gpu_index(entry, gpu_idx)
            for check_key in TIER2_PER_GPU_CHECKS:
                check_value = entry.get(check_key) if check_key in entry else None
                if (
                    check_value is None
                    and isinstance(tier2.get(check_key), list)
                    and gpu_idx < len(tier2.get(check_key))
                ):
                    check_value = tier2[check_key][gpu_idx]
                hits = _reasons_for_key(check_key, reasons) or _reasons_for_key(
                    "gemm" if check_key == "large_gemm" else "hbm",
                    reasons,
                )
                if hits:
                    status = "fail"
                    reason = "; ".join(hits)
                elif _tier2_point_measured(check_key, check_value, entry):
                    status = _own_status(check_value) or "pass"
                    reason = ""
                else:
                    status = "skip"
                    reason = _NOT_REPORTED
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
                        reason=reason,
                    )
                )
        if n_gpus > 1:
            rccl_value = tier2.get("rccl") if tier2.get("rccl") is not None else tier2.get(TIER2_RCCL_METRIC)
            hits = _reasons_for_key("rccl", reasons) or _reasons_for_key(TIER2_RCCL_METRIC, reasons)
            rccl_measured = _has_own_verdict(rccl_value) or _has_number(
                rccl_value if isinstance(rccl_value, dict) else {}, "rccl_gbs", "gbs"
            )
            if hits:
                status = "fail"
                reason = "; ".join(hits)
            elif rccl_measured:
                status = _own_status(rccl_value) or "pass"
                reason = ""
            else:
                status = "skip"
                reason = _NOT_REPORTED
            rows.append(
                _row(
                    host,
                    f"{host}/{TIER2_RCCL_METRIC}",
                    TIER2_CHECK_LABELS[TIER2_RCCL_METRIC],
                    status,
                    actual=_numeric_field(rccl_value if isinstance(rccl_value, dict) else {}, "rccl_gbs", "gbs"),
                    unit="GB/s",
                    spec=rccl_spec,
                    reason=reason,
                )
            )
    return rows


def _groups_named_by_reason(reason):
    """Map one Primus finding onto host, gpu, or network via the finding catalog."""
    text = str(reason).lower()
    named = set()
    for group, label in TIER3_CHECK_CATALOG:
        if label.lower() in text:
            named.add(group)
    for group in TIER3_TOP_LEVEL_GROUPS:
        if re.search(rf"\b{group}\b", text):
            named.add(group)
    return named


def build_tier3_metric_rows(tier3_results):
    """One pytest-html row per host/GPU/network group Primus was asked to run.

    A group passes only when Primus named it in ``checks=`` and the node status
    is pass. A finding fails the group it names. Every other group is skipped,
    including when the cluster failed without naming a group.
    """
    if not tier3_results or tier3_results.get("skipped"):
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
    reported = set(tier3_reported_groups(tier3_results))
    failed_groups = {}
    for reason in unique_reasons:
        for group in _groups_named_by_reason(reason):
            failed_groups.setdefault(group, []).append(str(reason))
    rows = []
    for group in TIER3_TOP_LEVEL_GROUPS:
        if group in failed_groups:
            status = "fail"
            reason = "; ".join(failed_groups[group])
        elif group in reported and node_status == "pass":
            status = "pass"
            reason = ""
        else:
            status = "skip"
            reason = _NOT_REPORTED
        rows.append(
            _row(
                "cluster",
                f"cluster/{group}",
                _TIER3_GROUP_LABELS.get(group, group),
                status,
                reason=reason,
            )
        )
    return rows


def row_is_failure(row):
    return str(row.get('status') or '').lower() not in ('pass', 'skip', 'record')


def rows_were_measured(rows):
    return any(str(row.get('status') or '').lower() != 'skip' for row in rows)


def tier2_runner_outcome(rows):
    """Pass only when a Tier 2 check was measured and none failed.

    ``tier2_perf`` still collects a row per configured check. If Primus reported
    none of them, the runner must skip rather than pass.
    """
    if any(row_is_failure(row) for row in rows):
        return 'fail'
    if rows and not rows_were_measured(rows):
        return 'skip'
    return 'pass'


def unexplained_tier1_nodes(results, tier2_rows):
    """Nodes whose node_smoke failure is not already a failing Tier 2 check.

    ``failed_nodes`` is the verdict of the single invocation, which includes
    ``--tier2-perf``. A GEMM or HBM miss must not be reported as a Tier 1 failure.
    A node that failed without a Tier 2 row still belongs to Tier 1.
    """
    explained = {row.get('node') for row in tier2_rows if row_is_failure(row)}
    nodes = []
    for node in list(results.get('failed_nodes') or []) + list(results.get('unknown_nodes') or []):
        if node and node not in explained and node not in nodes:
            nodes.append(node)
    return nodes


def tier1_runner_outcome(results, tier1_rows, tier2_rows):
    """Fail Tier 1 from its own rows, not from a Tier 2 threshold miss."""
    if any(row_is_failure(row) for row in tier1_rows):
        return 'fail'
    if unexplained_tier1_nodes(results, tier2_rows):
        return 'fail'
    if tier1_rows and not rows_were_measured(tier1_rows):
        return 'skip'
    return 'pass'


def tier_runner_row_hidden(runner_outcome, checks_collected, check_failed, check_passed):
    """Hide the tier runner row only when a check row already shows the same verdict.

    A failed runner stays in the HTML when every check row skipped or passed, so a
    setup error or an unparsed node failure cannot disappear behind the check list.
    A passed runner stays when no check row passed, so a node-level pass is not
    replaced by a list of unmeasured skips.
    """
    if not checks_collected:
        return False
    if runner_outcome == "failed":
        return bool(check_failed)
    if runner_outcome == "passed":
        return bool(check_passed)
    return True
