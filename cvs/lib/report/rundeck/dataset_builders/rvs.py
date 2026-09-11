'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

RVS dataset builder: module/GPU records → table, gate matrix, and bar charts.
'''

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder

_CHARTS = (
    ("gflops", "gst_gflops"),
    ("pcie_gbps", "pebb_gbps"),
    ("p2p_gbps", "pbqt_gbps"),
    ("babel_mbytes_s", "babel_mbytes_s"),
    ("power_w", "iet_power_w"),
)


def _records(sources):
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        return list(results.get("records") or [])
    return []


def _gate_tier(rec):
    passed = rec.get("passed")
    if passed is True:
        return "pass"
    if passed is False:
        return "fail"
    return "record"


def _gate_matrix(records):
    grouped = {}
    for rec in records:
        label = " · ".join(p for p in (rec.get("node"), rec.get("module"), rec.get("gpu")) if p)
        if not label:
            continue
        row = grouped.setdefault(
            label,
            {"label": label, "cell_id": label, "concurrency": 0, "tiers": {"result": "record"}},
        )
        tier = _gate_tier(rec)
        cur = row["tiers"]["result"]
        if tier == "fail" or cur == "fail":
            row["tiers"]["result"] = "fail"
        elif tier == "pass" and cur != "fail":
            row["tiers"]["result"] = "pass"
    return [grouped[k] for k in sorted(grouped)]


def _point_label(rec):
    node = rec.get("node") or ""
    gpu = rec.get("gpu") or ""
    dst = rec.get("dst")
    core = f"{gpu}->{dst}" if dst else gpu
    if node and core:
        return f"{node}/{core}"
    return node or core or ""


def _series_key(rec, metric):
    if metric == "babel_mbytes_s":
        return rec.get("kernel") or rec.get("action") or metric
    return rec.get("action") or metric


def _series_for(records, metric):
    by_key = {}
    for rec in records:
        if rec.get("metric") != metric or rec.get("value") is None:
            continue
        by_key.setdefault(_series_key(rec, metric), []).append(rec)
    out = {}
    for key, recs in by_key.items():
        recs = sorted(recs, key=lambda r: (str(r.get("node")), str(r.get("gpu")), str(r.get("dst") or "")))
        points = [(_point_label(r) or i, r["value"]) for i, r in enumerate(recs)]
        if points:
            out[key] = [{"label": key, "points": points}]
    return out


def _overall_status(records):
    if any(rec.get("passed") is False for rec in records):
        return "fail"
    if any(rec.get("passed") is True for rec in records):
        return "pass"
    return "record"


def _table(records):
    headers = ["Node", "GPU", "Module", "Action", "Metric", "Value", "Unit", "Target", "Pass"]
    rows = []
    for rec in records:
        passed = rec.get("passed")
        if passed is True:
            pass_s = "TRUE"
        elif passed is False:
            pass_s = "FALSE"
        else:
            pass_s = "—"
        val = rec.get("value")
        tgt = rec.get("target")
        rows.append(
            [
                rec.get("node") or "—",
                rec.get("gpu") or "—",
                rec.get("module") or "—",
                rec.get("action") or "—",
                rec.get("metric") or "—",
                val if val is not None else "—",
                rec.get("unit") or "—",
                tgt if tgt is not None else "—",
                pass_s,
            ]
        )
    return {"headers": headers, "rows": rows}


@register_dataset_builder("rvs")
def build_rvs_datasets(sources, profile):
    records = _records(sources)
    charts = {chart_id: _series_for(records, metric) for metric, chart_id in _CHARTS}
    return {
        "records": records,
        "charts": charts,
        "gate_matrix": _gate_matrix(records),
        "results_table": _table(records),
        "overall_status": _overall_status(records),
        "metric_tier_order": tuple((profile or {}).get("tier_order") or ("result",)),
    }
