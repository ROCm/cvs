'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``rvs_res_dict`` consumed by the Run Deck status_matrix
builder from RVS stdout CVS already collects per node.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "version_label": "RVS version", "suite": ...},
      "groups": {"<module>": {"nodes": {"<label>": <node record>}}},
    }

Optional per-node ``metrics``, ``series``, and ``heatmaps`` follow that builder's
contract. Parsing never feeds ``classify_output``; a missing or older RVS
measurement leaves those lists empty and the module verdict unchanged.
'''

import re

_STATUSES = ("pass", "fail", "na")

# Suite group names (gst_single, level_config, ...) are not the RVS module token
# printed after "Module name :". LEVEL output carries several modules in one blob.
_GROUP_MODULES = {
    "gst_single": "gst",
    "pebb_single": "pebb",
    "pbqt_single": "pbqt",
    "babel_stream": "babel",
    "iet_stress": "iet",
    "mem_test": "mem",
    "gpu_enumeration": "gpup",
    "peqt_single": "peqt",
    "level_config": "",
}

_PRECISION_MATCH = ("bf16", "fp16", "fp32", "fp64", "fp8")
_PRECISION_ORDER = ("fp8", "fp16", "bf16", "fp32", "fp64")

_MODULE_NAME = re.compile(r"Module name\s*:\s*(?P<module>\S+)", re.I)
_ACTION_NAME = re.compile(r"Action name\s*:\s*(?P<action>\S+)", re.I)
# PEBB/PBQT print "[GPU:: <index> - <device id> - <bdf>]". GST/IET/Babel print the device id only.
_GPU_INDEX = re.compile(r"\[GPU::\s*(\d+)\s*-\s*(\d+)\s*-\s*[0-9A-Fa-f:.]+\]")
_GST_FINAL = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\d+)\]\s*"
    r"GFLOPS\s+(?P<value>[\d.]+)\s+Target GFLOPS:\s+(?P<target>[\d.]+)\s+"
    r"met:\s*(?P<met>TRUE|FALSE)",
    re.I,
)
_IET_POWER = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\d+)\]\s*Power\(W\)\s+(?P<value>[\d.]+)",
    re.I,
)
_IET_PASS = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*\[GPU::\s*(?P<gpu>\d+)\]\s*pass:\s*(?P<met>TRUE|FALSE)",
    re.I,
)


def make_meta(cluster_dict, version, suite_name):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—"
    return {
        "cluster": cluster,
        "version": version or "—",
        "version_label": "RVS version",
        "suite": suite_name or "rvs_cvs",
        "generated_at": "",
    }


def classify_output(output, fail_patterns):
    '''
    Return (status, items) for one node's RVS output.

    A node fails when any caller-supplied pattern matches. The patterns are the
    same ones the suite uses to call fail_test, so the deck verdict matches the
    module check rather than a second parser.
    '''
    text = output if isinstance(output, str) else ""
    items = []
    for pattern in fail_patterns or []:
        if not pattern:
            continue
        try:
            matched = re.search(pattern, text, re.I)
        except re.error:
            matched = None
        if matched:
            items.append(
                {
                    "name": str(pattern),
                    "status": "fail",
                    "message": "matched failure pattern",
                }
            )
    if items:
        return "fail", items
    return "pass", []


def build_node_record(status, items=None, items_summary="", metrics=None, series=None, heatmaps=None):
    '''Assemble one node record for a module.'''
    normalized = str(status or "na").lower()
    if normalized not in _STATUSES:
        normalized = "na"
    record = {
        "status": normalized,
        "items_summary": items_summary or "",
        "items": items or [],
        "errors_json_href": "",
        "log_tarball_href": "",
    }
    if metrics:
        record["metrics"] = metrics
    if series:
        record["series"] = series
    if heatmaps:
        record["heatmaps"] = heatmaps
    return record


def record_group(res_dict, group, node_records, meta=None):
    '''
    Merge one module's per-node records into the accumulating results dict.

    Idempotent per (group, node): a re-run of the same module overwrites.
    '''
    if meta:
        existing = res_dict.get("_meta")
        if not existing:
            res_dict["_meta"] = dict(meta)
        elif meta.get("version") not in (None, "", "—") and existing.get("version") in (None, "", "—"):
            existing["version"] = meta["version"]
    groups = res_dict.setdefault("groups", {})
    group_entry = groups.setdefault(str(group), {"nodes": {}})
    group_entry["nodes"].update(node_records)
    return res_dict


def _hint_module(group):
    key = str(group or "").strip().lower()
    if not key:
        return ""
    if key in _GROUP_MODULES:
        return _GROUP_MODULES[key]
    return key


def _device_labels(text):
    '''Map RVS device ids to ``GPU<index>`` using index-id-bdf triples anywhere in the output.'''
    labels = {}
    for index, device in _GPU_INDEX.findall(text or ""):
        labels.setdefault(device, f"GPU{int(index)}")
    return labels


def _label_for(device, labels):
    device = str(device or "").strip()
    return labels.get(device, device)


def _label_key(label):
    text = str(label)
    if text.startswith("GPU") and text[3:].isdigit():
        return (0, int(text[3:]), text)
    if text.isdigit():
        return (1, int(text), text)
    return (2, 0, text)


def _passed(met):
    return str(met).upper() == "TRUE"


def _precision(action):
    text = str(action or "").lower()
    for token in _PRECISION_MATCH:
        if token in text:
            return token
    return ""


def _metric(name, value, unit, group, threshold=None, direction="", status=""):
    item = {"name": name, "value": float(value), "unit": unit, "group": group}
    if threshold is not None:
        item["threshold"] = float(threshold)
    if direction:
        item["direction"] = direction
    if status:
        item["status"] = status
    return item


def _series(name, points, unit, group, x_label, y_label):
    if not points:
        return None
    return {
        "name": name,
        "points": points,
        "unit": unit,
        "group": group,
        "x_label": x_label,
        "y_label": y_label,
    }


def _take_gst(line, labels, samples):
    match = _GST_FINAL.search(line)
    if not match:
        return False
    precision = _precision(match.group("action")) or str(match.group("action")).strip().lower()
    label = _label_for(match.group("gpu"), labels)
    samples[(precision, label)] = (
        float(match.group("value")),
        float(match.group("target")),
        _passed(match.group("met")),
    )
    return True


def _take_iet(line, labels, peaks, statuses):
    match = _IET_POWER.search(line)
    if match:
        label = _label_for(match.group("gpu"), labels)
        value = float(match.group("value"))
        previous = peaks.get(label)
        if previous is None or value > previous:
            peaks[label] = value
        return True
    match = _IET_PASS.search(line)
    if match:
        label = _label_for(match.group("gpu"), labels)
        statuses[label] = "pass" if _passed(match.group("met")) else "fail"
        return True
    return False


def _gst_performance(samples):
    '''One node-level metric per precision (slowest GPU) plus a per-GPU series.'''
    by_precision = {}
    for (precision, label), sample in samples.items():
        by_precision.setdefault(precision, []).append((label, sample))
    ordered = [name for name in _PRECISION_ORDER if name in by_precision]
    ordered.extend(sorted(name for name in by_precision if name not in _PRECISION_ORDER))
    metrics = []
    series = []
    for precision in ordered:
        rows = sorted(by_precision[precision], key=lambda item: _label_key(item[0]))
        slowest = min(rows, key=lambda item: item[1][0])
        _label, (value, target, _ok) = slowest
        status = "fail" if any(not sample[2] for _label, sample in rows) else "pass"
        metrics.append(_metric(precision, value, "GFLOPS", "gst", threshold=target, direction="higher", status=status))
        chart = _series(
            precision,
            [{"x": label, "y": sample[0]} for label, sample in rows],
            "GFLOPS",
            "gst",
            "GPU",
            "GFLOPS",
        )
        if chart:
            series.append(chart)
    return metrics, series


def _iet_performance(peaks, statuses):
    metrics = []
    points = []
    for label in sorted(peaks, key=_label_key):
        metrics.append(_metric(f"power {label}", peaks[label], "W", "iet", status=statuses.get(label, "")))
        points.append({"x": label, "y": peaks[label]})
    series = []
    chart = _series("power", points, "W", "iet", "GPU", "Power")
    if chart:
        series.append(chart)
    return metrics, series


def parse_node_performance(text, module_hint=""):
    '''
    Pull numeric RVS measurements out of one node's stdout.

    Returns ``{"metrics", "series", "heatmaps"}``. Unknown or empty output
    yields empty lists. Interval GST samples without ``Target GFLOPS`` are
    ignored. IET keeps the peak ``Power(W)`` per GPU; RVS does not print a
    temperature for this module.
    '''
    body = text if isinstance(text, str) else ""
    labels = _device_labels(body)
    gst_samples = {}
    iet_peaks = {}
    iet_status = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _MODULE_NAME.search(line) or _ACTION_NAME.search(line):
            continue
        if _take_gst(line, labels, gst_samples):
            continue
        _take_iet(line, labels, iet_peaks, iet_status)
    metrics, series = _gst_performance(gst_samples)
    iet_metrics, iet_series = _iet_performance(iet_peaks, iet_status)
    return {"metrics": metrics + iet_metrics, "series": series + iet_series, "heatmaps": []}


def record_outputs(res_dict, group, out_dict, fail_patterns, meta=None):
    '''Classify each node's output and merge the module into ``res_dict``.'''
    node_records = {}
    for node, output in (out_dict or {}).items():
        status, items = classify_output(output, fail_patterns)
        if status == "fail":
            summary = f"{len(items)} failure pattern(s)"
        else:
            summary = "passed"
        perf = parse_node_performance(output, module_hint=group)
        node_records[str(node)] = build_node_record(
            status,
            items,
            summary,
            metrics=perf["metrics"],
            series=perf["series"],
            heatmaps=perf["heatmaps"],
        )
    return record_group(res_dict, group, node_records, meta=meta)
