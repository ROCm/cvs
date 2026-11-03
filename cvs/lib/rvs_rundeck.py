'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``rvs_res_dict`` consumed by the Run Deck status_matrix
builder from RVS stdout CVS already collects per node.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "version_label": "RVS version", "suite": ...},
      "groups": {"<executed CVS test>": {"nodes": {"<label>": <node record>}}},
    }

Optional per-node ``metrics``, ``series``, and ``heatmaps`` follow that builder's
contract. Parsing never feeds ``classify_output``; a missing or older RVS
measurement leaves those lists empty and the test-group verdict unchanged.

For RVS 1.3 or newer with a nonzero test level, the matrix normally contains
``gpu_enumeration`` and one ``level_config`` group. Measurements from several
RVS modules can be attached to that LEVEL cell. Individual module groups are
recorded only when those CVS tests execute, such as level 0 or RVS before 1.3.
'''

import re

_STATUSES = ("pass", "fail", "na")

# Same indicators as scan_test_results. A matching line fails the cell even when
# the suite's own fail_regex_pattern does not, because pytest already failed the node.
_SCAN_FAIL_RE = re.compile(
    r"test FAIL |test ERROR |ABORT|Traceback|No such file|FATAL|"
    r"cannot allocate memory due to process memory policy",
    re.I,
)

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
# RVS can clip the module token in interleaved output ("Module name :babe"); only a known
# token is trusted over the action name.
_KNOWN_MODULES = frozenset(module for module in _GROUP_MODULES.values() if module)

_PRECISION_MATCH = ("bf16", "fp16", "fp32", "fp64", "fp8")
_PRECISION_ORDER = ("fp8", "fp16", "bf16", "fp32", "fp64")
_KERNELS = ("Read", "Write", "Copy", "Mul", "Add", "Triad", "Dot")
_KERNEL_NAMES = {name.lower(): name for name in _KERNELS}

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
# Duration lines are the completed sample. "(*)" lines are in-progress intervals.
_PEBB_DONE = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*pcie-bandwidth.*?\[GPU::\s*(?P<index>\d+)\s*-\s*(?P<gpu>\d+)\s*-"
    r".*?h2d::(?P<h2d>\S+)\s+d2h::(?P<d2h>\S+)\s+(?P<value>[\d.]+)\s+GBps\s+duration:",
    re.I,
)
_PBQT_DONE = re.compile(
    r"\[(?P<action>[^\]]+)\]\s*p2p-bandwidth.*?\[GPU::\s*(?P<src_index>\d+)\s*-\s*(?P<src>\d+)\s*-"
    r".*?\[GPU::\s*(?P<dst_index>\d+)\s*-\s*(?P<dst>\d+)\s*-"
    r".*?bidirectional:\s*(?P<bidi>\S+)\s+(?P<value>[\d.]+)\s+GBps\s+duration:",
    re.I,
)
_BABEL_ROW = re.compile(
    r"(?:^|\s)(?P<gpu>\d+)\s+(?P<kernel>Read|Write|Copy|Mul|Add|Triad|Dot)\s+"
    r"(?P<mbytes>[\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+",
    re.I,
)
_MEM_BW = re.compile(r"mem\s+Test\s+\d+\s*:.*?bandwidth\s*=\s*(?P<value>[\d.]+)\s*GB/s", re.I)


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

    A node fails when any caller-supplied pattern matches, or when the output
    contains a scan_test_results indicator. The caller patterns are the same
    ones the suite uses to call fail_test. The scan match is recorded for
    drill-down and does not call fail_test itself.
    '''
    text = output if isinstance(output, str) else ""
    items = []
    scan = _SCAN_FAIL_RE.search(text)
    if scan:
        items.append({"name": "scan", "status": "fail", "message": scan.group(0).strip()})
    for pattern in fail_patterns or []:
        if not pattern:
            continue
        try:
            matched = re.search(pattern, text, re.I)
        except re.error:
            matched = None
        if matched:
            line_start = text.rfind("\n", 0, matched.start()) + 1
            line_end = text.find("\n", matched.end())
            if line_end < 0:
                line_end = len(text)
            items.append(
                {
                    "name": str(pattern),
                    "status": "fail",
                    "message": text[line_start:line_end].strip() or matched.group(0),
                }
            )
    if items:
        return "fail", items
    return "pass", []


def build_node_record(status, items=None, items_summary="", metrics=None, series=None, heatmaps=None):
    '''Assemble one node record for an executed CVS test group.'''
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
    Merge one executed CVS test group's per-node records into the results dict.

    Idempotent per (group, node): a re-run of the same test overwrites.
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
        # One GPU is already the metric bar. A series is only the per-GPU spread.
        if len(rows) < 2:
            continue
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


def _keep_max(store, key, value):
    previous = store.get(key)
    if previous is None or value > previous:
        store[key] = value


def _take_pebb(line, samples):
    if "(*)" in line:
        return False
    match = _PEBB_DONE.search(line)
    if not match:
        return False
    label = f"GPU{int(match.group('index'))}"
    value = float(match.group("value"))
    # Each GPU is measured from more than one CPU socket; keep the faster socket.
    if match.group("h2d").lower() == "true":
        _keep_max(samples, ("h2d", label), value)
    if match.group("d2h").lower() == "true":
        _keep_max(samples, ("d2h", label), value)
    return True


def _take_pbqt(line, pairs):
    if "(*)" in line:
        return False
    match = _PBQT_DONE.search(line)
    if not match:
        return False
    src = f"GPU{int(match.group('src_index'))}"
    dst = f"GPU{int(match.group('dst_index'))}"
    pairs[(src, dst)] = float(match.group("value"))
    return True


def _pebb_performance(samples):
    metrics = []
    for direction in ("h2d", "d2h"):
        rows = [(label, value) for (kind, label), value in samples.items() if kind == direction]
        rows.sort(key=lambda item: _label_key(item[0]))
        for label, value in rows:
            metrics.append(_metric(f"{direction} {label}", value, "GB/s", "pebb", direction="higher"))
    # The per-GPU bars are the chart; a direction series would repeat them.
    return metrics, []


def _pbqt_performance(pairs):
    if not pairs:
        return [], []
    labels = sorted({gpu for pair in pairs for gpu in pair}, key=_label_key)
    values = []
    for src in labels:
        row = []
        for dst in labels:
            row.append(None if src == dst else pairs.get((src, dst)))
        values.append(row)
    heatmap = {
        "name": "xgmi",
        "unit": "GB/s",
        "group": "pbqt",
        "rows": labels,
        "cols": labels,
        "values": values,
        "direction": "higher",
        "row_label": "Src",
        "col_label": "Dst",
    }
    # The matrix already shows every link, including the slowest.
    return [], [heatmap]


def _in_babel(module, action, hint):
    if module == "babel" or (not module and hint == "babel"):
        return True
    if module in _KNOWN_MODULES:
        return False
    act = str(action or "").lower()
    return act.startswith("hbm") or "babel" in act


def _take_babel(line, labels, samples):
    match = _BABEL_ROW.search(line)
    if not match:
        return False
    kernel = _KERNEL_NAMES[match.group("kernel").lower()]
    label = _label_for(match.group("gpu"), labels)
    samples[(label, kernel)] = float(match.group("mbytes"))
    return True


def _take_mem(line, values):
    match = _MEM_BW.search(line)
    if not match:
        return False
    values.append(float(match.group("value")))
    return True


def _babel_performance(samples):
    '''The GPU × kernel heatmap is the chart. A slowest-GPU bar would repeat a column.'''
    if not samples:
        return [], []
    kernels = [name for name in _KERNELS if any(kernel == name for _label, kernel in samples)]
    labels = sorted({label for label, _kernel in samples}, key=_label_key)
    grid = [[samples.get((label, kernel)) for kernel in kernels] for label in labels]
    heatmap = {
        "name": "babel",
        "unit": "MB/s",
        "group": "babel",
        "rows": labels,
        "cols": kernels,
        "values": grid,
        "direction": "higher",
        "row_label": "GPU",
        "col_label": "Kernel",
    }
    return [], [heatmap]


def _mem_performance(values):
    return [_metric("bandwidth", value, "GB/s", "mem", direction="higher") for value in values]


def _iet_performance(peaks, statuses):
    metrics = []
    for label in sorted(peaks, key=_label_key):
        metrics.append(_metric(f"power {label}", peaks[label], "W", "iet", status=statuses.get(label, "")))
    # One bar per GPU already. A power series would plot the same peaks.
    return metrics, []


def parse_node_performance(text, module_hint=""):
    '''
    Pull numeric RVS measurements out of one node's stdout.

    Returns ``{"metrics", "series", "heatmaps"}``. Unknown or empty output
    yields empty lists. Interval GST samples without ``Target GFLOPS`` are
    ignored, as are PEBB/PBQT ``(*)`` samples. Babel and PBQT keep a heatmap
    (MiBytes/sec is the first Babel column) and skip a second summary bar.
    IET keeps the peak ``Power(W)`` per GPU; RVS does not print a temperature.
    '''
    body = text if isinstance(text, str) else ""
    labels = _device_labels(body)
    hint = _hint_module(module_hint)
    current_module = hint
    current_action = ""
    gst_samples = {}
    iet_peaks = {}
    iet_status = {}
    pebb_samples = {}
    pbqt_pairs = {}
    babel_samples = {}
    mem_values = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        module_match = _MODULE_NAME.search(line)
        if module_match:
            current_module = module_match.group("module").strip().lower()
            continue
        action_match = _ACTION_NAME.search(line)
        if action_match:
            current_action = action_match.group("action").strip()
            continue
        if _take_gst(line, labels, gst_samples):
            continue
        if _take_iet(line, labels, iet_peaks, iet_status):
            continue
        if _take_pebb(line, pebb_samples):
            continue
        if _take_pbqt(line, pbqt_pairs):
            continue
        if _in_babel(current_module, current_action, hint) and _take_babel(line, labels, babel_samples):
            continue
        _take_mem(line, mem_values)
    metrics, series = _gst_performance(gst_samples)
    iet_metrics, iet_series = _iet_performance(iet_peaks, iet_status)
    pebb_metrics, pebb_series = _pebb_performance(pebb_samples)
    pbqt_metrics, pbqt_heatmaps = _pbqt_performance(pbqt_pairs)
    babel_metrics, babel_heatmaps = _babel_performance(babel_samples)
    return {
        "metrics": metrics + iet_metrics + pebb_metrics + pbqt_metrics + babel_metrics + _mem_performance(mem_values),
        "series": series + iet_series + pebb_series,
        "heatmaps": pbqt_heatmaps + babel_heatmaps,
    }


def record_outputs(res_dict, group, out_dict, fail_patterns, meta=None):
    '''Classify each node's output and merge the executed CVS test group.'''
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
