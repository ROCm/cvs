'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``transferbench_res_dict`` consumed by the Run Deck
status_matrix builder from TransferBench stdout CVS already collects per node.

Bandwidth presets keep the measured GB/s in the cell drill-down. healthcheck
and a2asweep have no numeric gate, so their verdict uses the same failure
indicators as ``scan_test_results``. Chart fields never change that verdict:
missing or unrecognized output simply omits metrics, series, and heatmaps.

When the v1.67 tables are present, a2a emits one RTotal metric per GPU
(threshold ``gpu_to_gpu_a2a_rtotal``) and an ``RTotal`` series. p2p emits
``UniDir`` and ``BiDir`` GPU-to-GPU metrics plus a series of the four path
averages. The BiDir banner is ``Averages (During  BiDir)`` (two spaces).
``TransferBench vX.Y.Z`` from the banner is stored on ``_meta.version``.
healthcheck adds subtest items, per-GPU measured/criteria metrics, and an
``XGMI`` heatmap, but its node status stays on the scan indicators. a2asweep
keeps that scan verdict and adds a BlockSize/Unroll × SubExec heatmap plus
highest-bandwidth and best BlockSize / Unroll / NumSubExec metrics. scaling
adds one series per NumCUs endpoint and a Best-row metric per endpoint; only
GPU00 carries ``best_gpu0_bw``. schmoo adds one series per local and remote
column (separate charts, so the GB/s scales stay apart) and threshold metrics
for the gated 32 CU row.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version_label": "TransferBench", "suite": ...},
      "groups": {"<preset>": {"nodes": {"<label>": <node record>}}},
    }
'''

import re

_STATUSES = ("pass", "fail", "na")

# healthcheck and a2asweep do not publish a bandwidth row. Match the indicators
# scan_test_results already uses to fail those presets.
_SCAN_FAIL_RE = re.compile(
    r"test FAIL |test ERROR |ABORT|Traceback|No such file|FATAL|"
    r"cannot allocate memory due to process memory policy",
    re.I,
)
_A2A_RTOTAL_RE = re.compile(
    r"(?:│\s+)?RTotal\s+(?:│\s+)?"
    r"([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+"
    r"([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s*"
)
_SCALING_BEST_RE = re.compile(
    r"(?m)^[ \t]*Best\s+(?:[0-9\.]+\(\s*[0-9]+\)\s+){2}([0-9\.]+)",
)
_SCHMOO_32_RE = re.compile(
    r"(?m)^[ \t]*(?:[|│][ \t]*)*(?<![0-9])32(?![0-9])(?:[ \t]*[|│])?[ \t]+"
    r"([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+"
    r"([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)"
)

_SCHMOO_FIELDS = (
    ("local read", "32_cu_local_read"),
    ("local write", "32_cu_local_write"),
    ("local copy", "32_cu_local_copy"),
    ("remote read", "32_cu_rem_read"),
    ("remote write", "32_cu_rem_write"),
    ("remote copy", "32_cu_rem_copy"),
)
_P2P_PATHS = ("CPU->CPU", "CPU->GPU", "GPU->CPU", "GPU->GPU")
_VERSION_RE = re.compile(r"TransferBench\s+v([0-9]+(?:\.[0-9A-Za-z]+)*)")
_P2P_LINE_RE = re.compile(
    r"Averages\s+\(During\s+(UniDir|BiDir)\):\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.I,
)
_HEALTH_TEST_RE = re.compile(
    r"(?m)^[ \t]*Testing\s+(.+?)\s*\.{2,}\s*(PASS|FAIL)(?:\s*\((\d+)\s+test\(s\)\))?",
    re.I,
)
_HEALTH_GPU_RE = re.compile(
    r"GPU\s+(\d+)\s*:\s*Measured:\s*([0-9.]+)\s*GB/s\s*Criteria:\s*([0-9.]+)\s*GB/s",
    re.I,
)
_HEALTH_PAIR_RE = re.compile(
    r"GPU\s+(\d+)\s+to\s+GPU\s+(\d+)\s*:\s*([0-9.]+)\s*GB/s\s*Criteria:\s*([0-9.]+)",
    re.I,
)
_A2A_SWEEP_HEADER_RE = re.compile(r"BlkS\s+UnR\s+((?:SE\s*\d+\s*)+)", re.I)
_SCALING_HEADER_RE = re.compile(r"(?m)^[ \t]*NumCUs\s+((?:(?:CPU|GPU)\d+\s*)+)", re.I)
_SCHMOO_ROW_RE = re.compile(
    r"(?m)^[ \t]*(?:[|│][ \t]*)*(?<![0-9])(\d+)(?![0-9])(?:[ \t]*[|│])?[ \t]+"
    r"([0-9.]+)[ \t]+([0-9.]+)[ \t]+([0-9.]+)[ \t]+"
    r"([0-9.]+)[ \t]+([0-9.]+)[ \t]+([0-9.]+)"
)


def make_meta(cluster_dict, suite_name):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—"
    return {
        "cluster": cluster,
        "version": "—",
        "version_label": "TransferBench",
        "suite": suite_name or "transferbench_cvs",
        "generated_at": "",
    }


def _text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return str(output.get("output") or "")
    return str(output or "")


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _below(value, threshold):
    if threshold in (None, ""):
        return False
    left = _number(value)
    right = _number(threshold)
    if left is None or right is None:
        return False
    return left < right


def _bandwidth_metric(name, value, threshold, unit="GB/s"):
    numeric = _number(value)
    if numeric is None:
        return None
    metric = {
        "name": name,
        "value": numeric,
        "unit": unit,
        "direction": "higher",
        "status": "fail" if _below(numeric, threshold) else "pass",
    }
    limit = _number(threshold) if threshold not in (None, "") else None
    if limit is not None:
        metric["threshold"] = limit
    return metric


def _series(name, points, unit="GB/s", x_label="", y_label=""):
    kept = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        y_val = _number(point.get("y"))
        if y_val is None:
            continue
        kept.append({"x": point.get("x"), "y": y_val})
    if not kept:
        return None
    series = {"name": name, "unit": unit, "points": kept}
    if x_label:
        series["x_label"] = x_label
    if y_label:
        series["y_label"] = y_label
    return series


def _heatmap(name, rows, cols, values, unit="GB/s", threshold=None, direction="higher", row_label="", col_label=""):
    if not rows or not cols or len(values) != len(rows):
        return None
    if any(not isinstance(row, list) or len(row) != len(cols) for row in values):
        return None
    heat = {
        "name": name,
        "unit": unit,
        "rows": [str(row) for row in rows],
        "cols": [str(col) for col in cols],
        "values": values,
        "direction": direction,
    }
    limit = _number(threshold) if threshold not in (None, "") else None
    if limit is not None:
        heat["threshold"] = limit
    if row_label:
        heat["row_label"] = row_label
    if col_label:
        heat["col_label"] = col_label
    return heat


def _charts(metrics=None, series=None, heatmaps=None):
    extras = {}
    metrics = [item for item in (metrics or []) if item]
    series = [item for item in (series or []) if item]
    heatmaps = [item for item in (heatmaps or []) if item]
    if metrics:
        extras["metrics"] = metrics
    if series:
        extras["series"] = series
    if heatmaps:
        extras["heatmaps"] = heatmaps
    return extras


def _note_version(res_dict, version):
    if not version:
        return
    meta = res_dict.get("_meta")
    if not isinstance(meta, dict):
        return
    if meta.get("version") in (None, "", "—"):
        meta["version"] = version


def _metric_item(name, value, threshold):
    status = "fail" if _below(value, threshold) else "pass"
    if threshold in (None, ""):
        message = f"{value} GB/s"
    else:
        message = f"{value} GB/s (threshold {threshold})"
    return {"name": name, "status": status, "message": message}


def _missing_item(name, message):
    return {"name": name, "status": "fail", "message": message}


def build_node_record(status, items=None, items_summary="", metrics=None, series=None, heatmaps=None):
    '''Assemble one node record for a preset.'''
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
    record.update(_charts(metrics, series, heatmaps))
    return record


def record_group(res_dict, group, node_records, meta=None):
    '''Merge one preset's per-node records. A re-run of the same preset overwrites.'''
    if meta and not res_dict.get("_meta"):
        res_dict["_meta"] = dict(meta)
    groups = res_dict.setdefault("groups", {})
    group_entry = groups.setdefault(str(group), {"nodes": {}})
    group_entry["nodes"].update(node_records)
    return res_dict


def _unpack_classify(result):
    status, items, summary = result[0], result[1], result[2]
    extras = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
    return status, items, summary, extras


def _record_classified(res_dict, group, out_dict, classify, meta=None):
    node_records = {}
    version = None
    for node, output in (out_dict or {}).items():
        text = _text(output)
        found = _VERSION_RE.search(text)
        if found and not version:
            version = found.group(1)
        status, items, summary, extras = _unpack_classify(classify(text))
        node_records[str(node)] = build_node_record(
            status,
            items,
            summary,
            metrics=extras.get("metrics"),
            series=extras.get("series"),
            heatmaps=extras.get("heatmaps"),
        )
    recorded = record_group(res_dict, group, node_records, meta=meta)
    _note_version(recorded, version)
    return recorded


def _rollup(items, summary):
    status = "fail" if any(item["status"] == "fail" for item in items) else "pass"
    return status, items, summary


def record_a2a(res_dict, out_dict, exp_dict, meta=None):
    '''Record GPU RTotal bandwidths. Any GPU under the threshold fails the node.'''
    threshold = (exp_dict or {}).get("gpu_to_gpu_a2a_rtotal")

    def classify(text):
        match = _A2A_RTOTAL_RE.search(text)
        if not match:
            message = "RTotal row not found"
            return "fail", [_missing_item("RTotal", message)], message
        items = [_metric_item(f"GPU{idx}", raw, threshold) for idx, raw in enumerate(match.groups())]
        worst = min(float(raw) for raw in match.groups())
        # The per-GPU bars are the chart. A second RTotal series would plot the same eight numbers.
        metrics = [_bandwidth_metric(f"GPU{idx:02d}", raw, threshold) for idx, raw in enumerate(match.groups())]
        status, items, summary = _rollup(items, f"min RTotal {worst} GB/s")
        return status, items, summary, _charts(metrics)

    return _record_classified(res_dict, "a2a", out_dict, classify, meta)


def record_p2p(res_dict, out_dict, exp_dict, meta=None):
    '''Record average unidirectional and bidirectional GPU-to-GPU bandwidth.'''
    expected = exp_dict or {}
    gates = {
        "UniDir": expected.get("avg_gpu_to_gpu_p2p_unidir_bw"),
        "BiDir": expected.get("avg_gpu_to_gpu_p2p_bidir_bw"),
    }

    def classify(text):
        lines = {}
        for match in _P2P_LINE_RE.finditer(text):
            key = "UniDir" if match.group(1).lower() == "unidir" else "BiDir"
            if key not in lines:
                lines[key] = match.groups()[1:]
        items = []
        series = []
        for name in ("UniDir", "BiDir"):
            values = lines.get(name)
            if not values:
                items.append(_missing_item(name, f"{name} averages not found"))
                continue
            # The cell already shows the gated GPU->GPU number. The chart is the four-path breakdown.
            items.append(_metric_item(name, values[-1], gates[name]))
            points = [{"x": label, "y": raw} for label, raw in zip(_P2P_PATHS, values)]
            series.append(_series(name, points, x_label="path", y_label="GB/s"))
        if "UniDir" in lines and "BiDir" in lines:
            summary = f"UniDir {lines['UniDir'][-1]} / BiDir {lines['BiDir'][-1]} GB/s"
        else:
            summary = "p2p averages not found"
        status, items, summary = _rollup(items, summary)
        return status, items, summary, _charts(series=series)

    return _record_classified(res_dict, "p2p", out_dict, classify, meta)


def record_scaling(res_dict, out_dict, exp_dict, meta=None):
    '''Record the Best-row GPU00 bandwidth.

    The chart is the NumCUs curve per endpoint. The pass/fail gate stays the
    historical GPU00 capture, which skips two CPU columns, matching ``parse_tb_scaling_bw``.
    '''
    threshold = (exp_dict or {}).get("best_gpu0_bw")

    def classify(text):
        match = _SCALING_BEST_RE.search(text)
        gpu00 = match.group(1) if match else None
        extras = _scaling_visuals(text)
        if not match:
            message = "Best row GPU00 bandwidth not found"
            return "fail", [_missing_item("GPU00", message)], message, extras
        item = _metric_item("GPU00", gpu00, threshold)
        status, items, summary = _rollup([item], f"GPU00 best {gpu00} GB/s")
        return status, items, summary, extras

    return _record_classified(res_dict, "scaling", out_dict, classify, meta)


def _scaling_endpoints(text):
    header = _SCALING_HEADER_RE.search(text)
    if not header:
        return None, []
    endpoints = [f"{kind.upper()}{num}" for kind, num in re.findall(r"(CPU|GPU)(\d+)", header.group(1), re.I)]
    return header, endpoints


def _scaling_series(text, header, endpoints):
    if not header or not endpoints:
        return []
    rest = text[header.end() :]
    best_at = re.search(r"(?m)^[ \t]*Best\b", rest)
    body = rest[: best_at.start()] if best_at else rest
    points = {name: [] for name in endpoints}
    width = len(endpoints) + 1
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != width or not re.fullmatch(r"\d+", parts[0]):
            continue
        values = [_number(token) for token in parts[1:]]
        if any(item is None for item in values):
            continue
        for name, value in zip(endpoints, values):
            points[name].append({"x": int(parts[0]), "y": value})
    return [_series(name, points[name], x_label="CUs", y_label="GB/s") for name in endpoints]


def _scaling_visuals(text):
    # Best-row bars repeat the last point of each curve, and the GPU00 gate is already the cell summary.
    header, endpoints = _scaling_endpoints(text)
    return _charts(series=_scaling_series(text, header, endpoints))


def record_schmoo(res_dict, out_dict, exp_dict, meta=None):
    '''Record the 32 CU local and remote read/write/copy bandwidths.

    Each column is its own series so local and remote GB/s stay on separate
    scales. Threshold metrics are the gated 32 CU row.
    '''
    expected = exp_dict or {}

    def classify(text):
        extras = _schmoo_visuals(text, expected)
        match = _SCHMOO_32_RE.search(text)
        if not match:
            message = "32 CU row not found"
            return "fail", [_missing_item("32 CU", message)], message, extras
        items = [
            _metric_item(label, match.group(idx + 1), expected.get(key))
            for idx, (label, key) in enumerate(_SCHMOO_FIELDS)
        ]
        status, items, summary = _rollup(items, f"32 CU local copy {match.group(3)} GB/s")
        return status, items, summary, extras

    return _record_classified(res_dict, "schmoo", out_dict, classify, meta)


def _schmoo_visuals(text, expected):
    rows = []
    for match in _SCHMOO_ROW_RE.finditer(text):
        values = [_number(match.group(idx)) for idx in range(2, 8)]
        if any(item is None for item in values):
            continue
        rows.append((int(match.group(1)), values))
    series = []
    for idx, (label, _key) in enumerate(_SCHMOO_FIELDS):
        points = [{"x": cu, "y": values[idx]} for cu, values in rows]
        # One CU row is a bar, not a curve. The 32 CU bars carry the thresholds.
        if len(points) < 2:
            continue
        series.append(_series(label, points, x_label="CUs", y_label="GB/s"))
    metrics = []
    gated = _SCHMOO_32_RE.search(text)
    if gated:
        for idx, (label, key) in enumerate(_SCHMOO_FIELDS):
            metrics.append(_bandwidth_metric(label, gated.group(idx + 1), expected.get(key)))
    return _charts(metrics, series)


def _a2asweep_heatmap(text):
    header = _A2A_SWEEP_HEADER_RE.search(text)
    if not header:
        return None
    cols = [f"SE {num}" for num in re.findall(r"SE\s*(\d+)", header.group(1), re.I)]
    if not cols:
        return None
    rest = text[header.end() :]
    end = re.search(r"Highest|={5,}", rest)
    body = rest[: end.start()] if end else rest
    rows = []
    values = []
    width = len(cols) + 2
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != width or not re.fullmatch(r"\d+", parts[0]) or not re.fullmatch(r"\d+", parts[1]):
            continue
        nums = [_number(token) for token in parts[2:]]
        if any(item is None for item in nums):
            continue
        rows.append(f"{parts[0]}/{parts[1]}")
        values.append(nums)
    return _heatmap("a2a sweep", rows, cols, values, row_label="BlkS/UnR", col_label="SubExec")


def _a2asweep_visuals(text):
    # The heatmap is the sweep. Its brightest cell is the reported peak, and BlockSize/Unroll/NumSubExec
    # are the winning config rather than a result to chart.
    heat = _a2asweep_heatmap(text)
    return _charts(heatmaps=[heat] if heat else None)


def _xgmi_heatmap(pairs):
    if not pairs:
        return None
    indexes = sorted({src for src, _dst, _bw, _limit in pairs} | {dst for _src, dst, _bw, _limit in pairs})
    labels = [f"GPU{idx:02d}" for idx in indexes]
    position = {idx: pos for pos, idx in enumerate(indexes)}
    values = [[None for _label in labels] for _label in labels]
    criteria = []
    for src, dst, bandwidth, limit in pairs:
        values[position[src]][position[dst]] = bandwidth
        if limit is not None:
            criteria.append(limit)
    threshold = criteria[0] if criteria and all(item == criteria[0] for item in criteria) else None
    return _heatmap("XGMI", labels, labels, values, threshold=threshold, row_label="Src", col_label="Dst")


def _healthcheck_visuals(text):
    items = []
    for match in _HEALTH_TEST_RE.finditer(text):
        verdict = match.group(2)
        count = match.group(3)
        status = "fail" if verdict.lower() == "fail" else "pass"
        message = f"{verdict.upper()} ({count} test(s))" if count else verdict.upper()
        items.append({"name": match.group(1).strip(), "status": status, "message": message})
    metrics = []
    for match in _HEALTH_GPU_RE.finditer(text):
        metrics.append(_bandwidth_metric(f"GPU{int(match.group(1)):02d}", match.group(2), match.group(3)))
    pairs = []
    for match in _HEALTH_PAIR_RE.finditer(text):
        bandwidth = _number(match.group(3))
        if bandwidth is None:
            continue
        pairs.append((int(match.group(1)), int(match.group(2)), bandwidth, _number(match.group(4))))
    heat = _xgmi_heatmap(pairs)
    return items, _charts(metrics, heatmaps=[heat] if heat else None)


def record_completion(res_dict, group, out_dict, meta=None):
    '''Record a preset that is pass/fail from scan indicators, with no bandwidth row.'''

    def classify(text):
        match = _SCAN_FAIL_RE.search(text)
        subtests, extras = ([], {})
        if str(group) == "healthcheck":
            subtests, extras = _healthcheck_visuals(text)
        elif str(group) == "a2asweep":
            extras = _a2asweep_visuals(text)
        if match:
            snippet = match.group(0).strip()
            return "fail", list(subtests) + [_missing_item(str(group), snippet)], snippet, extras
        if subtests:
            # "Testing ... FAIL" is not a scan_test_results indicator, so it stays
            # in the drill-down and does not change the node verdict.
            failed = sum(1 for item in subtests if item["status"] == "fail")
            return "pass", subtests, f"{len(subtests) - failed} pass, {failed} fail", extras
        return "pass", [], "completed", extras

    return _record_classified(res_dict, group, out_dict, classify, meta)
