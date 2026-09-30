'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``transferbench_res_dict`` consumed by the Run Deck
status_matrix builder from TransferBench stdout CVS already collects per node.

Bandwidth presets keep the measured GB/s in the cell drill-down. healthcheck
and a2asweep have no numeric gate, so their verdict uses the same failure
indicators as ``scan_test_results``.

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
_P2P_UNIDIR_RE = re.compile(
    r"Averages\s+\(During\s+UniDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)",
    re.I,
)
_P2P_BIDIR_RE = re.compile(
    r"Averages\s+\(During\s+BiDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)",
    re.I,
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


def _below(value, threshold):
    if threshold in (None, ""):
        return False
    try:
        return float(value) < float(threshold)
    except (TypeError, ValueError):
        return False


def _metric_item(name, value, threshold):
    status = "fail" if _below(value, threshold) else "pass"
    if threshold in (None, ""):
        message = f"{value} GB/s"
    else:
        message = f"{value} GB/s (threshold {threshold})"
    return {"name": name, "status": status, "message": message}


def _missing_item(name, message):
    return {"name": name, "status": "fail", "message": message}


def build_node_record(status, items=None, items_summary=""):
    '''Assemble one node record for a preset.'''
    normalized = str(status or "na").lower()
    if normalized not in _STATUSES:
        normalized = "na"
    return {
        "status": normalized,
        "items_summary": items_summary or "",
        "items": items or [],
        "errors_json_href": "",
        "log_tarball_href": "",
    }


def record_group(res_dict, group, node_records, meta=None):
    '''Merge one preset's per-node records. A re-run of the same preset overwrites.'''
    if meta and not res_dict.get("_meta"):
        res_dict["_meta"] = dict(meta)
    groups = res_dict.setdefault("groups", {})
    group_entry = groups.setdefault(str(group), {"nodes": {}})
    group_entry["nodes"].update(node_records)
    return res_dict


def _record_classified(res_dict, group, out_dict, classify, meta=None):
    node_records = {}
    for node, output in (out_dict or {}).items():
        status, items, summary = classify(_text(output))
        node_records[str(node)] = build_node_record(status, items, summary)
    return record_group(res_dict, group, node_records, meta=meta)


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
        return _rollup(items, f"min RTotal {worst} GB/s")

    return _record_classified(res_dict, "a2a", out_dict, classify, meta)


def record_p2p(res_dict, out_dict, exp_dict, meta=None):
    '''Record average unidirectional and bidirectional GPU-to-GPU bandwidth.'''
    expected = exp_dict or {}

    def classify(text):
        items = []
        uni = _P2P_UNIDIR_RE.search(text)
        bi = _P2P_BIDIR_RE.search(text)
        if uni:
            items.append(_metric_item("UniDir", uni.group(1), expected.get("avg_gpu_to_gpu_p2p_unidir_bw")))
        else:
            items.append(_missing_item("UniDir", "UniDir averages not found"))
        if bi:
            items.append(_metric_item("BiDir", bi.group(1), expected.get("avg_gpu_to_gpu_p2p_bidir_bw")))
        else:
            items.append(_missing_item("BiDir", "BiDir averages not found"))
        if uni and bi:
            summary = f"UniDir {uni.group(1)} / BiDir {bi.group(1)} GB/s"
        else:
            summary = "p2p averages not found"
        return _rollup(items, summary)

    return _record_classified(res_dict, "p2p", out_dict, classify, meta)


def record_scaling(res_dict, out_dict, exp_dict, meta=None):
    '''Record the Best-row GPU00 bandwidth.'''
    threshold = (exp_dict or {}).get("best_gpu0_bw")

    def classify(text):
        match = _SCALING_BEST_RE.search(text)
        if not match:
            message = "Best row GPU00 bandwidth not found"
            return "fail", [_missing_item("GPU00", message)], message
        item = _metric_item("GPU00", match.group(1), threshold)
        return _rollup([item], f"GPU00 best {match.group(1)} GB/s")

    return _record_classified(res_dict, "scaling", out_dict, classify, meta)


def record_schmoo(res_dict, out_dict, exp_dict, meta=None):
    '''Record the 32 CU local and remote read/write/copy bandwidths.'''
    expected = exp_dict or {}

    def classify(text):
        match = _SCHMOO_32_RE.search(text)
        if not match:
            message = "32 CU row not found"
            return "fail", [_missing_item("32 CU", message)], message
        items = [
            _metric_item(label, match.group(idx + 1), expected.get(key))
            for idx, (label, key) in enumerate(_SCHMOO_FIELDS)
        ]
        return _rollup(items, f"32 CU local copy {match.group(3)} GB/s")

    return _record_classified(res_dict, "schmoo", out_dict, classify, meta)


def record_completion(res_dict, group, out_dict, meta=None):
    '''Record a preset that is pass/fail from scan indicators, with no bandwidth row.'''

    def classify(text):
        match = _SCAN_FAIL_RE.search(text)
        if not match:
            return "pass", [], "completed"
        snippet = match.group(0).strip()
        return "fail", [_missing_item(str(group), snippet)], snippet

    return _record_classified(res_dict, group, out_dict, classify, meta)
