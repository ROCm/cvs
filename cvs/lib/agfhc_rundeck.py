'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``agfhc_res_dict`` consumed by the Run Deck status_matrix
builder from AGFHC stdout CVS already collects per node.

The cell verdict matches ``scan_agfc_results``: missing ``code AGFHC_SUCCESS``
fails the node, and so does any ``FAIL``, ``ERROR``, or ``ABORT`` line. Parsed
test rows are drill-down only. Recipe-info contents (a test name, a title, and
an approximate duration) are not results, so they stay out of the cell.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "version_label": "AGFHC version", "suite": ...},
      "groups": {"<executed recipe>": {"nodes": {"<label>": <node record>}}},
    }
'''

import re

_STATUSES = ("pass", "fail", "na")
_RANK = {"na": 0, "pass": 1, "fail": 2}

# Same indicators as scan_agfc_results. AGFHC_FAILURE matches FAIL inside the token.
_SUCCESS_RE = re.compile(r"code\s+AGFHC_SUCCESS", re.I)
_SCAN_FAIL_RE = re.compile(r"FAIL|ERROR|ABORT", re.I)
_VERSION_RE = re.compile(r"(?is)agfhc version:\s*([0-9]+(?:\.[0-9A-Za-z]+)*)")
_COUNTS_RE = re.compile(r"Tests:\s*(\d+)\s+Total,\s*(\d+)\s+Executed,\s*(\d+)\s+Skipped", re.I)

_STATUS_WORD = r"passed|failed|skipped|queued|pass|fail|success|failure"
_TEST_NAME = r"[A-Za-z][A-Za-z0-9_]{0,63}"
# Recipe-info prints "pcie_link_status PCIe Link Status 1 iteration 0:00:08".
# A result row is a name plus a status word, not that contents table.
_ROW_RES = (
    re.compile(rf"(?i)^[ \t]*\|[ \t]*({_TEST_NAME})[ \t]*\|[ \t]*({_STATUS_WORD})\b"),
    re.compile(rf"(?i)^[ \t]*({_TEST_NAME})\s*\.{{2,}}\s*({_STATUS_WORD})\b"),
    re.compile(rf"(?i)^[ \t]*({_TEST_NAME})\s*:\s*({_STATUS_WORD})\b"),
    re.compile(rf"(?i)^[ \t]*({_TEST_NAME})[ \t]{{2,}}({_STATUS_WORD})\b"),
    re.compile(rf"(?i)^[ \t]*({_TEST_NAME})[ \t]+({_STATUS_WORD})[ \t]*$"),
)
_HEADER_NAMES = frozenset(
    {
        "test",
        "name",
        "title",
        "mode",
        "summary",
        "contents",
        "total",
        "tests",
        "status",
        "result",
        "iteration",
        "program",
        "log",
        "path",
        "package",
    }
)
_PASS_WORDS = frozenset({"passed", "pass", "success"})
_FAIL_WORDS = frozenset({"failed", "fail", "failure"})


def make_meta(cluster_dict, suite_name):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—"
    return {
        "cluster": cluster,
        "version": "—",
        "version_label": "AGFHC version",
        "suite": suite_name or "agfhc_cvs",
        "generated_at": "",
    }


def _text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return str(output.get("output") or "")
    return str(output or "")


def _row_status(word):
    token = str(word or "").strip().lower()
    if token in _FAIL_WORDS:
        return "fail"
    if token in _PASS_WORDS:
        return "pass"
    return "na"


def _take_row(rows, name, status, message):
    key = str(name or "").strip()
    if not key or key.lower() in _HEADER_NAMES:
        return
    current = rows.get(key)
    if current is not None and _RANK[status] < _RANK[current["status"]]:
        return
    rows[key] = {"name": key, "status": status, "message": message}


def _test_rows(text):
    rows = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        for pattern in _ROW_RES:
            match = pattern.search(line)
            if match:
                _take_row(rows, match.group(1), _row_status(match.group(2)), line)
                break
    return list(rows.values())


def _first_scan_line(text):
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line and _SCAN_FAIL_RE.search(line):
            return line
    if _SCAN_FAIL_RE.search(text or ""):
        return _SCAN_FAIL_RE.search(text).group(0).strip()
    return ""


def _counts_summary(text):
    match = _COUNTS_RE.search(text or "")
    if not match:
        return ""
    return f"{match.group(1)} total, {match.group(2)} executed, {match.group(3)} skipped"


def classify_output(output):
    '''
    Return (status, items, summary) for one node's AGFHC output.

    A node fails when ``code AGFHC_SUCCESS`` is absent or a scan line matches,
    which is the same gate ``scan_agfc_results`` uses to call fail_test.
    '''
    text = _text(output)
    items = _test_rows(text)
    if not _SUCCESS_RE.search(text):
        items.append(
            {
                "name": "AGFHC_SUCCESS",
                "status": "fail",
                "message": "return code AGFHC_SUCCESS not seen",
            }
        )
    scan_line = _first_scan_line(text)
    if scan_line and not any(item["status"] == "fail" and item["message"] == scan_line for item in items):
        items.append({"name": "scan", "status": "fail", "message": scan_line})
    status = "fail" if any(item["status"] == "fail" for item in items) else "pass"
    tests = [item for item in items if item["name"] not in ("scan", "AGFHC_SUCCESS")]
    if tests:
        passed = sum(1 for item in tests if item["status"] == "pass")
        failed = sum(1 for item in tests if item["status"] == "fail")
        skipped = sum(1 for item in tests if item["status"] == "na")
        summary = f"{passed} pass, {failed} fail, {skipped} skipped"
    elif any(item["name"] == "AGFHC_SUCCESS" for item in items) and not scan_line:
        summary = "AGFHC_SUCCESS not seen"
    elif status == "fail":
        summary = f"{sum(1 for item in items if item['status'] == 'fail')} failure pattern(s)"
    else:
        summary = _counts_summary(text) or "passed"
    return status, items, summary


def build_node_record(status, items=None, items_summary=""):
    '''Assemble one node record for an executed AGFHC recipe.'''
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


def _note_version(res_dict, version):
    if not version:
        return
    meta = res_dict.get("_meta")
    if isinstance(meta, dict) and meta.get("version") in (None, "", "—"):
        meta["version"] = version


def record_group(res_dict, group, node_records, meta=None):
    '''
    Merge one executed recipe's per-node records into the results dict.

    Idempotent per (group, node): a re-run of the same recipe overwrites.
    A later real version replaces a placeholder left by an earlier recipe.
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


def _version(text):
    match = _VERSION_RE.search(text or "")
    if not match:
        return ""
    return match.group(1)


def record_outputs(res_dict, group, out_dict, meta=None):
    '''Classify each node's output and merge the executed recipe group.'''
    node_records = {}
    found_version = ""
    for node, output in (out_dict or {}).items():
        text = _text(output)
        if not found_version:
            found_version = _version(text)
        status, items, summary = classify_output(text)
        node_records[str(node)] = build_node_record(status, items, summary)
    recorded = record_group(res_dict, group, node_records, meta=meta)
    _note_version(recorded, found_version)
    return recorded


def record_version_check(res_dict, out_dict, meta=None):
    '''
    Record the AGFHC ``-v`` probe used by CSP qualification.

    Pass when each node prints ``agfhc version:``. This path does not require
    ``AGFHC_SUCCESS``, which recipes print but ``-v`` does not.
    '''
    node_records = {}
    found_version = ""
    for node, output in (out_dict or {}).items():
        text = _text(output)
        if not found_version:
            found_version = _version(text)
        if re.search(r"agfhc version:", text, re.I):
            node_records[str(node)] = build_node_record("pass", [], "version printed")
        else:
            node_records[str(node)] = build_node_record(
                "fail",
                [{"name": "version", "status": "fail", "message": "agfhc version: not seen"}],
                "version missing",
            )
    recorded = record_group(res_dict, "version_check", node_records, meta=meta)
    _note_version(recorded, found_version)
    return recorded
