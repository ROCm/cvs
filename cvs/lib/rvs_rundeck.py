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
'''

import re

_STATUSES = ("pass", "fail", "na")


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


def build_node_record(status, items=None, items_summary=""):
    '''Assemble one node record for a module.'''
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


def record_outputs(res_dict, group, out_dict, fail_patterns, meta=None):
    '''Classify each node's output and merge the module into ``res_dict``.'''
    node_records = {}
    for node, output in (out_dict or {}).items():
        status, items = classify_output(output, fail_patterns)
        if status == "fail":
            summary = f"{len(items)} failure pattern(s)"
        else:
            summary = "passed"
        node_records[str(node)] = build_node_record(status, items, summary)
    return record_group(res_dict, group, node_records, meta=meta)
