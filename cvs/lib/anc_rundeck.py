'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``anc_res_dict`` consumed by the Run Deck ``status_matrix``
dataset builder from the artifacts ANC already writes and CVS already collects
per node (console.log for the per-item summary, errors.json for failed items).

This is CVS-side reporting glue only: the on-node ``anc.py`` tool is untouched.
The dict is accumulated across every ``test_<group>`` in a suite (one group per
pytest test) and bound into the Run Deck session store at module teardown.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "suite": ..., "generated_at": ...},
      "groups": {"<group>": {"nodes": {"<label>": <node record>}}},
    }
'''

import json
import os

from cvs.lib import globals

log = globals.log


def make_meta(cluster_dict, config_dict, suite_name, timestamp):
    '''Assemble the _meta block for the deck run card.'''
    anc = config_dict.get("anc", {}) if isinstance(config_dict, dict) else {}
    return {
        "cluster": (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—",
        "version": anc.get("anc_version") or "—",
        "suite": suite_name or "anc",
        "generated_at": timestamp or "",
    }


def _parse_items_summary(console_path):
    '''
    Return ANC's verbatim "Items: N Total | X PASSED, Y FAILED" roll-up line from
    a collected console.log (used as the cell's pass/fail label), or "".
    Best-effort: a missing/unreadable file yields "".
    '''
    if not console_path or not os.path.isfile(console_path):
        return ""
    try:
        with open(console_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        log.warning("ANC Run Deck: could not read %s: %s", console_path, exc)
        return ""
    for raw in text.splitlines():
        if raw.strip().startswith("Items:"):
            return raw.strip()
    return ""


def _parse_error_items(errors_json_path):
    '''
    Extract per-item failure records from a node's errors.json, best-effort.

    ANC's errors.json (schema 0.2) is::

        {"errors": {"<item_id>-<name>": {"status": "FAILED", "name": ...,
                    "rc_enum": ..., "rc_desc": ..., "summary": ...}, ...}, ...}

    i.e. ``errors`` is a DICT keyed by item id whose values are failure records.
    A list-of-records form is also accepted for robustness. The display name is
    the record's ``name`` (falling back to the map key), and the message is built
    from ``rc_enum`` + ``summary`` (or ``rc_desc``) so the deck cell shows what
    actually failed. The file is empty/absent on a clean run.
    '''
    if not errors_json_path or not os.path.isfile(errors_json_path):
        return []
    try:
        with open(errors_json_path, encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("ANC Run Deck: could not parse %s: %s", errors_json_path, exc)
        return []

    errors = data.get("errors") if isinstance(data, dict) else data
    if isinstance(errors, dict):
        pairs = list(errors.items())
    elif isinstance(errors, list):
        pairs = [(None, rec) for rec in errors]
    else:
        pairs = []

    items = []
    for key, rec in pairs:
        if not isinstance(rec, dict):
            items.append({"name": str(key if key is not None else rec), "status": "fail", "message": ""})
            continue
        name = rec.get("name") or rec.get("test") or rec.get("item") or key or "error"
        items.append({"name": str(name), "status": "fail", "message": _error_message(rec)})
    return items


def _error_message(rec):
    '''Build a concise failure message from an ANC errors.json record.'''
    enum = rec.get("rc_enum") or ""
    detail = rec.get("summary") or rec.get("rc_desc") or rec.get("message") or rec.get("error") or ""
    parts = [str(p) for p in (enum, detail) if p]
    return ": ".join(parts) if parts else ""


def build_node_record(*, status, console_path, errors_json_path, errors_json_href, log_tarball_href):
    '''
    Assemble one node record for a group from the collected artifacts.

    ``status`` is "pass" | "fail" | "na". ``items_summary`` carries ANC's own
    "Items: N Total | X PASSED, Y FAILED" roll-up (the cell label uses it for the
    pass/fail count). ``items`` holds ONLY the real per-item failures read from
    errors.json -- ANC's console.log item names are not machine-stable, so we
    never fabricate passed/failed rows; a passing node simply has no item rows to
    expand.
    '''
    summary_line = _parse_items_summary(console_path)
    items = _parse_error_items(errors_json_path) if status == "fail" else []
    return {
        "status": status,
        "items_summary": summary_line,
        "items": items,
        "errors_json_href": errors_json_href or "",
        "log_tarball_href": log_tarball_href or "",
    }


def record_group(anc_res_dict, group, node_records, meta=None):
    '''
    Merge one group's per-node records into the accumulating anc_res_dict.

    ``node_records`` maps node label -> node record (from build_node_record).
    Idempotent per (group, node): a re-run of the same group overwrites.
    '''
    if meta and not anc_res_dict.get("_meta"):
        anc_res_dict["_meta"] = meta
    groups = anc_res_dict.setdefault("groups", {})
    group_entry = groups.setdefault(str(group), {"nodes": {}})
    group_entry["nodes"].update(node_records)
    return anc_res_dict
