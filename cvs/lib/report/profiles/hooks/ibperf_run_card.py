'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Run-card fields for the ibperf suite.
'''

from cvs.lib.ibperf_reporting import OUTLIER_RULE, node_names, size_text
from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def _display(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "on" if value else "off"
    if value in (None, ""):
        return "—"
    return str(value)


def _nodes(count):
    if not isinstance(count, int):
        return _display(count)
    pairs = count // 2
    return f"{count} ({pairs} server/client pair{'' if pairs == 1 else 's'})"


def _sizes(sizes):
    try:
        sizes = sorted(int(size) for size in sizes or [])
    except (TypeError, ValueError):
        return _display(sizes)
    return f"{size_text(sizes[0])} – {size_text(sizes[-1])} · {len(sizes)} sizes" if sizes else "—"


def _thresholds(expected):
    parts = []
    for test, by_size in (expected or {}).items():
        if str(test).endswith("_bw") and isinstance(by_size, dict):
            sizes = sorted(int(size) for size in by_size if str(size).isdigit())
            parts.append(f"{test} at {', '.join(size_text(size) for size in sizes)}")
    return "; ".join(parts) or "none configured"


def ibperf_run_card_display(variant, provenance):
    config = variant if isinstance(variant, dict) else {}
    duration = config.get("duration")
    rows = [
        ("Nodes", _nodes(config.get("node_count")), False),
        (
            "Node names",
            _display([f"{name} = {host}" for host, name in node_names(config.get("hosts")).items()] or None),
            False,
        ),
        ("NICs per node", _display(config.get("nic_count")), False),
        ("BW tests", _display(config.get("ib_bw_test_list")), False),
        ("Latency tests", _display(config.get("ib_lat_test_list")), False),
        ("Message sizes", _sizes(config.get("msg_size_list")), False),
        ("QP counts", _display(config.get("qp_count_list")), False),
        ("Duration per test", f"{duration} s" if duration not in (None, "") else "—", False),
        ("dmabuf", _display(config.get("dmabuf")), False),
        ("BW thresholds", _thresholds(config.get("expected_results")), False),
        ("Outlier rule", OUTLIER_RULE, False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["ibperf_run_card_display"]
