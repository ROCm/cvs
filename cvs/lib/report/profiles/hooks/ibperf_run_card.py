'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Run-card fields for the ibperf suite.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def _display(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    if value in (None, ""):
        return "—"
    return str(value)


def ibperf_run_card_display(variant, provenance):
    config = variant if isinstance(variant, dict) else {}
    duration = config.get("duration")
    rows = [
        ("Nodes", _display(config.get("node_count")), False),
        ("NICs per node", _display(config.get("nic_count")), False),
        ("Message sizes (bytes)", _display(config.get("msg_size_list")), False),
        ("QP counts", _display(config.get("qp_count_list")), False),
        ("Duration per test", f"{duration} s" if duration not in (None, "") else "—", False),
        ("dmabuf", _display(config.get("dmabuf")), False),
        ("Orchestrator", _display(config.get("orchestrator")), False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["ibperf_run_card_display"]
