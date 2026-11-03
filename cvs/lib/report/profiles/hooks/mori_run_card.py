'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Run-card fields for the Mori benchmark suite.
'''

from cvs.lib.mori_reporting import node_names
from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def _display(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if value in (None, ""):
        return "—"
    return str(value)


def mori_run_card_display(variant, provenance):
    config = variant if isinstance(variant, dict) else {}
    rows = [
        ("GPU", _display(config.get("gpu_name")), False),
        ("Nodes", _display(config.get("node_count")), False),
        (
            "Node names",
            _display([f"{name} = {host}" for host, name in node_names(config.get("hosts")).items()] or None),
            False,
        ),
        ("Mori devices", _display(config.get("mori_device_list")), False),
        ("Container image", _display(config.get("container_image")), False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["mori_run_card_display"]
