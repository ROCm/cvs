'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Dataset builder for host configuration inventory and cross-node drift.
'''

import re

from cvs.lib.platform.host_inventory import FACT_LABELS
from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _profile_config(profile):
    return profile.get("host_inventory") or {}


def _node_names(results):
    nodes = set((results.get("nodes") or {}).keys())
    nodes.update((results.get("firmware") or {}).keys())
    return sorted(str(node) for node in nodes)


def _fact_specs(profile):
    configured = _profile_config(profile).get("inventory_columns") or [
        "os_release",
        "kernel",
        "bios",
        "rocm",
        "gpu_count",
        "online_memory",
    ]
    specs = []
    for entry in configured:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            specs.append((str(entry[0]), str(entry[1])))
        else:
            key = str(entry)
            specs.append((key, FACT_LABELS.get(key, key.replace("_", " ").title())))
    return specs


def _inventory_table(results, profile, nodes):
    specs = _fact_specs(profile)
    headers = ["Node"] + [label for _key, label in specs]
    rows = []
    node_facts = results.get("nodes") or {}
    for node in nodes:
        facts = node_facts.get(node) or {}
        rows.append([node] + [facts.get(key, "\u2014") for key, _label in specs])
    return {"headers": headers, "rows": rows}


def _flatten_value(rows, key, value, node):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _flatten_value(rows, f"{key}.{child_key}", child_value, node)
        return
    rows.setdefault(key, {})[node] = value


def _flatten_node_facts(results):
    rows = {}
    for node, facts in (results.get("nodes") or {}).items():
        if not isinstance(facts, dict):
            continue
        for key, value in facts.items():
            _flatten_value(rows, str(key), value, str(node))

    for node, gpu_entries in (results.get("firmware") or {}).items():
        if not isinstance(gpu_entries, dict):
            continue
        for gpu, firmware in gpu_entries.items():
            if not isinstance(firmware, dict):
                continue
            for firmware_id, version in firmware.items():
                rows.setdefault(f"firmware.{gpu}.{firmware_id}", {})[str(node)] = version
    return rows


def _natural_key(label):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(label))]


def _fact_label(key):
    parts = key.split(".")
    if parts[0] == "firmware" and len(parts) >= 3:
        return f"GPU {parts[1]} firmware {'.'.join(parts[2:])}"
    if parts[0] in ("gpu_pcie", "nic_pcie") and len(parts) >= 2:
        return f"{FACT_LABELS[parts[0]]} card {'.'.join(parts[1:])}"
    return FACT_LABELS.get(key, key.replace("_", " ").title())


def _drift_matrix(results, nodes):
    flattened = _flatten_node_facts(results)
    rows = []
    for key in sorted(flattened, key=lambda item: (_natural_key(_fact_label(item)), item)):
        by_node = flattened[key]
        values = [by_node.get(node, "\u2014") for node in nodes]
        mismatch = len({str(value) for value in values}) > 1
        rows.append(
            {
                "key": key,
                "label": _fact_label(key),
                "values": values,
                "mismatch": mismatch,
            }
        )
    return {
        "headers": ["Configuration"] + nodes + ["Drift"],
        "rows": rows,
        "mismatch_count": sum(1 for row in rows if row["mismatch"]),
    }


def _firmware_table(results):
    rows = []
    for node, gpu_entries in sorted((results.get("firmware") or {}).items()):
        if not isinstance(gpu_entries, dict):
            continue
        for gpu, firmware in sorted(gpu_entries.items()):
            if not isinstance(firmware, dict):
                continue
            for firmware_id, version in sorted(firmware.items()):
                rows.append([node, gpu, firmware_id, version])
    if not rows:
        return {}
    return {
        "headers": ["Node", "GPU", "Firmware", "Version"],
        "rows": rows,
    }


@register_dataset_builder("host_inventory")
def build_host_inventory_datasets(sources, profile):
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    nodes = _node_names(results)
    return {
        "inventory_table": _inventory_table(results, profile, nodes),
        "drift_matrix": _drift_matrix(results, nodes),
        "firmware_table": _firmware_table(results),
        "node_count": len(nodes),
    }
