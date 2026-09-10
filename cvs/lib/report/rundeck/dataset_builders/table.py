'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Table dataset builder for pass/fail suites (RVS, similar health tests).
'''

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _flatten_node_results(results):
    rows = []
    for test_name in sorted((results or {}).keys(), key=str):
        nodes = results[test_name]
        if not isinstance(nodes, dict):
            continue
        for node in sorted(nodes.keys(), key=str):
            rows.append([str(test_name), str(node), str(nodes[node])])
    return rows


@register_dataset_builder("table")
def build_table_datasets(sources, profile):
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    table_cfg = (profile or {}).get("table") or {}
    headers = table_cfg.get("headers") or ["Test", "Node", "Result"]
    return {"results_table": {"headers": headers, "rows": _flatten_node_results(results)}}
