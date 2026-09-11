'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Run-card fields for the host configuration inventory deck.
'''

from cvs.lib.platform.host_inventory import FACT_LABELS, collected_fact_keys
from cvs.lib.report.rundeck.config_builder import provenance_link_rows


def host_configs_run_card_display(results, provenance):
    results = results or {}
    nodes = set((results.get("nodes") or {}).keys())
    nodes.update((results.get("firmware") or {}).keys())
    fact_keys = collected_fact_keys(results)
    fact_labels = [FACT_LABELS.get(key, key.replace("_", " ").title()) for key in fact_keys]
    rows = [
        ("Nodes sampled", str(len(nodes)), False),
        ("Fact categories", str(len(fact_keys)), False),
        ("Facts collected", ", ".join(fact_labels) or "\u2014", False),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["host_configs_run_card_display"]
