"""
xDiT Run Deck run-card hook.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _nodes(variant):
    inference = variant.inference
    nodes = inference.get("_execution_hosts") or []
    if not nodes:
        if variant.topology == "distributed":
            nodes = inference.get("server_node_list") or []
        else:
            nodes = [inference.get("benchmark_serv_node")]
    return ", ".join(str(node) for node in nodes if node) or "\u2014"


def _workload(variant):
    if variant.benchmark_params.get("flux1_dev_t2i"):
        return "FLUX", variant.benchmark_params["flux1_dev_t2i"]
    params = variant.benchmark_params.get("wan22_i2v_a14b") or {}
    backend = "Diffusers" if params.get("model_format") == "diffusers" else "native"
    return f"WAN ({backend})", params


def xdit_run_card_display(variant, provenance):
    workload, params = _workload(variant)
    rows = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Workload", workload, False),
        ("Topology", variant.topology, False),
        ("Execution nodes", _nodes(variant), False),
        ("Workers/node", str(params.get("torchrun_nproc", "-")), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["xdit_run_card_display"]
