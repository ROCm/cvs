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


def _nnodes(variant, inference):
    nnodes = inference.get("nnodes")
    if nnodes is not None and str(nnodes).strip() != "":
        return int(nnodes)
    nodes = inference.get("_execution_hosts") or inference.get("server_node_list") or []
    if variant.topology == "distributed":
        return max(len(nodes), 1)
    return 1


def _ulysses_ring(params):
    nproc = int(params.get("torchrun_nproc") or 1)
    ulysses = params.get("ulysses_degree", params.get("ulysses_size", nproc))
    ring = params.get("ring_degree", params.get("ring_size", 1))
    return int(ulysses), int(ring)


def xdit_run_card_display(variant, provenance):
    workload, params = _workload(variant)
    inference = variant.inference or {}
    nproc = int(params.get("torchrun_nproc") or 1)
    ulysses, ring = _ulysses_ring(params)
    workers = _nnodes(variant, inference) * nproc
    rows = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Workload", workload, False),
        ("Topology", variant.topology, False),
        ("Execution nodes", _nodes(variant), False),
        ("GPUs/node", str(nproc), False),
        ("Workers", str(workers), False),
        ("Ulysses", str(ulysses), False),
        ("Ring", str(ring), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["xdit_run_card_display"]
