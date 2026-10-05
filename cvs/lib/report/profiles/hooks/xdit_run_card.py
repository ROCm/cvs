"""
xDiT Run Deck run-card hook.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _format_nodes(raw):
    if not raw:
        return "\u2014"
    if isinstance(raw, (list, tuple)):
        hosts = [str(host) for host in raw if host]
    else:
        hosts = [str(raw)]
    return ", ".join(hosts) if hosts else "\u2014"


def _nnodes(variant, inference):
    nnodes = inference.get("nnodes")
    if nnodes is not None and str(nnodes).strip() != "":
        return str(nnodes)
    nodes = _execution_hosts(inference)
    if variant.topology == "distributed":
        return str(max(len(nodes), 1))
    return "1"


def _server_nodes(inference):
    return _format_nodes(_execution_hosts(inference))


def _execution_hosts(inference):
    return inference.get("_execution_hosts") or inference.get("server_node_list") or []


def _benchmark_node(variant, inference):
    hosts = _execution_hosts(inference)
    if not hosts:
        return "\u2014"
    if variant.topology != "distributed":
        return _format_nodes(hosts)
    return str(hosts[0])


def _ulysses_ring(params):
    nproc = int(params.get("torchrun_nproc") or 1)
    ulysses = params.get("ulysses_degree", params.get("ulysses_size", nproc))
    ring = params.get("ring_degree", params.get("ring_size", 1))
    return int(ulysses), int(ring)


def _workload_params(variant):
    if variant.benchmark_params.get("flux1_dev_t2i"):
        return variant.benchmark_params["flux1_dev_t2i"]
    return variant.benchmark_params.get("wan22_i2v_a14b") or {}


def xdit_run_card_display(variant, provenance):
    inference = variant.inference or {}
    ulysses, ring = _ulysses_ring(_workload_params(variant))
    rows = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
        ("Server nodes", _server_nodes(inference), False),
        ("nnodes", _nnodes(variant, inference), False),
        ("Benchmark node", _benchmark_node(variant, inference), False),
        ("Ulysses", str(ulysses), False),
        ("Ring", str(ring), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["xdit_run_card_display"]
