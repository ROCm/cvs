'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unified SGLang run-card hook for all single / distributed / disagg suite stems.
'''

from __future__ import annotations

from typing import Any, List, Tuple

from cvs.lib.inference.sglang.sglang_common import as_node_list, resolve_server_node_list
from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _format_nodes(raw: Any) -> str:
    if not raw:
        return "\u2014"
    hosts = as_node_list(raw)
    return ", ".join(hosts) if hosts else "\u2014"


def sglang_run_card_display(variant: Any, provenance: dict) -> List[Tuple[str, str, bool]]:
    """Topology-aware run card shared by every SGLang suite entry point."""
    bp = getattr(variant, "benchmark_params", None) or {}
    inf = getattr(variant, "inference", None) or {}
    rows: List[Tuple[str, str, bool]] = [
        ("Model", variant.model.id, False),
        ("GPU", variant.gpu_arch, False),
    ]

    if inf.get("prefill_node_list") or inf.get("decode_node_list") or inf.get("proxy_router_node"):
        rows.extend(
            [
                ("Prefill nodes", _format_nodes(inf.get("prefill_node_list")), False),
                ("Decode nodes", _format_nodes(inf.get("decode_node_list")), False),
                ("Proxy router", _format_nodes(inf.get("proxy_router_node")), False),
                ("Benchmark node", _format_nodes(inf.get("benchmark_serv_node")), False),
            ]
        )
    else:
        try:
            server_nodes = resolve_server_node_list(inf)
        except ValueError:
            server_nodes = []
        if len(server_nodes) > 1 or inf.get("nnodes", 1) not in (1, "1", None):
            rows.extend(
                [
                    ("Server nodes", ", ".join(server_nodes) if server_nodes else "\u2014", False),
                    ("nnodes", str(inf.get("nnodes", len(server_nodes) or "-")), False),
                    ("Benchmark node", _format_nodes(inf.get("benchmark_serv_node")), False),
                ]
            )
        else:
            bench_raw = inf.get("benchmark_serv_node")
            bench_node = as_node_list(bench_raw)[0] if bench_raw else "\u2014"
            rows.append(("Benchmark node", bench_node, False))

    rows.extend(
        [
            ("TP", str(bp.get("tensor_parallelism", "-")), False),
            ("PP", str(bp.get("pipeline_parallelism", "-")), False),
            thresholds_run_card_row(variant),
        ]
    )
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = ["sglang_run_card_display"]
