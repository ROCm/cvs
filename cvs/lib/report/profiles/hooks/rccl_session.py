'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared Run Deck session helpers for RCCL suite stems.
'''

from types import SimpleNamespace

from cvs.lib import rccl_lib


def variant_from_config(config_dict, cluster_dict):
    node_dict = (cluster_dict or {}).get("node_dict") or {}
    cvs_params = (config_dict or {}).get("cvs_params") or {}
    verify = str(cvs_params.get("verify_bus_bw") or "").lower() in ("true", "1", "yes")
    return SimpleNamespace(
        framework="rccl",
        gpu_arch=(config_dict or {}).get("gpu_arch") or "\u2014",
        nnodes=len(node_dict),
        mpi_params=(config_dict or {}).get("mpi_params") or {},
        rccl_test_params=(config_dict or {}).get("rccl_test_params") or {},
        cvs_params=cvs_params,
        enforce_thresholds=verify,
    )


def publish_graph(raw_results, store):
    graph = rccl_lib.convert_to_graph_dict(raw_results or {})
    store.clear()
    store.update(graph)
    return graph
