'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared Run Deck session helpers for RCCL suite stems.
'''

from types import SimpleNamespace

from cvs.lib import globals, rccl_lib


def variant_from_config(config_dict, cluster_dict, suite_name=None, raw_results=None, run_nodes=None):
    """Describe RCCL configuration, retaining live results for the final run card."""
    config_dict = config_dict or {}
    node_dict = (cluster_dict or {}).get("node_dict") or {}
    cvs_params = config_dict.get("cvs_params") or {}
    test_params = dict(config_dict.get("rccl_test_params") or {})
    if suite_name == "rccl_pairwise":
        test_params["rccl_collective"] = ["all_reduce_perf"]
    elif "rccl_collective" in config_dict:
        test_params["rccl_collective"] = config_dict["rccl_collective"]
    elif suite_name == "rccl_regression":
        test_params["rccl_collective"] = ["all_reduce_perf"]
    mpi_params = {"no_of_nodes": "2", "no_of_local_ranks": "8"}
    mpi_params.update(config_dict.get("mpi_params") or {})

    def _on(key):
        return str(cvs_params.get(key) or "").lower() in ("true", "1", "yes")

    verify = _on("verify_bus_bw") or _on("verify_bw_dip") or _on("verify_lat_dip")
    return SimpleNamespace(
        framework="rccl",
        gpu_arch=config_dict.get("gpu_arch") or "\u2014",
        nnodes=len(node_dict),
        mpi_params=mpi_params,
        rccl_test_params=test_params,
        raw_results=raw_results,
        run_nodes=run_nodes,
        cvs_params=cvs_params,
        enforce_thresholds=verify,
    )


def publish_graph(raw_results, store):
    """Return the legacy graph even if publishing it to the optional deck fails."""
    graph = rccl_lib.convert_to_graph_dict(raw_results or {})
    try:
        # The module binder holds this dict, so replacing it would lose the results.
        store.clear()
        store.update(graph)
    except Exception:
        globals.log.warning("RCCL Run Deck results unavailable", exc_info=True)
    return graph
