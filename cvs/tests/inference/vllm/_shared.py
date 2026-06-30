'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared test helpers for the vllm_single suite.

`test_print_results_table` is exported via `from ._shared import *` so each
framework-specific suite file picks it up as a sibling test that pytest
runs LAST (lexically after `test_vllm_inference`).
'''

from tabulate import tabulate

from cvs.lib import globals

log = globals.log

__all__ = ["test_print_results_table"]


def _cell(m, key):
    """Table cell: a missing OR present-but-None metric renders as '-'."""
    v = m.get(key)
    return "-" if v is None else v


def test_print_results_table(inf_res_dict):
    if not inf_res_dict:
        log.info("inf_res_dict empty, nothing to print")
        return
    headers = [
        "Model",
        "GPU",
        "ISL",
        "OSL",
        "Policy",
        "Conc",
        "Host",
        "Req/s",
        "Total tok/s",
        "Mean TTFT (ms)",
        "P95 TTFT (ms)",
        "Mean TPOT (ms)",
        "P95 TPOT (ms)",
        "P99 ITL (ms)",
        "Goodput (req/s)",
        "Peak VRAM (MB)",
        "Compute %",
        "BW %",
    ]
    rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        for host, m in host_dict.items():
            rows.append(
                [
                    model,
                    gpu,
                    isl,
                    osl,
                    policy,
                    conc,
                    host,
                    _cell(m, "client.request_throughput"),
                    _cell(m, "client.total_token_throughput"),
                    _cell(m, "client.mean_ttft_ms"),
                    _cell(m, "client.p95_ttft_ms"),
                    _cell(m, "client.mean_tpot_ms"),
                    _cell(m, "client.p95_tpot_ms"),
                    _cell(m, "client.p99_itl_ms"),
                    _cell(m, "client.goodput"),
                    _cell(m, "gpu.peak_gpu_memory_mb"),
                    _cell(m, "gpu.gpu_compute_util_pct"),
                    _cell(m, "gpu.gpu_bandwidth_util_pct"),
                ]
            )
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
