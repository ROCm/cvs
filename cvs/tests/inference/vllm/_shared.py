'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared test helpers for the dtni vllm_single suite.

`test_print_results_table` is exported via `from ._shared import *` so each
framework-specific suite file picks it up as a sibling test that pytest
runs LAST (lexically after `test_vllm_inference`).
'''

from tabulate import tabulate

from cvs.lib import globals

log = globals.log

__all__ = ["test_print_results_table"]


def test_print_results_table(inf_res_dict):
    if not inf_res_dict:
        log.info("inf_res_dict empty, nothing to print")
        return
    headers = [
        "Model", "GPU", "ISL", "OSL", "Policy", "Conc", "Host",
        "Req/s", "Total tok/s", "Mean TTFT (ms)", "Mean TPOT (ms)", "P99 ITL (ms)",
    ]
    rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        for host, m in host_dict.items():
            rows.append([
                model, gpu, isl, osl, policy, conc, host,
                m.get("request_throughput_per_sec", "-"),
                m.get("total_throughput_per_sec", "-"),
                m.get("mean_ttft_ms", "-"),
                m.get("mean_tpot_ms", "-"),
                m.get("p99_itl_ms", "-"),
            ])
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
