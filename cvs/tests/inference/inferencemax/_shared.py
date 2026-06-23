'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from tabulate import tabulate

from cvs.lib import globals

log = globals.log

__all__ = ["test_print_results_table"]


def _cell(m, key):
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
        "Output tok/s",
        "Total tok/s",
        "Mean TTFT (ms)",
        "Mean TPOT (ms)",
        "P99 ITL (ms)",
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
                    _cell(m, "client.output_throughput"),
                    _cell(m, "client.total_token_throughput"),
                    _cell(m, "client.mean_ttft_ms"),
                    _cell(m, "client.mean_tpot_ms"),
                    _cell(m, "client.p99_itl_ms"),
                ]
            )
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
