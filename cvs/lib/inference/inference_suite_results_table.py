'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Reusable end-of-session results table for DTNI inference suites.

Bind a column preset with ``make_print_results_table(columns)`` and export the
returned callable as ``test_print_results_table`` from the suite module (see
``cvs/tests/inference/inferencex_atom/_shared.py``). Other suites can supply
their own ``(header_label, client.*_key)`` tuples without duplicating the
tabulate loop.
'''

from __future__ import annotations

from tabulate import tabulate

from cvs.lib import globals

log = globals.log

# Column tuple: ``(header label, client.* metric key or None for fixed key fields)``.
# First seven columns are always Model, GPU, ISL, OSL, Policy, Conc, Host.

INFERENCEX_ATOM_RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    ("Output tok/s", "client.output_throughput"),
    ("Total tok/s", "client.total_token_throughput"),
    ("Mean TTFT (ms)", "client.mean_ttft_ms"),
    ("Mean TPOT (ms)", "client.mean_tpot_ms"),
    ("P99 ITL (ms)", "client.p99_itl_ms"),
)

# Optional preset for suites that want vLLM-style columns (not wired in vllm_single yet).
VLLM_SINGLE_RESULTS_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    ("Req/s", "client.request_throughput"),
    ("Total tok/s", "client.total_token_throughput"),
    ("Mean TTFT (ms)", "client.mean_ttft_ms"),
    ("P95 TTFT (ms)", "client.p95_ttft_ms"),
    ("Mean TPOT (ms)", "client.mean_tpot_ms"),
    ("P95 TPOT (ms)", "client.p95_tpot_ms"),
    ("P99 ITL (ms)", "client.p99_itl_ms"),
    ("Goodput (req/s)", "client.goodput"),
)


def _cell(metrics, key):
    v = metrics.get(key)
    return "-" if v is None else v


def print_results_table(inf_res_dict, columns):
    if not inf_res_dict:
        log.info("inf_res_dict empty, nothing to print")
        return
    headers = [label for label, _key in columns]
    metric_keys = [key for _label, key in columns]
    rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        fixed = [model, gpu, isl, osl, policy, conc]
        for host, metrics in host_dict.items():
            row = list(fixed)
            row.append(host)
            row.extend(_cell(metrics, mk) if mk else "-" for mk in metric_keys[7:])
            rows.append(row)
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def make_print_results_table(columns):
    """Return a ``test_print_results_table(inf_res_dict)`` callable for one column spec."""

    def test_print_results_table(inf_res_dict):
        print_results_table(inf_res_dict, columns)

    return test_print_results_table
