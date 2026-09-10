'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared test helpers for the unified vllm suite.

`test_print_results_table` is imported by both explicit vLLM suites and runs
after `test_vllm_inference`.
'''

from tabulate import tabulate
import pytest

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_metrics import VLLM_RESULTS_COLUMNS

log = globals.log

__all__ = ["test_print_results_table", "validate_vllm_execution_mode"]


def validate_vllm_execution_mode(pytestconfig):
    """Reject execution modes that split or overwrite process-local cell results."""
    workers = pytestconfig.getoption("numprocesses", default=0)
    if workers not in (None, 0, "0"):
        raise pytest.UsageError("vLLM suites require serial pytest execution; xdist is unsupported")

    repeat_count = pytestconfig.getoption("count", default=1)
    if repeat_count not in (None, 1, "1"):
        raise pytest.UsageError("vLLM suites do not support pytest-repeat counts above one")


def _cell(m, key):
    """Table cell: a missing OR present-but-None metric renders as '-'."""
    v = m.get(key)
    return "-" if v is None else v


def test_print_results_table(inf_res_dict):
    if not inf_res_dict:
        log.info("inf_res_dict empty, nothing to print")
        return
    headers = [label for label, _metric in VLLM_RESULTS_COLUMNS]
    metric_keys = [metric for _label, metric in VLLM_RESULTS_COLUMNS[7:]]
    rows = []
    for key, host_dict in inf_res_dict.items():
        model, gpu, isl, osl, policy, conc = key
        for host, m in host_dict.items():
            rows.append([model, gpu, isl, osl, policy, conc, host] + [_cell(m, metric) for metric in metric_keys])
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
