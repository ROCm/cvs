'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared test helpers for the pytorch_vision suite.

`test_print_results_table` is imported by the suite module so pytest runs it as a
sibling test that lands LAST (pinned in pytest_collection_modifyitems). It logs
whatever cells were recorded -- a partial sweep still produces a useful table.
'''

from tabulate import tabulate

from cvs.lib import globals

log = globals.log

__all__ = ["test_print_results_table"]


def _cell(m, key):
    """Table cell: a missing OR present-but-None metric renders as '-'."""
    v = m.get(key)
    return "-" if v is None else v


def test_print_results_table(res_dict):
    if not res_dict:
        log.info("res_dict empty, nothing to print")
        return
    headers = [
        "Model",
        "GPU",
        "Precision",
        "Res",
        "BS",
        "Host",
        "Throughput (img/s)",
        "Mean lat (ms)",
        "P50 lat (ms)",
        "P90 lat (ms)",
        "P99 lat (ms)",
    ]
    rows = []
    for key, host_dict in res_dict.items():
        arch, gpu, precision, input_size, _combo_name, batch_size = key
        for host, m in host_dict.items():
            rows.append(
                [
                    arch,
                    gpu,
                    precision,
                    input_size,
                    batch_size,
                    host,
                    _cell(m, "vision.throughput_img_s"),
                    _cell(m, "vision.latency_ms_mean"),
                    _cell(m, "vision.latency_ms_p50"),
                    _cell(m, "vision.latency_ms_p90"),
                    _cell(m, "vision.latency_ms_p99"),
                ]
            )
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
