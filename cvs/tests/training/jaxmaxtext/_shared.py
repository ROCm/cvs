'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.jax.utils.maxtext_parsing import TRAINING_METRICS

log = globals.log


def test_print_results_table(training_res_dict):
    """Print a summary table of training results."""
    results = training_res_dict.get("results", {})
    if not results:
        log.info("training_res_dict empty, nothing to print")
        return

    headers = ["Metric", "Value", "Unit"]
    rows = []
    for short, unit in TRAINING_METRICS:
        full = "training." + short
        val = results.get(full)
        if isinstance(val, float):
            rows.append([short, f"{val:.4f}", unit])
        else:
            rows.append([short, str(val), unit])
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    step_metrics = training_res_dict.get("step_metrics", [])
    if step_metrics:
        loss_rows = []
        for s in step_metrics:
            if "loss" in s:
                loss_rows.append([s["step"], f"{s['loss']:.6f}"])
        if loss_rows:
            log.info("\n\nLoss Curve:\n" + tabulate(loss_rows, headers=["Step", "Loss"], tablefmt="github"))
