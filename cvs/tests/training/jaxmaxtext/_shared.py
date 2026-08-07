'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import html as _html

from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import TRAINING_METRICS

log = globals.log

_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "N/A": "#f9a825",
    "RECORD": "#555555",
}


def _write_metric_results_html(training_res_dict, request):
    """Write ALL metric verdicts (expected/actual/status) to one HTML file.

    The file lands in the report bundle dir; every parametrized `test_metric`
    row links to this same file (see the suite conftest makereport hook). No-op
    when HTML reporting is disabled or no metric rows were collected.
    """
    metric_rows = training_res_dict.get("metric_rows") or []
    mgr = getattr(request.config, "_html_report_manager", None)
    if not metric_rows or mgr is None or not getattr(mgr, "is_enabled", False):
        return
    try:
        out_dir = mgr.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metric_results.html"
        body = ""
        for r in metric_rows:
            color = _STATUS_COLORS.get(r["status"], "#000000")
            body += (
                "<tr>"
                f"<td>{_html.escape(str(r['metric']))}</td>"
                f"<td>{_html.escape(str(r['expected']))}</td>"
                f"<td>{_html.escape(str(r['actual']))}</td>"
                f"<td>{_html.escape(str(r['unit']))}</td>"
                f"<td style=\"color:{color};font-weight:bold;\">{_html.escape(str(r['status']))}</td>"
                "</tr>"
            )
        doc = (
            "<html><head><meta charset='utf-8'><title>Training Metric Results</title></head>"
            "<body><h2>Training Metric Results</h2>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Metric</th><th>Expected</th><th>Actual</th><th>Unit</th><th>Status</th></tr>"
            f"{body}</table></body></html>"
        )
        path.write_text(doc, encoding="utf-8")
        log.info("wrote metric results HTML: %s", path)
    except Exception as e:  # noqa: BLE001 - reporting must never break the run
        log.warning("could not write metric results HTML: %s", e)


def test_print_results_table(training_res_dict, request):
    """Print a summary table of training results and write the metric-results HTML."""
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

    _write_metric_results_html(training_res_dict, request)
