'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import html as _html

from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import TRAINING_METRICS
from cvs.lib.utils_lib import fail_test, update_test_result

log = globals.log

_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "N/A": "#f9a825",
    "RECORD": "#555555",
}


def _write_metric_results_html(training_res_dict, request):
    """Write ALL metric verdicts (sweep/metric/expected/actual/status) to ONE HTML
    file in the report bundle dir. Every parametrized `test_metric` row's Full Log
    link opens this same file (see the suite conftest makereport hook). No-op when
    HTML reporting is disabled or no metric rows were collected.
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
                f"<td>{_html.escape(str(r.get('sweep', '-')))}</td>"
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
            "<tr><th>Sweep</th><th>Metric</th><th>Expected</th><th>Actual</th><th>Unit</th><th>Status</th></tr>"
            f"{body}</table></body></html>"
        )
        path.write_text(doc, encoding="utf-8")
        log.info("wrote metric results HTML: %s", path)
    except Exception as e:  # noqa: BLE001 - reporting must never break the run
        log.warning("could not write metric results HTML: %s", e)


def _print_sweep_tables(training_res_dict):
    """Log a per-sweep metric table + loss curve to the console."""
    sweeps = training_res_dict.get("sweeps", {})
    if not sweeps:
        log.info("no sweep results to print")
        return
    for sweep_name, rec in sweeps.items():
        results = rec.get("results", {})
        rows = []
        for short, unit in TRAINING_METRICS:
            val = results.get("training." + short)
            rows.append([short, f"{val:.4f}" if isinstance(val, float) else str(val), unit])
        log.info("\n[sweep %s]\n%s", sweep_name, tabulate(rows, headers=["Metric", "Value", "Unit"], tablefmt="github"))

        loss_rows = [[s["step"], f"{s['loss']:.6f}"] for s in rec.get("step_metrics", []) if "loss" in s]
        if loss_rows:
            log.info("\nLoss Curve [%s]:\n%s", sweep_name, tabulate(loss_rows, headers=["Step", "Loss"], tablefmt="github"))


def test_print_results_table(training_res_dict, request):
    """Summarize all sweeps: console tables, single metric-results HTML, and a
    consolidated PASS/FAIL summary recorded via globals.error_list for the pytest
    final summary."""
    if not training_res_dict.get("sweeps"):
        log.info("training_res_dict empty, nothing to print")
        return

    _print_sweep_tables(training_res_dict)
    _write_metric_results_html(training_res_dict, request)

    # Consolidated failure summary for the pytest final summary / console log.
    # Individual metric rows already fail per (sweep, metric); this aggregates
    # them into one message via the shared globals.error_list helpers.
    failures = training_res_dict.get("metric_failures", [])
    globals.error_list = []
    for f in failures:
        fail_test(f)
    update_test_result()
