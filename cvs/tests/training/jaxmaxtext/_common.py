'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared implementations for the JAX MaxText training suites. This is NOT a
runnable suite (leading underscore -> excluded by `cvs list`/`cvs run`); the two
suite files import these helpers and wrap each as an explicit `test_*` method:

  - jaxmaxtext_single.py       (single-node: no RDMA stage)
  - jaxmaxtext_distributed.py  (adds the test_setup_rdma stage)

Kept deliberately simple: plain shared functions + a couple of small helpers,
no framework-y generalization. The single vs distributed mode is recorded in
`training_res_dict["mode"]` and reflected in the console tables, the metric
results HTML title, and the loss-curve title/artifact.
'''

import html as _html
import json
import re
import shlex
import time
import uuid as _uuid
from pathlib import Path as _Path

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib import MaxTextTrainingJob
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import (
    TRAINING_METRICS,
    TRAINING_METRIC_UNITS,
    compute_scaling_efficiency,
    compute_convergence,
    sample_loss_curve,
    evaluate_loss_decreasing,
)
from cvs.lib.training.jaxmaxtext.utils.loss_curve import render_loss_curve_png
from cvs.lib.utils.verdict import evaluate_all, ThresholdViolation
from cvs.lib.utils_lib import fail_test, update_test_result

log = globals.log

_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "N/A": "#f9a825",
    "RECORD": "#555555",
}


# ---------- small helpers ----------


def _sweep_label(name):
    """Compact, unique-per-sweep id used in every parametrized test row and in
    the reports: PRECISION[-SL<seqlen>][-B<batch>], e.g. "BF16-SL4096-B3".

    The full sweep name still drives results/threshold lookups; this is only the
    display label. Falls back to a sanitized full name when PRECISION is absent.
    """
    name = name or ""

    def _tok(key):
        m = re.search(rf"{key}=([^,]+)", name)
        return m.group(1).strip() if m else None

    precision = _tok("PRECISION")
    seqlen = _tok("SEQLEN")
    batch = _tok("BATCH")

    parts = []
    if precision:
        parts.append(precision)
    if seqlen:
        parts.append(f"SL{seqlen}")
    if batch:
        parts.append(f"B{batch}")
    if parts:
        return "-".join(parts)
    return (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")) or "default"


def _enabled_sweep_names(config_file):
    """Read sweep names to run from the raw config (collection time, no fixtures).

    Honors training.enabled_sweep_list (subset selector); falls back to every
    declared sweep, or a single implicit "default" when none are declared.
    """
    try:
        with open(config_file) as fp:
            raw = json.load(fp)
    except Exception:
        return ["default"]
    training = raw.get("training", {})
    names = [s.get("name") for s in training.get("sweeps", []) if s.get("name")]
    if not names:
        return ["default"]
    enabled = training.get("enabled_sweep_list") or names
    return [n for n in enabled if n in names] or names


def _find_sweep(variant_config, sweep_name):
    for s in variant_config.enabled_sweeps():
        if s.name == sweep_name:
            return s
    return None


def _mode(variant_config):
    return "distributed" if variant_config.training.distributed else "single"


def _format_expected(spec):
    """Human-readable expected-threshold string for the console log + summary file."""
    if not spec:
        return "-"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "info":
        return f"info ({value})" if value is not None else "info"
    if kind in ("min", "min_tok_s"):
        return f">= {value}"
    if kind == "max":
        return f"<= {value}"
    if kind == "max_ms":
        return f"<= {value} ms"
    if kind == "within":
        return f"{value} +/-{spec.get('tolerance_pct')}%"
    if kind == "min_ratio":
        return f">= {value} x {spec.get('reference')}"
    return str(spec)


def _format_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# ---------- lifecycle stage implementations ----------
# Plain helpers (no test_ prefix): the suite files wrap each as a `test_*`
# method with a docstring so the suites read as self-documenting.


def _precreate_tmp_bind_mounts(orch):
    """Create host-side ``/tmp/...`` bind-mount source dirs before launch.

    Docker auto-creates a missing bind-mount source directory owned by root; a
    leftover root-owned dir (``docker system prune`` does not remove host bind
    dirs) then blocks the next user on a shared GPU node with a permission
    error. Creating them here over SSH -- via ``exec_on_host``, which runs on
    the cluster host OS as the invoking user -- makes them user-owned instead.

    Only ``/tmp/`` sources are touched so device/system mounts (``/dev/*``,
    ``/lib/*``) are never created.
    """
    exec_host = getattr(orch, "exec_on_host", None)
    if not callable(exec_host):
        return
    try:
        volumes = orch.get_volumes()
    except Exception:  # noqa: BLE001 - best-effort; docker still auto-creates
        return
    sources, seen = [], set()
    for vol in volumes or []:
        src = str(vol).split(":", 1)[0].strip()
        if src.startswith("/tmp/") and src not in seen:
            seen.add(src)
            sources.append(src)
    if not sources:
        return
    quoted = " ".join(shlex.quote(p) for p in sources)
    try:
        exec_host(f"mkdir -p {quoted}")
    except Exception:  # noqa: BLE001 - non-fatal; fall back to docker auto-create
        pass


def launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container. Verify it is running."""
    t = time.monotonic()
    _precreate_tmp_bind_mounts(orch)
    ok = orch.setup_containers()
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        pytest.fail(f"setup_containers() returned False for {name}")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def setup_rdma(orch, variant_config, hf_token, lifecycle, request):
    """Distributed-only: copy RDMA library into container (thor2 NIC only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: RDMA not needed")
    if not variant_config.training.nic_type or "thor" not in variant_config.training.nic_type.lower():
        pytest.skip(f"nic_type={variant_config.training.nic_type}: RDMA lib copy not needed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_rdma_lib()
    lifecycle.record(request.node.nodeid, "rdma_setup", time.monotonic() - t)


def setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Download HF tokenizer into models dir."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_tokenizer()
    lifecycle.record(request.node.nodeid, "tokenizer_setup", time.monotonic() - t)


def training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request):
    """Per sweep: build the command, train, poll, parse results.

    Runs once per enabled sweep with that sweep's maxtext overrides. A failure is
    isolated to this sweep's row (it does NOT set lifecycle.failed) so the other
    sweeps still run and report.
    """
    training_res_dict.setdefault("mode", _mode(variant_config))
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    sweep = _find_sweep(variant_config, sweep_name)
    job = MaxTextTrainingJob(orch, variant_config, hf_token, sweep=sweep)
    try:
        job.setup_training_env()
        job.build_training_cmd()
        t = time.monotonic()
        job.start_training()
        job.poll_for_completion()
        wall_time = time.monotonic() - t
        results = job.parse_results()
    except Exception as e:  # noqa: BLE001 - isolate the failure to this sweep
        log.error("training run failed for sweep '%s': %s", sweep_name, e)
        # Reap any lingering ranks so the next sweep does not launch on top of
        # them (and so persistent containers are not left with orphan processes).
        try:
            job.stop_training()
        except Exception:  # noqa: BLE001
            pass
        pytest.fail(f"training run failed for sweep '{sweep_name}': {e}")

    results["training.wall_time_seconds"] = wall_time
    results["training.convergence_steps"] = variant_config.training.steps
    results["training.convergence_wall_time"] = wall_time

    baseline = variant_config.training.scaling_baseline
    results["training.scaling_efficiency_pct"] = compute_scaling_efficiency(
        results.get("training.tokens_per_sec_total"),
        job.num_nodes,
        baseline.tokens_per_sec_total,
        baseline.num_nodes,
    )

    conv = variant_config.training.convergence
    steps_to_target, time_to_target = compute_convergence(
        job.step_metrics,
        job.eval_metrics,
        conv.target_metric,
        conv.target_value,
    )
    results["training.steps_to_target"] = steps_to_target
    results["training.time_to_target_seconds"] = time_to_target

    training_res_dict.setdefault("sweeps", {})[sweep_name] = {
        "results": results,
        "step_metrics": job.step_metrics,
        "eval_metrics": job.eval_metrics,
        "num_nodes": job.num_nodes,
    }


def metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request):
    """One test (row) per (sweep, metric). Threshold-driven PASS/FAIL; logs
    `sweep | metric | expected | actual | status` and collects rows for the
    single metric-results HTML file (linked from every metric row)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    rec = training_res_dict.get("sweeps", {}).get(sweep_name)
    results = rec.get("results") if rec else None
    if not results:
        pytest.skip(f"no results for sweep '{sweep_name}' (training did not complete)")

    label = _sweep_label(sweep_name)
    full = "training." + metric
    value = results.get(full)
    unit = TRAINING_METRIC_UNITS.get(metric, "-")
    # The sweep name IS the threshold cell key.
    spec = (variant_config.thresholds.get(sweep_name) or {}).get(full)
    expected = _format_expected(spec)
    actual = _format_value(value)

    rows = training_res_dict.setdefault("metric_rows", [])

    def _record(status):
        rows.append(
            {
                "sweep": label,
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "unit": unit,
                "status": status,
            }
        )

    if value is None:
        log.info("[metric] %-6s %-24s | expected %-14s | actual None | %s -> N/A", label, metric, expected, unit)
        _record("N/A")
        pytest.skip(f"{metric}: no value produced this run")

    if spec is None or not variant_config.enforce_thresholds:
        log.info(
            "[metric] %-6s %-24s | expected %-14s | actual %s | %s -> RECORD", label, metric, expected, actual, unit
        )
        _record("RECORD")
        return

    try:
        evaluate_all(results, {full: spec})
    except ThresholdViolation as e:
        log.error(
            "[metric] %-6s %-24s | expected %-14s | actual %s | %s -> FAIL", label, metric, expected, actual, unit
        )
        _record("FAIL")
        training_res_dict.setdefault("metric_failures", []).append(
            f"[{label}] {metric}: expected {expected}, actual {actual}"
        )
        pytest.fail(str(e))
    else:
        log.info("[metric] %-6s %-24s | expected %-14s | actual %s | %s -> PASS", label, metric, expected, actual, unit)
        _record("PASS")


def loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request):
    """Row 32 (per sweep): sample the training loss, render a PNG, gate on trend."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    label = _sweep_label(sweep_name)
    mode = _mode(variant_config)
    rec = training_res_dict.get("sweeps", {}).get(sweep_name)
    step_metrics = rec.get("step_metrics") if rec else None
    if not step_metrics:
        pytest.skip(f"no step metrics for sweep '{sweep_name}' (training did not complete)")

    cfg = variant_config.training.loss_curve
    points = sample_loss_curve(step_metrics, cfg.sample_every, cfg.milestone_steps)
    verdict = evaluate_loss_decreasing(points, cfg.max_slope)

    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is not None and getattr(mgr, "is_enabled", False):
        out_dir = mgr.log_dir
    else:
        out_dir = _Path(variant_config.paths.log_dir)
    png_path = None
    try:
        _Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = f"loss_curve_{variant_config.model.id}_{mode}_{label}_{str(_uuid.uuid4()).split('-')[-1]}.png"
        abs_path = _Path(out_dir) / fname
        title = f"Training Loss Curve — {variant_config.model.id} [{mode}/{label}]"
        png_path = render_loss_curve_png(points, abs_path, title=title)
    except Exception as e:  # noqa: BLE001 - plotting must never break the verdict
        log.warning("loss curve: could not prepare PNG output (%s)", e)

    if png_path and mgr is not None and getattr(mgr, "is_enabled", False):
        try:
            rel_path = str(_Path(png_path).relative_to(mgr.htmlpath.parent))
            lifecycle.add_artifact(request.node.nodeid, f"Loss Curve [{mode}/{label}]", rel_path, str(png_path))
        except Exception as e:  # noqa: BLE001
            log.warning("loss curve: could not register report link (%s)", e)

    if verdict is not None:
        _decreasing, _slope, detail = verdict
        log.info("loss curve: %s", detail)

    if verdict is None:
        pytest.skip(f"loss curve needs >= 2 sampled points (got {len(points)})")
    decreasing, _slope, detail = verdict
    if cfg.enforce and not decreasing:
        pytest.fail(f"training loss is not decreasing: {detail}")


# ---------- reporting ----------


def _write_metric_results_html(training_res_dict, request):
    """Write ALL metric verdicts to ONE HTML file in the report bundle dir."""
    metric_rows = training_res_dict.get("metric_rows") or []
    mgr = getattr(request.config, "_html_report_manager", None)
    if not metric_rows or mgr is None or not getattr(mgr, "is_enabled", False):
        return
    mode = training_res_dict.get("mode", "")
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
        title = f"Training Metric Results ({mode})" if mode else "Training Metric Results"
        doc = (
            f"<html><head><meta charset='utf-8'><title>{_html.escape(title)}</title></head>"
            f"<body><h2>{_html.escape(title)}</h2>"
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
    mode = training_res_dict.get("mode", "")
    if not sweeps:
        log.info("no sweep results to print")
        return
    for sweep_name, rec in sweeps.items():
        results = rec.get("results", {})
        rows = []
        for short, unit in TRAINING_METRICS:
            val = results.get("training." + short)
            rows.append([short, f"{val:.4f}" if isinstance(val, float) else str(val), unit])
        log.info(
            "\n[%s | sweep %s]\n%s",
            mode,
            sweep_name,
            tabulate(rows, headers=["Metric", "Value", "Unit"], tablefmt="github"),
        )
        loss_rows = [[s["step"], f"{s['loss']:.6f}"] for s in rec.get("step_metrics", []) if "loss" in s]
        if loss_rows:
            log.info(
                "\nLoss Curve [%s]:\n%s", sweep_name, tabulate(loss_rows, headers=["Step", "Loss"], tablefmt="github")
            )


def print_results_table(training_res_dict, request):
    """Summarize all sweeps: console tables, single metric-results HTML, and a
    consolidated PASS/FAIL summary recorded via globals.error_list for the pytest
    final summary."""
    if not training_res_dict.get("sweeps"):
        log.info("training_res_dict empty, nothing to print")
        return

    _print_sweep_tables(training_res_dict)
    _write_metric_results_html(training_res_dict, request)

    failures = training_res_dict.get("metric_failures", [])
    globals.error_list = []
    for f in failures:
        fail_test(f)
    update_test_result()


def teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
