'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Reusable **lifecycle-as-tests** helpers for DTNI inference suites.

``inferencex_atom_single`` imports the stage tests from here today; other suites
(``vllm_single``, future IX parity frameworks) can reuse the same module instead
of copying launch / sshd / model-fetch / teardown blocks.

**Suite module** — import stage tests so pytest collects them::

    from cvs.lib.inference.utils.inference_suite_lifecycle import (
        test_accuracy_eval,
        test_launch_container,
        test_model_fetch,
        test_setup_sshd,
        test_teardown,
    )

**conftest.py** — wire shared fixtures and HTML hooks::

    from cvs.lib.inference.utils.inference_suite_lifecycle import (
        InferenceLifecycle,
        attach_lifecycle_html_table,
        html_metric_table_header,
        html_metric_table_row,
        sort_lifecycle_items,
    )

Also provides ``sweep_cell_result_key``; see :mod:`cvs.lib.inference.utils.cache_probe` for ``du_bytes``.

Optional HTML/JSON suite report: add ``cvs/lib/report/presets/<cvs_run_stem>.py`` (see
``cvs/lib/report/README.md``); root ``cvs/conftest.py`` auto-wires hooks when ``--html`` is set.
'''

from __future__ import annotations

import shlex
import time

import pytest

try:
    import pytest_html
except ImportError:
    pytest_html = None

from cvs.lib import globals
from cvs.lib.inference.utils.cache_probe import du_bytes
from cvs.lib.inference.utils.lm_eval_job import run_accuracy_tasks
from cvs.lib.utils.verdict import evaluate_all

log = globals.log

FETCH_POLL_COUNT = 80
FETCH_POLL_WAIT_S = 30
FETCH_PRESENCE_RETRIES = 5


class InferenceLifecycle:
    """Cross-test state shared by lifecycle stage tests in one module scope."""

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


def sweep_cell_result_key(variant_config, seq_combo, isl, osl, concurrency):
    """Canonical ``inf_res_dict`` key for one sweep cell."""
    return (
        variant_config.model.id,
        variant_config.gpu_arch,
        isl,
        osl,
        seq_combo.get("name", "default"),
        concurrency,
    )


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container."""
    t = time.monotonic()
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


def test_setup_sshd(orch, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    t = time.monotonic()
    ok = orch.setup_sshd()
    lifecycle.record(request.node.nodeid, "sshd_setup", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail("setup_sshd() returned False")
    if len(orch.hosts) > 1:
        probe = orch.exec("bash -c 'ss -ltn 2>/dev/null | grep -q :2224 && echo OK || echo NO'")
        if not any("OK" in (v or "") for v in (probe or {}).values()):
            lifecycle.failed = True
            pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_model_fetch(orch, variant_config, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    models_dir = variant_config.paths.models_dir
    if not models_dir:
        pytest.skip("paths.models_dir unset; cannot locate/verify the HF cache")

    remote = getattr(variant_config.model, "remote", 0)
    t = time.monotonic()
    orch.exec(f"mkdir -p {shlex.quote(models_dir)}")

    if not remote:
        final = 0
        for it in range(FETCH_PRESENCE_RETRIES):
            cur = du_bytes(orch, models_dir)
            if cur is None:
                lifecycle.failed = True
                pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
            final = cur
            log.info("[fetch presence %d] size=%.1fGB", it, final / 1e9)
            if final > 0:
                break
            time.sleep(FETCH_POLL_WAIT_S)
    else:
        fetch = (
            f"HF_HUB_CACHE={shlex.quote(models_dir)} "
            f"nohup hf download {shlex.quote(variant_config.model.id)} "
            f"> /tmp/hf_fetch.log 2>&1 &"
        )
        orch.exec("bash -c " + shlex.quote(fetch))
        prev = -1
        stable = 0
        final = du_bytes(orch, models_dir)
        if final is None:
            lifecycle.failed = True
            pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
        for it in range(FETCH_POLL_COUNT):
            cur = du_bytes(orch, models_dir)
            if cur is None:
                lifecycle.failed = True
                pytest.fail(f"could not measure model cache size under {models_dir} (du error)")
            final = cur
            log.info("[fetch poll %d] size=%.1fGB", it, cur / 1e9)
            if cur > 0 and cur == prev:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = cur
            time.sleep(FETCH_POLL_WAIT_S)

    lifecycle.record(request.node.nodeid, "model_fetch", time.monotonic() - t)
    lifecycle.record(request.node.nodeid, "model_size", final / 1e9, "GB")
    if final <= 0:
        lifecycle.failed = True
        pytest.fail(f"no model bytes under {models_dir} after fetch")


def test_accuracy_eval(orch, variant_config, lifecycle, request):
    """Opt-in stage: run lm-eval-harness accuracy tasks against the already-live server.

    Selection lives in config.json's `accuracy.tasks` (an AccuracyConfig); an
    absent block or empty `tasks: []` means this suite run has no accuracy
    tasks configured, so the stage is skipped, not failed -- same convention
    as a perf metric with no threshold entry. Gating values live in the
    sibling threshold.json's `accuracy` block, keyed by task id, and are
    joined against only the tasks that actually ran (see
    cvs/lib/inference/utils/AGENTS.md for the full design).
    """
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    accuracy_config = getattr(variant_config, "accuracy", None)
    tasks = accuracy_config.tasks if accuracy_config else []
    if not tasks:
        pytest.skip("no accuracy tasks configured (accuracy.tasks empty or absent)")

    params = variant_config.params
    output_dir = f"{variant_config.paths.log_dir}/accuracy"

    t = time.monotonic()
    try:
        actuals_by_id = run_accuracy_tasks(
            orch=orch,
            tasks=tasks,
            base_url=f"{params.base_url}:{params.port_no}",
            model_id=variant_config.model.id,
            model_path=variant_config.model.id,
            output_dir=output_dir,
        )
    except RuntimeError as e:
        lifecycle.failed = True
        lifecycle.record(request.node.nodeid, "accuracy_eval", time.monotonic() - t)
        pytest.fail(str(e))
    lifecycle.record(request.node.nodeid, "accuracy_eval", time.monotonic() - t)

    active_ids = {task.id for task in tasks}
    accuracy_thresholds = (variant_config.thresholds or {}).get("accuracy", {})
    for task_id, actual in actuals_by_id.items():
        if task_id not in active_ids:
            continue
        for metric_key, value in actual.items():
            lifecycle.record(request.node.nodeid, f"{task_id}.{metric_key}", value, "")
        if not variant_config.enforce_thresholds:
            continue
        evaluate_all(actual, accuracy_thresholds.get(task_id, {}))


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True


def sort_lifecycle_items(items, rank):
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


def attach_lifecycle_html_table(item, report):
    if report.when != "call":
        return
    lc = item.funcargs.get("lifecycle")
    rows = getattr(lc, "report", {}).get(item.nodeid) if lc else None
    if not rows:
        return
    if pytest_html is None:
        return
    body = "".join(f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>" for label, value, unit in rows)
    html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(html))
    report.extras = extras


def html_metric_table_header(cells):
    cells.insert(-1, "<th>Value</th>")
    cells.insert(-1, "<th>Unit</th>")


def html_metric_table_row(report, cells):
    props = dict(report.user_properties)
    has = "metric_value" in props
    val = props.get("metric_value")
    unit = props.get("metric_unit", "") if has else ""
    if not has:
        shown = ""
    elif val is None:
        shown = "-"
    elif isinstance(val, float):
        shown = f"{val:.3f}"
    else:
        shown = str(val)
    cells.insert(-1, f"<td>{shown}</td>")
    cells.insert(-1, f"<td>{unit}</td>")
