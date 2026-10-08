'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os
from pathlib import Path

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.report.benchmark_metric_registry import (
    benchmark_metric_columns_for_nodeid,
    benchmark_metric_rows_from_item,
    benchmark_metric_rows_from_report,
    mark_collapsible_result_cell,
    patch_benchmark_metrics_into_html,
    stamp_benchmark_metric_rows_on_report,
)
from cvs.lib.report.render.perf_metric_table import render_benchmark_metrics_html
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.lib.training.megatron.utils.training_config_loader import load_training_variant

# test_metric records one pass/fail verdict row per metric; these hooks render
# them as an expandable per-metric panel under the parent test row and inject a
# subtest summary count, mirroring the inference benchmark suites.
_METRIC_TEST_NAME = "test_metric"

log = globals.log


def pytest_generate_tests(metafunc):
    """Parametrize per-sweep tests for both suites from sweep.runs.

    Tests that take sweep_name get one row per combination key listed in
    sweep.runs (must exist in sweep.combinations). No cartesian product.
    The pytest ID is the combination key so it matches the threshold cell.
    """
    if "sweep_name" not in metafunc.fixturenames:
        return
    names = []
    combinations = {}
    config_file = metafunc.config.getoption("config_file")
    if config_file and os.path.isfile(config_file):
        with open(config_file) as fp:
            raw = json.load(fp)

        sweep = raw.get("sweep") or {}
        combinations = sweep.get("combinations") or {}
        runs = sweep.get("runs", list(combinations.keys()))
        for run_id in runs:
            if run_id not in combinations:
                log.warning("sweep.runs entry '%s' not found in sweep.combinations; skipping", run_id)
                continue
            names.append(run_id)
    if not names and not combinations:
        names = ["default"]
    if names:
        metafunc.parametrize("sweep_name", names, ids=names)


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced).

    Protects cluster-set scalar and dict container keys from being wiped by a
    top-level replace: they survive unless the training block overrides that same
    key. List keys (e.g. runtime.args, volumes) are replaced here and recombined
    additively downstream in container.py's getters.
    """
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file) as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_training_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model.

    The container is launched once (test_launch_container), all sweep combos
    run inside it (test_training), GPU memory is freed between combos via
    stop_training_processes(), and the container is torn down once at the end
    (test_teardown). `failed` lets a broken stage skip the rest. `torn_down`
    suppresses the orch fixture leak-guard when test_teardown already ran.
    `report` maps each nodeid to its recorded (label, value, unit) rows.
    """

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}  # nodeid -> list[(label, value, unit)]
        self.artifacts = {}  # nodeid -> list[(link_name, rel_path)]

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))

    def add_artifact(self, nodeid, link_name, rel_path, abs_path=None):
        self.artifacts.setdefault(nodeid, []).append((link_name, rel_path))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def train_res_dict():
    return {}


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    """Construct a ContainerOrchestrator and own a final teardown safety net.

    The container is launched once in test_launch_container and torn down once
    in test_teardown, which sets lifecycle.torn_down=True. This finalizer only
    fires when torn_down is False -- i.e. test_teardown did not run (e.g. a
    crash before teardown) -- so nothing leaks past the module without
    double-tearing down in the normal case.
    """
    container_block = _deep_merge(cluster_dict.get("container", {}), variant_config.container.model_dump())
    env = dict(container_block.get("env") or {})
    node_dict = cluster_dict.get("node_dict") or {}
    env["NNODES"] = str(len(node_dict))
    container_block["env"] = env
    testsuite_config = {"orchestrator": "container", "container": container_block}
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    try:
        if not lifecycle.torn_down:
            log.info("orch fixture leak-guard: tearing down container (per-combo teardown did not run)")
            o.teardown_containers()
    finally:
        o.close()


def pytest_collection_modifyitems(items):
    """Pin lifecycle order: launch → training combos → metric → teardown."""
    rank = {
        "test_launch_container": 0,
        "test_download_tokenizer": 1,
        "test_smoke": 2,
        "test_checkpoint": 3,
        "test_training": 4,
        "test_metric": 5,
        "test_loss_curve": 6,
        "test_print_results_table": 7,
        "test_teardown": 8,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


def _nodeid_test_name(nodeid):
    return nodeid.rsplit("::", 1)[-1].split("[", 1)[0]


def _is_full_log_extra(extra):
    return isinstance(extra, dict) and extra.get("format_type") == "url" and extra.get("name") == "Full Log"


def _is_url_extra(extra):
    return isinstance(extra, dict) and extra.get("format_type") == "url"


def _attach_metric_artifact_extras(item, report):
    """Add this combo's results-table (and any lifecycle) artifact links to the test_metric row."""
    lc = item.funcargs.get("lifecycle")
    artifacts = getattr(lc, "artifacts", {}).get(item.nodeid) if lc else None
    if not artifacts:
        return
    try:
        import pytest_html
    except ImportError:
        return
    extras = list(getattr(report, "extras", []) or [])
    existing = {(e.get("name"), e.get("content")) for e in extras if isinstance(e, dict)}
    for link_name, rel_path in artifacts:
        if (link_name, rel_path) in existing:
            continue
        extras.append(pytest_html.extras.url(rel_path, name=link_name))
    report.extras = extras


def _attach_lifecycle_extras(item, report):
    """Attach recorded stage-timing rows and artifact links to a test's report."""
    lc = item.funcargs.get("lifecycle")
    if not lc:
        return
    rows = getattr(lc, "report", {}).get(item.nodeid)
    artifacts = getattr(lc, "artifacts", {}).get(item.nodeid)
    if not rows and not artifacts and not report.failed:
        return
    try:
        import pytest_html
    except ImportError:
        return
    extras = getattr(report, "extras", [])
    if rows:
        body = "".join(f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>" for label, value, unit in rows)
        html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
        extras.append(pytest_html.extras.html(html))
    if artifacts:
        for link_name, rel_path in artifacts:
            extras.append(pytest_html.extras.url(rel_path, name=link_name))
    report.extras = extras


def _attach_metric_verdict_extras_for_nodeid(report, nodeid, rows):
    """Keep the Full Log and results-table links and add the collapsible per-metric panel."""
    if not rows:
        return
    try:
        import pytest_html
    except ImportError:
        return
    extras = [extra for extra in getattr(report, "extras", []) or [] if _is_url_extra(extra)]
    columns = benchmark_metric_columns_for_nodeid(nodeid)
    extras.append(pytest_html.extras.html(render_benchmark_metrics_html(rows, columns=columns)))
    report.extras = extras
    stamp_benchmark_metric_rows_on_report(report, rows)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Attach stage-timing rows (most tests) or the per-metric panel (test_metric)."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    if _nodeid_test_name(report.nodeid) == _METRIC_TEST_NAME:
        _attach_metric_artifact_extras(item, report)
        _attach_metric_verdict_extras_for_nodeid(report, item.nodeid, benchmark_metric_rows_from_item(item))
        return
    _attach_lifecycle_extras(item, report)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    """Re-attach the per-metric panel after pytest-html stores the call report."""
    yield
    if report.when != "call" or _nodeid_test_name(report.nodeid) != _METRIC_TEST_NAME:
        return
    rows = benchmark_metric_rows_from_report(report)
    if rows:
        _attach_metric_verdict_extras_for_nodeid(report, report.nodeid, rows)


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_html(report, data):
    """Drop the inline log for test_metric rows; the panel replaces it."""
    if _nodeid_test_name(report.nodeid) != _METRIC_TEST_NAME:
        return
    if benchmark_metric_rows_from_report(report):
        del data[:]


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_row(report, cells):
    """Mark test_metric result cells collapsible so the metric panel expands."""
    if report.when != "call" or _nodeid_test_name(report.nodeid) != _METRIC_TEST_NAME:
        return
    if benchmark_metric_rows_from_report(report):
        cells[0] = mark_collapsible_result_cell(str(cells[0]))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Patch the written pytest-html so metric rows expose the collapsible panel
    and the filter bar shows the per-metric subtest counts."""
    yield
    htmlpath = getattr(session.config.option, "htmlpath", None)
    if htmlpath:
        patch_benchmark_metrics_into_html(Path(htmlpath), benchmark_test_name=_METRIC_TEST_NAME)


# def pytest_html_results_table_header(cells):
#     cells.insert(-1, "<th>Value</th>")
#     cells.insert(-1, "<th>Unit</th>")


# def pytest_html_results_table_row(report, cells):
#     if not hasattr(report, 'user_properties'):
#         return
#     props = dict(report.user_properties)
#     has = "metric_value" in props
#     val = props.get("metric_value")
#     unit = props.get("metric_unit", "") if has else ""
#     if not has:
#         shown = ""
#     elif val is None:
#         shown = "-"
#     elif isinstance(val, float):
#         shown = f"{val:.3f}"
#     else:
#         shown = str(val)
#     cells.insert(-1, f"<td>{shown}</td>")
#     cells.insert(-1, f"<td>{unit}</td>")
