'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from pathlib import Path

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.vllm_topology import resolve_vllm_topology, scope_vllm_cluster
from cvs.lib.inference.utils.vllm_metrics import VLLM_RESULTS_COLUMNS
from cvs.lib.conf_lib.loader import load_config
from cvs.lib.conf_lib.schemas.vllm import VariantConfig
from cvs.lib.report.benchmark_metric_registry import (
    benchmark_metric_columns_for_nodeid,
    benchmark_metric_rows_from_item,
    benchmark_metric_rows_from_report,
    mark_collapsible_result_cell,
    patch_benchmark_metrics_into_html,
    stamp_benchmark_metric_rows_on_report,
)
from cvs.lib.report.render.perf_metric_table import (
    is_benchmark_metrics_extra,
    render_benchmark_metrics_html,
)
from cvs.lib.report.subtest_reports import called_from_subtest_context, is_subtest_report
from cvs.tests.inference.vllm._shared import validate_vllm_execution_mode

log = globals.log
VLLM_INFERENCE_TEST = 'test_vllm_inference'


def _is_inference_report(report) -> bool:
    test_name = report.nodeid.rsplit('::', 1)[-1].split('[', 1)[0]
    return report.when == 'call' and test_name == VLLM_INFERENCE_TEST and not is_subtest_report(report)


def _is_full_log_extra(extra: object) -> bool:
    return isinstance(extra, dict) and extra.get('format_type') == 'url' and extra.get('name') == 'Full Log'


def _attach_metric_panel(report, rows) -> None:
    if not rows:
        return

    try:
        from pytest_html import extras as pytest_html_extras
    except ImportError:
        return

    extras = []
    has_full_log = False
    has_metric_table = False
    for extra in getattr(report, 'extras', []) or []:
        if _is_full_log_extra(extra) and not has_full_log:
            extras.append(extra)
            has_full_log = True
        elif is_benchmark_metrics_extra(extra) and not has_metric_table:
            extras.append(extra)
            has_metric_table = True
    if not has_metric_table:
        columns = benchmark_metric_columns_for_nodeid(report.nodeid) or VLLM_RESULTS_COLUMNS
        extras.append(pytest_html_extras.html(render_benchmark_metrics_html(rows, columns=columns)))
    report.extras = extras
    stamp_benchmark_metric_rows_on_report(report, rows)


@pytest.fixture(scope="module")
def loaded_config(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_config(VariantConfig, config_file, cluster_file)


@pytest.fixture(scope="module")
def variant_config(loaded_config):
    return loaded_config.config


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model.

    The container launch / sshd / fetch / teardown stages are individual tests
    (so each is a timed, pass/fail row in the HTML) rather than fixture body
    code. They share this object: `failed` lets a broken stage skip the rest
    instead of cascading; `torn_down` lets the explicit teardown test suppress
    the fixture's leak-guard finalizer; `report` maps a test's nodeid to the
    rows it recorded, each carrying its own unit, so attach_inference_suite_lifecycle_table
    renders only that test's stages -- not every stage on every row.
    """

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}  # nodeid -> list[(label, value, unit)]
        self.live_server_sig = None
        self.live_server_job = None
        self.model_load_s = None
        self.model_load_memory_mb = None

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def orch(loaded_config, lifecycle, vllm_mode):
    """Construct a ContainerOrchestrator and own ONLY its teardown safety net.

    The actual launch/sshd happen in test_launch_container / test_setup_sshd
    so they appear as timed rows. This fixture builds the object and registers a
    leak-guard finalizer: if a mid-sweep test fails before test_teardown runs,
    the container is still torn down here. When test_teardown ran successfully
    it sets lifecycle.torn_down, so the finalizer no-ops (no double teardown).
    """
    # from_configs replaces top-level keys, so the container block passed here must
    # already hold the cluster's container settings; load_config merged them.
    suite_cluster = scope_vllm_cluster(vllm_mode, loaded_config.cluster)
    testsuite_config = {
        "orchestrator": "container",
        "container": loaded_config.container,
    }
    cfg = OrchestratorConfig.from_configs(suite_cluster, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    try:
        if not lifecycle.torn_down:
            log.info("orch fixture leak-guard: tearing down container (explicit teardown did not run)")
            o.teardown_containers()
    finally:
        o.close()


@pytest.fixture(scope="module")
def vllm_mode(request):
    stem = request.module.__name__.rsplit(".", 1)[-1]
    if stem == "vllm_single":
        return "single"
    if stem == "vllm_distributed":
        return "distributed"
    pytest.fail(f"vLLM suite must be vllm_single or vllm_distributed, got {stem!r}")


@pytest.fixture(scope="module")
def vllm_targets(orch, variant_config, vllm_mode):
    try:
        topology = resolve_vllm_topology(vllm_mode, variant_config, orch.hosts)
    except ValueError as exc:
        pytest.fail(str(exc))

    variant_config.bind_effective_topology(topology)
    return topology.target_groups


@pytest.fixture(scope="module")
def cvs_results_dict():
    return {}


@pytest.fixture(scope="module")
def inf_res_dict(cvs_results_dict):
    """Same dict as ``cvs_results_dict`` so existing cell tests fill the report store."""
    return cvs_results_dict


def pytest_collection_modifyitems(config, items):
    """Pin the lifecycle order explicitly instead of relying on definition order.

    Sort deterministically: launch, sshd, fetch, the benchmark cells, the
    results table, then teardown last. Items from other modules keep their
    relative order.
    """
    validate_vllm_execution_mode(config)
    rank = {
        "test_launch_container": 0,
        "test_setup_sshd": 1,
        "test_discover_topology": 2,
        "test_model_fetch": 3,
        "test_openai_compatible_smoke": 4,
        VLLM_INFERENCE_TEST: 5,
        "test_accuracy_eval": 6,
        "test_print_results_table": 7,
        "test_teardown": 8,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Attach metric rows before pytest-html consumes the parent call report."""
    outcome = yield
    report = outcome.get_result()
    if called_from_subtest_context() or not _is_inference_report(report):
        return
    _attach_metric_panel(report, benchmark_metric_rows_from_item(item))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    yield
    if _is_inference_report(report):
        rows = benchmark_metric_rows_from_report(report)
        if rows:
            _attach_metric_panel(report, rows)


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_html(report, data):
    if _is_inference_report(report) and benchmark_metric_rows_from_report(report):
        del data[:]


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_row(report, cells):
    if _is_inference_report(report) and benchmark_metric_rows_from_report(report):
        cells[0] = mark_collapsible_result_cell(str(cells[0]))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    htmlpath = getattr(session.config.option, 'htmlpath', None)
    if htmlpath:
        patch_benchmark_metrics_into_html(Path(htmlpath), benchmark_test_name=VLLM_INFERENCE_TEST)
