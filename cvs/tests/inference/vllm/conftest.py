'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os
from pathlib import Path

import pytest

try:
    from _pytest.subtests import SubtestReport as _BuiltinSubtestReport
except ImportError:
    _BuiltinSubtestReport = None
try:
    from pytest_subtests.plugin import SubTestReport as _PluginSubtestReport
except ImportError:
    _PluginSubtestReport = None

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.vllm_topology import resolve_vllm_topology, scope_vllm_cluster
from cvs.lib.inference.utils.vllm_config_loader import load_variant
from cvs.lib.inference.utils.vllm_parsing import VLLM_RESULTS_COLUMNS
from cvs.lib.report.benchmark_metric_registry import (
    benchmark_metric_columns_for_nodeid,
    benchmark_metric_rows_from_report,
    mark_collapsible_result_cell,
    patch_benchmark_metrics_into_html,
    stamp_benchmark_metric_rows_on_report,
)
from cvs.lib.report.render.perf_metric_table import (
    is_benchmark_metrics_extra,
    render_benchmark_metrics_html,
)
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.tests.inference.vllm._shared import validate_vllm_execution_mode

log = globals.log
VLLM_METRIC_VERIFICATION_TEST = 'test_verify_cell_metrics'


def _is_subtest_report(report) -> bool:
    if _BuiltinSubtestReport is not None and isinstance(report, _BuiltinSubtestReport):
        return True
    if _PluginSubtestReport is not None and isinstance(report, _PluginSubtestReport):
        return True
    return False


def _is_verification_report(report) -> bool:
    test_name = report.nodeid.rsplit('::', 1)[-1].split('[', 1)[0]
    return report.when == 'call' and test_name == VLLM_METRIC_VERIFICATION_TEST and not _is_subtest_report(report)


def _is_full_log_extra(extra: object) -> bool:
    return isinstance(extra, dict) and extra.get('format_type') == 'url' and extra.get('name') == 'Full Log'


def _attach_metric_panel(report, rows) -> None:
    try:
        import pytest_html
    except ImportError:
        return

    extras = []
    for extra in getattr(report, 'extras', []) or []:
        if _is_full_log_extra(extra) or is_benchmark_metrics_extra(extra):
            extras.append(extra)
    if not any(is_benchmark_metrics_extra(extra) for extra in extras):
        columns = benchmark_metric_columns_for_nodeid(report.nodeid) or VLLM_RESULTS_COLUMNS
        extras.append(pytest_html.extras.html(render_benchmark_metrics_html(rows, columns=columns)))
    report.extras = extras
    stamp_benchmark_metric_rows_on_report(report, rows)


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced).

    Protects cluster-set SCALAR and DICT container keys (e.g. shm_size, an env
    map) from being wiped by a top-level replace: they survive unless the variant
    overrides that same key. List keys (e.g. runtime.args, volume mounts) are
    REPLACED here, not unioned -- the cluster's list values are recombined with
    the variant's additively further downstream, in container.py's getters.
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
    return load_variant(config_file, cluster_dict)


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

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle, vllm_mode):
    """Construct a ContainerOrchestrator and own ONLY its teardown safety net.

    The actual launch/sshd happen in test_launch_container / test_setup_sshd
    so they appear as timed rows. This fixture builds the object and registers a
    leak-guard finalizer: if a mid-sweep test fails before test_teardown runs,
    the container is still torn down here. When test_teardown ran successfully
    it sets lifecycle.torn_down, so the finalizer no-ops (no double teardown).
    """
    # OrchestratorConfig.from_configs does a top-level dict.update, so a bare variant
    # container block would wipe the cluster file's container settings. Deep-merge the
    # variant ONTO the cluster block so cluster-set scalar/dict keys survive, with the
    # variant winning on conflicting keys. (List keys like runtime.args are replaced
    # here but recombined additively downstream in container.py's getters.)
    suite_cluster = scope_vllm_cluster(vllm_mode, cluster_dict)
    container_block = _deep_merge(
        suite_cluster.get("container", {}),
        variant_config.container.model_dump(),
    )
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
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
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        # vLLM configs always reference a mounted, pre-staged model.
        return ""
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def inf_res_dict():
    return {}


def pytest_collection_modifyitems(config, items):
    """Pin the lifecycle order explicitly instead of relying on definition order.

    `test_print_results_table` is an imported function (its source line points
    into _shared.py), so default ordering collects it FIRST -- which would log an
    empty table before any cell ran. Sort deterministically: launch, sshd, fetch,
    the benchmark cells, the results table, then teardown last. Items from other
    modules keep their relative order.
    """
    validate_vllm_execution_mode(config)
    rank = {
        "test_launch_container": 0,
        "test_setup_sshd": 1,
        "test_discover_topology": 2,
        "test_model_fetch": 3,
        "test_openai_compatible_smoke": 4,
        "test_vllm_inference": 5,
        VLLM_METRIC_VERIFICATION_TEST: 6,
        "test_accuracy_eval": 7,
        "test_print_results_table": 8,
        "test_teardown": 9,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    yield
    if _is_verification_report(report):
        rows = benchmark_metric_rows_from_report(report)
        if rows:
            _attach_metric_panel(report, rows)


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_html(report, data):
    if _is_verification_report(report) and benchmark_metric_rows_from_report(report):
        del data[:]


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_row(report, cells):
    if _is_verification_report(report) and benchmark_metric_rows_from_report(report):
        cells[0] = mark_collapsible_result_cell(str(cells[0]))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    htmlpath = getattr(session.config.option, 'htmlpath', None)
    if htmlpath:
        patch_benchmark_metrics_into_html(Path(htmlpath), benchmark_test_name=VLLM_METRIC_VERIFICATION_TEST)
