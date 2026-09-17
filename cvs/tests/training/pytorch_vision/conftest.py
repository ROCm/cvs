"""Fixtures and sweep parametrization for the PyTorch Vision training suites.

Shared by ``pytorch_vision_single`` and ``pytorch_vision_distributed``.
"""

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.training.pytorch_vision.utils.config_loader import (
    load_vision_variant,
    validate_sweep_selector,
)
from cvs.lib.training.pytorch_vision.utils.metrics import METRICS
from cvs.lib.utils_lib import resolve_cluster_config_placeholders


log = globals.log


def _deep_merge(base, override):
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(base[key], value) if key in base else value
    return merged


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file) as stream:
        return resolve_cluster_config_placeholders(json.load(stream))


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_vision_variant(config_file, cluster_dict)


@pytest.fixture(scope="module", autouse=True)
def _guard_topology(pytestconfig, cluster_dict, variant_config):
    """Fail fast when the suite, the config, and the cluster file disagree.

    A distributed config launched under the single-node suite would silently
    train on one node and report it as a scaled result, so the mismatch is a
    hard failure rather than a warning.
    """
    if not variant_config.training.enabled:
        pytest.skip(
            f"protected {variant_config.training.run_mode} profile is disabled; set training.enabled=true explicitly"
        )

    suite = getattr(pytestconfig, "_suite_name", "") or ""
    suite_is_distributed = suite.endswith("_distributed")
    config_is_distributed = variant_config.training.distributed
    if suite_is_distributed != config_is_distributed:
        wanted = "true" if suite_is_distributed else "false"
        pytest.fail(
            f"{suite or 'this suite'} requires training.distributed={wanted}, "
            f"config has {str(config_is_distributed).lower()}"
        )

    nodes = list((cluster_dict.get("node_dict") or {}).keys())
    if config_is_distributed:
        if len(nodes) < 2:
            pytest.fail(f"pytorch_vision_distributed requires two or more cluster nodes, received {nodes}")
    elif len(nodes) != 1:
        pytest.fail(f"pytorch_vision_single requires exactly one cluster node, received {nodes}")


class Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}
        self.artifacts = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))

    def add_artifact(self, nodeid, name, rel_path, abs_path):
        """Register a per-test report artifact (e.g. loss-curve PNG) for linking."""
        self.artifacts.setdefault(nodeid, []).append((name, rel_path, abs_path))


@pytest.fixture(scope="module")
def lifecycle():
    return Lifecycle()


@pytest.fixture(scope="module")
def training_results():
    return {}


@pytest.fixture(scope="module")
def loss_series():
    """Ordered (step, loss) pairs per sweep, used to render loss-curve PNGs."""
    return {}


@pytest.fixture(scope="module")
def metric_rows():
    """Accumulated metric verdicts rendered into the shared metric-results page."""
    return []


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    container = _deep_merge(
        cluster_dict.get("container", {}),
        variant_config.container.model_dump(),
    )
    config = OrchestratorConfig.from_configs(
        cluster_dict,
        {"orchestrator": "container", "container": container},
    )
    orchestrator = OrchestratorFactory.create_orchestrator(log, config)
    yield orchestrator
    if not lifecycle.torn_down:
        log.info("PyTorch Vision leak guard: tearing down container")
        orchestrator.teardown_containers()


def pytest_collection_modifyitems(items):
    order = {
        "test_launch_container": 0,
        "test_verify_environment": 1,
        "test_real_data_smoke": 2,
        "test_training": 3,
        "test_rocal_overhead_comparisons": 4,
        "test_metric": 5,
        "test_loss_curve": 6,
        "test_convergence": 7,
        "test_print_results_table": 8,
        "test_teardown": 9,
    }
    items.sort(key=lambda item: order.get(item.originalname or item.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    current = item.funcargs.get("lifecycle")
    rows = getattr(current, "report", {}).get(item.nodeid) if current else None
    artifacts = getattr(current, "artifacts", {}).get(item.nodeid) if current else None

    # Every metric row also links to the one shared metric-results page written
    # by test_print_results_table, alongside its own per-test log link.
    metric_link = None
    if (item.originalname or "") == "test_metric":
        mgr = getattr(item.config, "_html_report_manager", None)
        if mgr is not None and getattr(mgr, "is_enabled", False):
            metric_link = f"{mgr._test_html_dir}/metric_results.html"

    if not rows and not artifacts and not metric_link:
        return
    try:
        import pytest_html
    except ImportError:
        return

    extras = getattr(report, "extras", [])

    if metric_link:
        extras.append(pytest_html.extras.url(metric_link, name="Metric Results"))

    if rows:
        body = "".join(f"<tr><td>{label}</td><td>{value:.3f}</td><td>{unit}</td></tr>" for label, value, unit in rows)
        extras.append(
            pytest_html.extras.html(f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>")
        )

    for name, rel_path, abs_path in artifacts or []:
        extras.append(pytest_html.extras.url(rel_path, name=name))
        try:
            import base64

            with open(abs_path, "rb") as fp:
                b64 = base64.b64encode(fp.read()).decode("ascii")
            extras.append(pytest_html.extras.png(b64, name=name))
        except Exception:  # noqa: BLE001 - a missing thumbnail must not fail the row
            pass

    report.extras = extras


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as stream:
        raw = json.load(stream)

    training = raw.get("training", {})
    sweeps = training.get("sweeps", [])
    by_name = {sweep["name"]: sweep for sweep in sweeps}
    enabled = training.get("enabled_sweep_list") or list(by_name)
    validate_sweep_selector(
        by_name.keys(),
        enabled,
        [sweep["label"] for sweep in sweeps],
    )

    if "metric" in metafunc.fixturenames:
        cases = [(sweep_name, metric) for sweep_name in enabled for metric, _unit in METRICS]
        metafunc.parametrize(
            "sweep_name,metric",
            cases,
            ids=[f"{by_name[sweep_name]['label']}-{metric}" for sweep_name, metric in cases],
        )
    elif "sweep_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "sweep_name",
            enabled,
            ids=[by_name[sweep_name]["label"] for sweep_name in enabled],
        )
