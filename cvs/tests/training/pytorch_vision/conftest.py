"""Fixtures for the PyTorch Vision training lifecycle."""

import json

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.training.pytorch_vision.utils.config_loader import load_vision_variant
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


class Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


@pytest.fixture(scope="module")
def lifecycle():
    return Lifecycle()


@pytest.fixture(scope="module")
def training_results():
    return {}


@pytest.fixture(scope="module")
def inf_res_dict():
    """Report-engine result map, shaped like the SGLang/vLLM run-deck contract."""
    return {}


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
        "test_training": 2,
        "test_metric": 3,
        "test_print_results_table": 4,
        "test_teardown": 5,
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
    if not rows:
        return
    try:
        import pytest_html
    except ImportError:
        return
    body = "".join(f"<tr><td>{label}</td><td>{value:.3f}</td><td>{unit}</td></tr>" for label, value, unit in rows)
    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"))
    report.extras = extras
