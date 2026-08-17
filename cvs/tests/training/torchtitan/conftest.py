'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.lib.training.torchtitan.utils.training_config_loader import load_training_variant

log = globals.log


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
    path = variant_config.config['hf_token_file']
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
    testsuite_config = {"orchestrator": "container", "container": container_block}
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down container (per-combo teardown did not run)")
        o.teardown_containers()


def pytest_collection_modifyitems(items):
    """Pin lifecycle order: launch → training combos → metric → teardown."""
    rank = {
        "test_launch_container": 0,
        "test_download_tokenizer": 1,
        "test_setup_rdma": 1,
        "test_smoke": 2,
        "test_training": 3,
        "test_metric": 4,
        "test_loss_curve": 5,
        "test_teardown": 6,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach this test's recorded timing rows to its HTML report detail panel."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
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
    if report.failed:
        props = dict(item.user_properties)
        log_tail = props.get("training_log_tail")
        if log_tail:
            extras.append(pytest_html.extras.text(log_tail, name="Training Log (tail)"))
    report.extras = extras
