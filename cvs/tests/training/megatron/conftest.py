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
from cvs.lib.training.megatron.training_config_loader import load_training_variant

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
    """Cross-test state for the per-combo lifecycle model.

    Each combo's container launch, training, and teardown are timed sub-stages
    of test_training. `report` maps each nodeid to its recorded (label, value,
    unit) rows, which pytest_runtest_makereport renders into the HTML detail
    panel. `torn_down` suppresses the orch fixture leak-guard: test_training
    sets it True after its own teardown so the module-end finalizer does not
    tear down a second time.
    """

    def __init__(self):
        self.torn_down = False
        self.report = {}  # nodeid -> list[(label, value, unit)]

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def train_res_dict():
    return {}


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    """Construct a ContainerOrchestrator and own a final teardown safety net.

    Each combo's test_training launches and tears down its own container in a
    finally block, setting lifecycle.torn_down=True afterwards. This finalizer
    only fires when torn_down is False -- i.e. a combo crashed hard before its
    own teardown ran -- so nothing leaks past the module without double-tearing
    down in the normal case.
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
    """Pin test order: each combo's test_training (which owns the full container
    lifecycle) runs before any test_throughput, which only reads saved results."""
    rank = {
        "test_training": 0,
        "test_throughput": 1,
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
    rows = getattr(lc, "report", {}).get(item.nodeid) if lc else None
    if not rows:
        return
    try:
        import pytest_html
    except ImportError:
        return
    body = "".join(
        f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>"
        for label, value, unit in rows
    )
    html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(html))
    report.extras = extras


def pytest_html_results_table_header(cells):
    cells.insert(-1, "<th>Value</th>")
    cells.insert(-1, "<th>Unit</th>")


def pytest_html_results_table_row(report, cells):
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
