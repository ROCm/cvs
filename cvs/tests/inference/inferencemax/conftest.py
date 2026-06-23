'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.utils.inferencemax_config_loader import (
    benchmark_model_key,
    legacy_benchmark_params_from_variant,
    legacy_inference_dict_from_variant,
    load_variant,
    orchestrator_container_from_variant,
)
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
)

log = globals.log


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced)."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


class _Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


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


@pytest.fixture(scope="module")
def inference_dict(variant_config):
    """Legacy ``config`` block for :class:`InferenceMaxJob` (Phase 3 will drop this)."""
    return legacy_inference_dict_from_variant(variant_config)


@pytest.fixture(scope="module")
def benchmark_params_dict(variant_config):
    """Legacy ``benchmark_params`` block for :class:`InferenceMaxJob` (Phase 3 will drop this)."""
    return legacy_benchmark_params_from_variant(variant_config)


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def model_name(variant_config):
    return benchmark_model_key(variant_config)


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    """Container orchestrator: launch/teardown and ``exec`` into the inference container."""
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        orchestrator_container_from_variant(variant_config),
    )
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
    }
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down InferenceMax containers")
        o.teardown_containers()


@pytest.fixture(scope="module")
def gpu_type(orch):
    head_node = orch.head.host_list[0]
    smi_out_dict = orch.head.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    return get_model_from_rocm_smi_output(smi_out)


@pytest.fixture(scope="session")
def inf_res_dict():
    return {}


def pytest_collection_modifyitems(items):
    rank = {
        "test_launch_container": 0,
        "test_inferencemax_inference": 1,
        "test_print_results_table": 2,
        "test_teardown": 3,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
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
