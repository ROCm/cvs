'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.dtni.config_loader import inferencemax_benchmark_model_name, load_inferencemax_suite_raw
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib.verify_lib import update_test_result

log = globals.log


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced)."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base and isinstance(base[k], dict) and isinstance(v, dict) else v
    return out


def _container_block_from_inference(inference_dict, benchmark_params_dict, model_name):
    """Build a ``container`` dict for :class:`OrchestratorConfig` from legacy InferenceMax suite JSON."""
    cc = inference_dict["container_config"]
    bp = benchmark_params_dict.get(model_name, {})
    image = bp.get("container_image", inference_dict["container_image"])
    volumes = [f"{h}:{c}" for h, c in cc["volume_dict"].items()]
    devices = list(cc.get("device_list", []))
    env = dict(cc.get("env_dict", {}))
    return {
        "lifetime": "per_run",
        "name": inference_dict["container_name"],
        "image": image,
        "env": env,
        "runtime": {
            "name": "docker",
            "args": {
                "volumes": volumes,
                "devices": devices,
                "network": "host",
                "ipc": "host",
                "privileged": True,
            },
        },
    }


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
def suite_raw(pytestconfig):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_inferencemax_suite_raw(config_file)


@pytest.fixture(scope="module")
def inference_dict(suite_raw, cluster_dict):
    cfg = suite_raw["config"]
    return resolve_test_config_placeholders(cfg, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_params_dict(suite_raw, cluster_dict):
    bp = suite_raw["benchmark_params"]
    return resolve_test_config_placeholders(bp, cluster_dict)


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    path = inference_dict["hf_token_file"]
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def model_name(suite_raw):
    return inferencemax_benchmark_model_name(suite_raw)


@pytest.fixture(scope="module")
def orch(cluster_dict, inference_dict, benchmark_params_dict, lifecycle, model_name):
    """Container orchestrator: launch/teardown and ``exec`` into the inference container (see vllm_single)."""
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        _container_block_from_inference(inference_dict, benchmark_params_dict, model_name),
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


def pytest_generate_tests(metafunc):
    """Single sweep cell derived from ``benchmark_params`` for collection-time parametrization."""
    if "seq_combo" not in metafunc.fixturenames or "concurrency" not in metafunc.fixturenames:
        return
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    raw = load_inferencemax_suite_raw(config_file)
    mname = inferencemax_benchmark_model_name(raw)
    bp = raw.get("benchmark_params", {}).get(mname, {})
    combo = {
        "isl": str(bp.get("input_sequence_length", "8192")),
        "osl": str(bp.get("output_sequence_length", "1024")),
        "name": "legacy_profile",
    }
    conc = int(bp.get("max_concurrency", "64"))
    cid = f"{combo['name']}-conc{conc}"
    metafunc.parametrize("seq_combo,concurrency", [(combo, conc)], ids=[cid])


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
