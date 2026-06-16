'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

"""
InferenceMax suite scaffolding — DTNI-style layout (lifecycle hooks, staged tests,
``orch`` leak-guard, ``pytest_generate_tests`` / collection ordering) aligned with
``cvs/tests/inference/vllm/conftest.py`` as a reference implementation.

Differences (documented):
  * Workloads still use **host SSH + docker run/exec** (``InferenceMaxJob`` in
    ``cvs.lib.inference.inferencemax_orch``), not
    ``ContainerOrchestrator.exec`` inside an sshd container. The ``orch`` fixture
    therefore yields an ``InferenceMaxHostContext`` with the same **stage method
    names** as a real orchestrator so tests stay structurally parallel to DTNI
    container suites.
  * Optional sibling ``*threshold.json`` (same ``glob`` discovery as ``load_variant`` in
    ``cvs.lib.dtni.config_loader``); merged by ``load_inferencemax_suite_raw``.
  * Optional **host-mounted** vLLM server scripts live under
    ``cvs/input/config_file/inference/inferencemax_single/mi300x_gpt_oss_120b_single/benchmark_server_scripts/``;
    see ``docs/reference/configuration-files/inferencemax.rst`` for deploy steps and
    ``use_host_mounted_server_script`` in the inference JSON.
  * No ``test_ab_setup_sshd`` / ``test_ac_model_fetch`` rows (not applicable).
"""

import json
import os
import re
import time

import pytest

from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.lib import docker_lib, globals
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib.dtni.config_loader import inferencemax_benchmark_model_name, load_inferencemax_suite_raw
from cvs.lib.verify_lib import update_test_result

log = globals.log


class _Lifecycle:
    """Shared with vLLM pattern: gate skips + teardown-once + per-stage HTML rows."""

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


class InferenceMaxHostContext:
    """Host-orchestrated Docker lifecycle with an orchestrator-like surface for tests."""

    def __init__(self, cluster_dict, inference_dict, benchmark_params_dict, model_name):
        self.cluster_dict = cluster_dict
        self.inference_dict = inference_dict
        self.benchmark_params_dict = benchmark_params_dict
        self.model_name = model_name
        env_vars = cluster_dict.get("env_vars")
        node_list = list(cluster_dict["node_dict"].keys())
        user = cluster_dict["username"]
        pkey = cluster_dict["priv_key_file"]
        self.s_phdl = Pssh(log, node_list, user=user, pkey=pkey, env_vars=env_vars)
        self.c_phdl = Pssh(log, node_list, user=user, pkey=pkey, env_vars=env_vars)
        bp = benchmark_params_dict.get(model_name, {})
        image = bp.get("container_image", inference_dict["container_image"])
        self.container_config = {
            "name": inference_dict["container_name"],
            "image": image,
        }

    def get_container_name(self, cfg, image):
        return ContainerOrchestrator.get_container_name(cfg, image)

    def setup_containers(self):
        globals.error_list = []
        name = self.inference_dict["container_name"]
        docker_lib.kill_docker_container(self.s_phdl, name)
        docker_lib.delete_all_containers_and_volumes(self.s_phdl)

        bp = self.benchmark_params_dict.get(self.model_name, {})
        container_image = bp.get("container_image", self.inference_dict["container_image"])
        shm = self.inference_dict.get("shm_size", "48G")

        docker_lib.launch_docker_container(
            self.s_phdl,
            name,
            container_image,
            self.inference_dict["container_config"]["device_list"],
            self.inference_dict["container_config"]["volume_dict"],
            self.inference_dict["container_config"]["env_dict"],
            shm_size=shm,
            timeout=60 * 20,
        )

        time.sleep(30)
        out_dict = self.s_phdl.exec("docker ps")
        for node in out_dict.keys():
            if not re.search(re.escape(name), out_dict[node], re.I):
                return False
        update_test_result()
        return True

    def verify_containers_running(self, name):
        out_dict = self.s_phdl.exec("docker ps")
        for node in out_dict.keys():
            if not re.search(re.escape(name), out_dict[node], re.I):
                return False
        return True

    def teardown_containers(self):
        name = self.inference_dict["container_name"]
        docker_lib.kill_docker_container(self.s_phdl, name)
        docker_lib.delete_all_containers_and_volumes(self.s_phdl)
        return True


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
    """Leak-guard + same yield shape as vLLM; implementation is host Docker."""
    ctx = InferenceMaxHostContext(cluster_dict, inference_dict, benchmark_params_dict, model_name)
    yield ctx
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down InferenceMax containers")
        ctx.teardown_containers()


@pytest.fixture(scope="module")
def gpu_type(orch):
    head_node = orch.s_phdl.host_list[0]
    smi_out_dict = orch.s_phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    return get_model_from_rocm_smi_output(smi_out)


@pytest.fixture(scope="session")
def inf_res_dict():
    return {}


def pytest_generate_tests(metafunc):
    """Single sweep cell from legacy ``benchmark_params`` (parity hook with vLLM)."""
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
        "test_aa_launch_container": 0,
        "test_inferencemax_inference": 1,
        "test_print_results_table": 2,
        "test_zz_teardown": 3,
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
