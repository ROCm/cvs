'''Fixtures for the llm-d vLLM lifecycle suite.'''

import json
import os
import time

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.orchestrator.llm_d.vllm import LlmdTopology, LlmdVllmStack, load_config, scope_cluster
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

log = globals.log


class Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}
        self.smoke_results = []

    def record(self, nodeid, label, started):
        self.report.setdefault(nodeid, []).append((label, time.monotonic() - started, "s"))


@pytest.fixture(scope="module")
def lifecycle():
    return Lifecycle()


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    with open(path, encoding="utf-8") as stream:
        return resolve_cluster_config_placeholders(json.load(stream))


@pytest.fixture(scope="module")
def llmd_config(pytestconfig, cluster_dict):
    path = pytestconfig.getoption("config_file")
    if not path:
        pytest.fail("--config_file is required")
    return load_config(path, cluster_dict)


@pytest.fixture(scope="module")
def llmd_topology(llmd_config, cluster_dict):
    return LlmdTopology(llmd_config, cluster_dict)


@pytest.fixture(scope="module")
def hf_token(llmd_config):
    path = llmd_config.paths.hf_token_file
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as stream:
        return stream.read().strip()


@pytest.fixture(scope="module")
def orch(cluster_dict, llmd_topology):
    scoped = scope_cluster(cluster_dict, llmd_topology)
    config = OrchestratorConfig.from_configs(scoped)
    instance = OrchestratorFactory.create_orchestrator(log, config)
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def stack(orch, llmd_config, llmd_topology, lifecycle, hf_token):
    instance = LlmdVllmStack(orch, llmd_config, llmd_topology, log, hf_token=hf_token)
    yield instance
    if not lifecycle.torn_down:
        log.info("llm-d leak guard: removing named stack containers")
        instance.cleanup()


@pytest.fixture(autouse=True)
def skip_after_failure(request, lifecycle):
    if request.node.name != "test_teardown" and lifecycle.failed:
        pytest.skip("a prior llm-d lifecycle stage failed")


def pytest_collection_modifyitems(items):
    order = {
        "test_resolve_topology": 0,
        "test_stage_gateway_config": 1,
        "test_launch_vllm_workers": 2,
        "test_poll_workers_ready": 3,
        "test_launch_epp": 4,
        "test_launch_envoy": 5,
        "test_gateway_ready": 6,
        "test_openai_compatible_http_endpoints": 7,
        "test_verify_dmesg": 8,
        "test_teardown": 9,
    }
    items.sort(key=lambda item: order.get(item.originalname or item.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed and "lifecycle" in item.funcargs:
        item.funcargs["lifecycle"].failed = True
