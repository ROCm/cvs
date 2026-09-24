"""Module-scoped Aorta configuration, orchestration and cleanup."""

import json
import logging
from types import SimpleNamespace

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib.benchmark.aorta.aorta_config_loader import load_training_variant
from cvs.lib.benchmark.aorta.aorta_job import AortaJob
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.parsers.schemas import ClusterConfigFile

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    with open(path) as source:
        raw = resolve_cluster_config_placeholders(json.load(source))
    ClusterConfigFile.model_validate(raw)
    return raw


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict):
    path = pytestconfig.getoption("config_file")
    if not path:
        pytest.fail("--config_file is required")
    return load_training_variant(path, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return SimpleNamespace(failed=False, torn_down=False, container_started=False, benchmark_result=None, parser=None)


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    config = OrchestratorConfig.from_configs(
        cluster_dict, {"orchestrator": "container", "container": variant_config.container.model_dump()}
    )
    orchestrator = OrchestratorFactory.create_orchestrator(log, config)
    try:
        yield orchestrator
    finally:
        try:
            if not lifecycle.torn_down:
                if not orchestrator.teardown_containers():
                    raise RuntimeError("Aorta container teardown failed in fixture cleanup")
        finally:
            orchestrator.close()


@pytest.fixture(scope="module")
def aorta_job(orch, variant_config, cluster_dict, lifecycle, request):
    single = request.module.__name__.endswith("_single")
    if single and len(orch.hosts) != 1:
        pytest.fail("aorta_single requires exactly one node; use aorta_distributed for multiple nodes")
    if not single and len(orch.hosts) < 2:
        pytest.fail("aorta_distributed requires at least two nodes; use aorta_single for one node")
    addresses = {host: data.get("vpc_ip") for host, data in cluster_dict["node_dict"].items()}
    job = AortaJob(orch, variant_config, node_vpc_ips=addresses)
    try:
        yield job
    finally:
        if lifecycle.container_started and not lifecycle.torn_down:
            job.teardown()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    state = item.funcargs.get("lifecycle")
    if state is not None and report.failed:
        state.failed = True
