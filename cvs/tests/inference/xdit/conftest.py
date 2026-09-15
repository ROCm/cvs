"""
Shared pytest wiring for xDiT inference suites.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.utils_lib import get_model_from_rocm_smi_output, resolve_cluster_config_placeholders
from cvs.tests.inference.xdit._shared import (
    Lifecycle,
    XDIT_TEST_ORDER,
    inference_from_variant,
    resolve_execution_hosts,
    scoped_cluster_dict,
    suite_spec,
)

log = globals.log


class _SecretValue:
    def __init__(self, value):
        self.value = value or ""

    def __bool__(self):
        return bool(self.value)

    def __str__(self):
        return self.value

    def __repr__(self):
        return "<redacted>"


def _deep_merge(base, override):
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(base[key], value) if key in base else value
    return merged


def _create_container_orchestrator(cluster_dict, variant_config):
    from cvs.lib.inference.xdit.xdit_config_loader import orchestrator_container_from_variant

    container = _deep_merge(
        cluster_dict.get("container", {}),
        orchestrator_container_from_variant(variant_config),
    )
    config = OrchestratorConfig.from_configs(
        cluster_dict,
        {"orchestrator": "container", "container": container},
    )
    return OrchestratorFactory.create_orchestrator(log, config)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    return path


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    path = pytestconfig.getoption("config_file")
    if not path:
        pytest.fail("--config_file is required")
    return path


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file, encoding="utf-8") as fp:
        return resolve_cluster_config_placeholders(json.load(fp))


@pytest.fixture(scope="module")
def variant_config(config_file, cluster_dict):
    from cvs.lib.inference.xdit.xdit_config_loader import load_variant

    return load_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def xdit_spec(request):
    return suite_spec(request.module.__name__)


@pytest.fixture(scope="module")
def lifecycle():
    return Lifecycle()


@pytest.fixture(scope="module")
def cvs_results_dict(lifecycle):
    return lifecycle.report_results


@pytest.fixture(scope="module")
def inference_dict(variant_config):
    return inference_from_variant(variant_config)


@pytest.fixture(scope="module")
def benchmark_params_dict(variant_config):
    value = getattr(variant_config, "benchmark_params", None)
    if value is None and isinstance(variant_config, dict):
        value = variant_config.get("benchmark_params")
    return value


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    path = inference_dict.get("hf_token_file") or ""
    if not path:
        return _SecretValue("")
    if not os.path.isfile(path):
        log.warning("HF token file missing: %s", path)
        return _SecretValue("")
    with open(path, encoding="utf-8") as fp:
        return _SecretValue(fp.read().strip())


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, xdit_spec, lifecycle):
    inference = inference_from_variant(variant_config)
    try:
        hosts = resolve_execution_hosts(cluster_dict, inference, xdit_spec["distributed"])
    except ValueError as exc:
        pytest.fail(str(exc))
    scoped_cluster = scoped_cluster_dict(cluster_dict, hosts)
    log.info("xDiT orchestrator scoped to hosts=%s head=%s", hosts, hosts[0])
    orchestrator = _create_container_orchestrator(scoped_cluster, variant_config)

    yield orchestrator

    try:
        if not lifecycle.torn_down:
            log.info("xDiT orchestrator leak-guard: tearing down containers")
            orchestrator.teardown_containers()
    finally:
        orchestrator.close()


@pytest.fixture(scope="module")
def gpu_type(orch):
    output_by_host = orch.all.exec("rocm-smi -a | head -30")
    output = next(iter(output_by_host.values()), "")
    return get_model_from_rocm_smi_output(output)


def pytest_collection_modifyitems(items):
    items.sort(key=lambda item: XDIT_TEST_ORDER.get(item.originalname or item.name.split("[")[0], 99))
