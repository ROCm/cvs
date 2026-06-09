'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.dtni.config_loader import load_variant
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

log = globals.log


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
def orch(cluster_dict, variant_config):
    """Build a ContainerOrchestrator from cluster_dict + variant.container, own its lifetime."""
    container_block = variant_config.container.model_dump()
    container_block["image"] = variant_config.image.tag
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
    }
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    if not o.setup_containers():
        pytest.fail(
            f"Failed to launch container: {o.get_container_name(o.container_config, o.container_config['image'])}"
        )
    if not o.setup_sshd():
        pytest.fail("Failed to setup sshd in container")
    yield o
    o.teardown_containers()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="session")
def inf_res_dict():
    return {}


def pytest_generate_tests(metafunc):
    """Parametrize test_vllm_inference over sequence_combinations × concurrency_levels."""
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    concs = sweep.get("concurrency_levels", [])
    cases = []
    ids = []
    for combo in combos:
        default_name = "isl" + combo["isl"] + "_osl" + combo["osl"]
        for c in concs:
            cases.append((combo, c))
            ids.append(combo.get("name", default_name) + "-conc" + str(c))
    if "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)
