'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.inference_suite_lifecycle import (
    InferenceLifecycle,
    attach_lifecycle_html_table,
    html_metric_table_header,
    html_metric_table_row,
    sort_lifecycle_items,
)
from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    load_variant,
    orchestrator_container_from_variant,
)
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

log = globals.log


def _log_variant_run_card(variant_config):
    rc = variant_config.run_card
    parts = [
        f"gpu_arch={variant_config.gpu_arch}",
        f"driver={variant_config.params.driver}",
        f"model={variant_config.model.id}",
    ]
    atom_args = variant_config.roles.server.atom_args
    if atom_args:
        parts.append(f"atom_args={len(atom_args)} tokens")
    if rc.atom_image_pin:
        parts.append(f"image_pin={rc.atom_image_pin}")
    if rc.upstream_run_url:
        parts.append(f"upstream_run={rc.upstream_run_url}")
    if rc.notes:
        parts.append(f"notes={rc.notes}")
    log.info("InferenceX ATOM run card: %s", "; ".join(parts))


@pytest.fixture(scope="module", autouse=True)
def _emit_variant_run_card(variant_config):
    """Log the variant run card once per module (not once per sweep cell)."""
    _log_variant_run_card(variant_config)


LIFECYCLE_RANK = {
    "test_launch_container": 0,
    "test_setup_sshd": 1,
    "test_model_fetch": 2,
    "test_inferencex_atom_inference": 3,
    "test_cell_metrics": 4,
    "test_print_results_table": 5,
    "test_teardown": 6,
}


def _deep_merge(base, override):
    """Recursively merge ``override`` onto ``base`` (dicts merged key-wise; scalars/lists replaced)."""
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
    return load_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return InferenceLifecycle()


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        # also injects roles.server.env — do not replace with .container.model_dump()
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
        log.info("orch fixture leak-guard: tearing down InferenceX ATOM containers")
        o.teardown_containers()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def server_session():
    """Tracks the active server session key to allow reuse across sweep cells when reuse_server_across_sweep=true."""
    return {"key": None}


@pytest.fixture(scope="module")
def inf_res_dict():
    return {}


def pytest_collection_modifyitems(items):
    sort_lifecycle_items(items, LIFECYCLE_RANK)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    attach_lifecycle_html_table(item, outcome.get_result())


def pytest_html_results_table_header(cells):
    html_metric_table_header(cells)


def pytest_html_results_table_row(report, cells):
    html_metric_table_row(report, cells)
