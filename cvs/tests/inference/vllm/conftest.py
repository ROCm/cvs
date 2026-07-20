'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.utils.vllm_config_loader import load_variant
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

log = globals.log


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced).

    Protects cluster-set SCALAR and DICT container keys (e.g. shm_size, an env
    map) from being wiped by a top-level replace: they survive unless the variant
    overrides that same key. List keys (e.g. runtime.args, volume mounts) are
    REPLACED here, not unioned -- the cluster's list values are recombined with
    the variant's additively further downstream, in container.py's getters.
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
    return load_variant(config_file, cluster_dict)


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model.

    The container launch / sshd / fetch / teardown stages are individual tests
    (so each is a timed, pass/fail row in the HTML) rather than fixture body
    code. They share this object: `failed` lets a broken stage skip the rest
    instead of cascading; `torn_down` lets the explicit teardown test suppress
    the fixture's leak-guard finalizer; `report` maps a test's nodeid to the
    rows it recorded, each carrying its own unit, so attach_inference_suite_lifecycle_table
    renders only that test's stages -- not every stage on every row.
    """

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}  # nodeid -> list[(label, value, unit)]

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    """Construct a ContainerOrchestrator and own ONLY its teardown safety net.

    The actual launch/sshd happen in test_launch_container / test_setup_sshd
    so they appear as timed rows. This fixture builds the object and registers a
    leak-guard finalizer: if a mid-sweep test fails before test_teardown runs,
    the container is still torn down here. When test_teardown ran successfully
    it sets lifecycle.torn_down, so the finalizer no-ops (no double teardown).
    """
    # OrchestratorConfig.from_configs does a top-level dict.update, so a bare variant
    # container block would wipe the cluster file's container settings. Deep-merge the
    # variant ONTO the cluster block so cluster-set scalar/dict keys survive, with the
    # variant winning on conflicting keys. (List keys like runtime.args are replaced
    # here but recombined additively downstream in container.py's getters.)
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        variant_config.container.model_dump(),
    )
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
    }
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down container (explicit teardown did not run)")
        o.teardown_containers()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        if variant_config.model.remote == 0:
            # Pre-staged model: token not needed for download; server env sets
            # HF_HUB_OFFLINE=1 to skip Hub auth checks entirely.
            return ""
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def inf_res_dict():
    return {}


def pytest_collection_modifyitems(items):
    """Pin the lifecycle order explicitly instead of relying on definition order.

    `test_print_results_table` is an imported function (its source line points
    into _shared.py), so default ordering collects it FIRST -- which would log an
    empty table before any cell ran. Sort deterministically: launch, sshd, fetch,
    the benchmark cells, the results table, then teardown last. Items from other
    modules keep their relative order.
    """
    rank = {
        "test_launch_container": 0,
        "test_setup_sshd": 1,
        "test_discover_topology": 2,
        "test_model_fetch": 3,
        "test_vllm_inference": 4,
        "test_metric": 5,
        "test_print_results_table": 6,
        "test_teardown": 7,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


def pytest_html_results_table_header(cells):
    """Add Value + Unit columns just before the trailing Links column.

    Populated for test_metric rows; blank for lifecycle/inference rows (they
    record no metric_value user-property). Scoped to this suite's conftest, so
    other suites' result tables are unaffected.
    """
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
