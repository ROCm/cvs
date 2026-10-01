'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.mori_lib import DMESG_DATE_CMD
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, resolve_test_config_placeholders

log = globals.log

_HERE = os.path.dirname(os.path.abspath(__file__))


def _deep_merge(base, override):
    """Recursively merge ``override`` onto ``base``; dicts merge key-wise, scalars and lists replace."""
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
def mori_dict(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    with open(config_file) as fp:
        d = json.load(fp)
    return resolve_test_config_placeholders(d, cluster_dict)


class _Lifecycle:
    """Cross-test state for the mori suite.

    ``failed`` lets a failed container launch skip the benchmark tests.
    ``torn_down`` suppresses the orch fixture leak-guard once test_teardown ran.
    ``dmesg_start`` is the per-host start of the dmesg window, taken when the
    orchestrator is created so the scan covers the whole suite whatever the
    test selection.
    """

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.dmesg_start = None


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def orch(cluster_dict, mori_dict, lifecycle):
    """Suite-local orchestrator for container or baremetal mode.

    Unlike the shared ``orch`` fixture this never calls ``setup_sshd()``:
    torchrun rendezvous is TCP and every mpiexec is node-local, so mori needs
    no inter-container SSH, and MORI images may not ship sshd. The container
    is launched by test_launch_mori_container and removed by test_teardown;
    the finalizer is a leak guard for runs that never reach test_teardown.
    """
    testsuite_config = {"container": _deep_merge(cluster_dict.get("container", {}), mori_dict.get("container", {}))}
    if mori_dict.get("orchestrator"):
        testsuite_config["orchestrator"] = mori_dict["orchestrator"]
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    log.info("mori orchestrator: %s", cfg.orchestrator)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    lifecycle.dmesg_start = o.all.exec(DMESG_DATE_CMD)
    yield o
    try:
        if o.orchestrator_type == "container" and not lifecycle.torn_down:
            log.info("orch fixture leak-guard: tearing down container (test_teardown did not run)")
            o.teardown_containers()
    finally:
        o.close()


def pytest_collection_modifyitems(items):
    """Pin lifecycle order: cleanup → launch → setup → single-node → I/O → dmesg → teardown."""
    rank = {
        "test_cleanup_stale_containers": 0,
        "test_launch_mori_container": 1,
        "test_setup_ibv_devices": 2,
        "test_install_container_packages": 3,
        "test_setup_env": 4,
        "test_shmem_api": 5,
        "test_concurrent_put_threads": 6,
        "test_concurrent_put_imm_threads": 7,
        "test_concurrent_put_signal_thread": 8,
        "test_ibgda_write_test": 9,
        "test_io_read": 10,
        "test_io_write": 11,
        "test_verify_dmesg": 12,
        "test_teardown": 13,
    }

    def key(it):
        if str(it.path.parent) != _HERE:
            return 99
        return rank.get(it.originalname or it.name.split("[")[0], 50)

    items.sort(key=key)
