'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals, mori_reporting
from cvs.lib.conf_lib.loader import load_config
from cvs.lib.conf_lib.schemas.mori import MoriConfig
from cvs.lib.mori_lib import DMESG_DATE_CMD, built_commit, io_initiator

log = globals.log

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_KEY = pytest.StashKey[dict]()


@pytest.fixture(scope="module")
def loaded_config(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_config(MoriConfig, config_file, cluster_file)


@pytest.fixture(scope="module")
def mori_dict(loaded_config):
    return loaded_config.config.model_dump()


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
def orch(loaded_config, mori_dict, lifecycle):
    """Suite-local orchestrator for container or baremetal mode.

    Unlike the shared ``orch`` fixture this never calls ``setup_sshd()``:
    torchrun rendezvous is TCP and every mpiexec is node-local, so mori needs
    no inter-container SSH, and MORI images may not ship sshd. The container
    is launched by test_launch_mori_container and removed by test_teardown;
    the finalizer is a leak guard for runs that never reach test_teardown.
    """
    cluster = loaded_config.cluster
    testsuite_config = {"container": loaded_config.container or cluster.get("container", {})}
    if mori_dict.get("orchestrator"):
        testsuite_config["orchestrator"] = mori_dict["orchestrator"]
    cfg = OrchestratorConfig.from_configs(cluster, testsuite_config)
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


def launch_mori_container(orch, lifecycle):
    if orch.orchestrator_type != "container":
        pytest.skip("baremetal orchestrator: no container to launch")
    image = orch.container_config.get("image")
    if not image:
        lifecycle.failed = True
        pytest.fail("container.image is not set in the mori config")
    name = orch.get_container_name(orch.container_config, image)
    lifecycle.torn_down = False
    if not orch.setup_containers():
        lifecycle.failed = True
        pytest.fail(f"setup_containers() returned False for {name}")
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def teardown_mori_container(orch, lifecycle):
    if orch.orchestrator_type != "container":
        lifecycle.torn_down = True
        pytest.skip("baremetal orchestrator: no container to tear down")
    image = orch.container_config.get("image")
    if not image:
        lifecycle.torn_down = True
        pytest.skip("container.image is not set in the mori config: no container was launched")
    name = orch.get_container_name(orch.container_config, image)
    orch.teardown_containers()
    lifecycle.torn_down = True
    # no_launch / persistent containers are left running by design.
    if orch.container_config.get("lifetime", "per_run") == "per_run" and orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")


@pytest.fixture(scope="module")
def cvs_results_dict(request):
    """The Run Deck store, shared with pytest_runtest_makereport whichever of them needs it first."""
    return request.config.stash.setdefault(_RESULTS_KEY, {})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record every failed or errored test in the Run Deck, so a test the deck does not chart still fails it."""
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        reason = call.excinfo.exconly() if call.excinfo else report.outcome
        results = item.config.stash.setdefault(_RESULTS_KEY, {})
        mori_reporting.record_test_failure(results, item.name, report.when, reason)


@pytest.fixture(scope="module")
def mori_variant_config(mori_dict, orch):
    """Run-card fields for the mori Run Deck."""
    return {
        "gpu_name": mori_dict.get("gpu_name"),
        "node_count": len(orch.hosts),
        "mori_device_list": (mori_dict.get("env") or {}).get("MORI_RDMA_DEVICES"),
        "container_image": orch.container_config.get("image") if orch.orchestrator_type == "container" else None,
        "hosts": list(orch.hosts),
        "initiator": io_initiator(mori_dict, orch),
        "mori_commit": built_commit(orch, mori_dict["mori_dir"]),
    }


def pytest_collection_modifyitems(items):
    """Pin lifecycle order: cleanup → launch → install → setup → single-node → I/O → dmesg → teardown."""
    rank = {
        "test_cleanup_stale_containers": 0,
        "test_launch_mori_container": 1,
        "test_install_mori": 2,
        "test_setup_ibv_devices": 3,
        "test_install_container_packages": 4,
        "test_setup_env": 5,
        "test_shmem_api": 6,
        "test_concurrent_put_threads": 7,
        "test_concurrent_put_imm_threads": 8,
        "test_concurrent_put_signal_thread": 9,
        "test_ibgda_write_test": 10,
        "test_io_read": 11,
        "test_io_write": 12,
        "test_verify_dmesg": 13,
        "test_teardown": 14,
    }

    def key(it):
        return rank.get(it.originalname or it.name.split("[")[0], 50)

    # This hook sees the whole session; permute only mori's own slots so other suites stay put.
    slots = [i for i, it in enumerate(items) if str(it.path.parent) == _HERE]
    for i, it in zip(slots, sorted((items[i] for i in slots), key=key)):
        items[i] = it
