'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import verify_dmesg_for_errors
from cvs.lib import mori_lib
from cvs.lib import globals

log = globals.log


# -----------------------------------------------------------------------------
# Pytest Fixture: mori_obj
#
# This fixture creates and provides a single instance of MoriBenchmark
# for all tests within a module.
#
# Scope: module
#   - The fixture is instantiated once per test module
#   - The same MoriBenchmark object is shared across all tests in that module
#
# Dependencies:
#   orch      : Orchestrator (container or baremetal) from the suite conftest
#   mori_dict : Configuration dictionary for MORI benchmark parameters
#   lifecycle : Cross-test state; a failed container launch skips every user
#
# Side effects:
#   - Resets global error list at fixture creation time to ensure a clean
#     error state before any tests are executed.
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mori_obj(orch, mori_dict, lifecycle):
    if lifecycle.failed:
        pytest.skip("mori container launch failed")
    globals.error_list = []
    return mori_lib.MoriBenchmark(orch, mori_dict)


def test_cleanup_stale_containers(orch):
    """Kept for test-ID compatibility. Under the container orchestrator a
    ``per_run`` launch force-removes a same-named container itself, and
    ``docker system prune`` is not safe on shared nodes."""
    pytest.skip("stale-container cleanup is handled by the orchestrator launch")


def test_launch_mori_container(orch, lifecycle):
    if orch.orchestrator_type != "container":
        pytest.skip("baremetal orchestrator: no container to launch")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    lifecycle.torn_down = False
    if not orch.setup_containers():
        lifecycle.failed = True
        pytest.fail(f"setup_containers() returned False for {name}")
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


# Ensure the MORI RDMA devices show up where the benchmarks run
def test_setup_ibv_devices(mori_obj):
    globals.error_list = []
    mori_obj.check_ibv_devices()
    update_test_result()


def test_install_container_packages(mori_obj):
    globals.error_list = []
    if not mori_obj.install_packages():
        pytest.skip("baremetal orchestrator: host packages are not installed by the suite")
    update_test_result()


def test_setup_env(mori_obj):
    globals.error_list = []
    mori_obj.create_run_log_dir()
    update_test_result()


def test_shmem_api(mori_obj):
    globals.error_list = []
    mori_obj.run_shmem_apitest()
    update_test_result()


# def test_dispatch_combine(mori_obj):
#    globals.error_list = []
#    mori_obj.run_dispatch_combine()
#    update_test_result()


# def test_bench_dispatch_combine(mori_obj):
#    globals.error_list = []
#    mori_obj.run_bench_dispatch_combine()
#    update_test_result()


def test_concurrent_put_threads(mori_obj):
    globals.error_list = []
    mori_obj.run_concurrent_put_threads()
    update_test_result()


def test_concurrent_put_imm_threads(mori_obj):
    globals.error_list = []
    mori_obj.run_concurrent_put_imm_threads()
    update_test_result()


def test_concurrent_put_signal_thread(mori_obj):
    globals.error_list = []
    mori_obj.run_concurrent_put_signal_thread()
    update_test_result()


def test_ibgda_write_test(mori_obj):
    globals.error_list = []
    mori_obj.run_ibgda_dist_write(no_of_procs=2, min_val=2, max_val='64m', ctas=2, threads=256, qp_count=4, iters=1)

    update_test_result()


# -----------------------------------------------------------------------------
# Input Test Matrix
#
# This matrix defines combinations of:
#   - buffer_size              : Size of the IO buffer (bytes)
#   - transfer_batch_size      : Number of transfers grouped per batch
#   - no_of_qp_per_transfer    : Number of Queue Pairs used per transfer
#
# Each tuple represents one independent test configuration.
# The matrix is used by pytest.parametrize to generate multiple test cases,
# allowing systematic coverage of different IO and parallelism settings.
#
# These combinations are chosen to:
#   - Exercise different buffer sizes
#   - Validate single-QP vs multi-QP transfers
#   - Test multiple batch sizes for scalability
# -----------------------------------------------------------------------------
input_test_matrix = [
    (16384, 128, 1),
    (16384, 128, 8),
    (32768, 128, 1),
    (32768, 128, 8),
    (32768, 256, 1),
    (32768, 256, 8),
]


# -----------------------------------------------------------------------------
# Read IO Test
#
# This test validates MORI read-path performance using the torch-based
# distributed IO benchmark.
#
# For each tuple in input_test_matrix:
#   - A fresh test run is executed
#   - Errors are collected and tracked globally
#   - Performance results are validated inside run_mori_torch_io_test()
#
# pytest.parametrize automatically expands this function into multiple
# independent test cases, one per input configuration.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("buffer_size,transfer_batch_size,no_of_qp_per_transfer", input_test_matrix)
def test_io_read(mori_obj, buffer_size, transfer_batch_size, no_of_qp_per_transfer):
    globals.error_list = []
    mori_obj.run_mori_torch_io_test(
        op_type='read', enable_sess=True, buffer_size=buffer_size, transfer_batch_size=transfer_batch_size
    )
    update_test_result()


# -----------------------------------------------------------------------------
# Write IO Test
#
# This test mirrors test_io_read, but exercises the WRITE path instead.
# Using the same input matrix ensures direct read-vs-write comparison
# across identical IO configurations.
#
# Each parameter combination produces a separate pytest test case.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("buffer_size,transfer_batch_size,no_of_qp_per_transfer", input_test_matrix)
def test_io_write(mori_obj, buffer_size, transfer_batch_size, no_of_qp_per_transfer):
    globals.error_list = []
    mori_obj.run_mori_torch_io_test(
        op_type='write', enable_sess=True, buffer_size=buffer_size, transfer_batch_size=transfer_batch_size
    )
    update_test_result()


def test_verify_dmesg(orch, lifecycle):
    """Scan host dmesg over the suite's window. Always via ``orch.all`` (the host
    handle), because a container cannot read the host kernel log reliably."""
    if not lifecycle.dmesg_start:
        pytest.skip("no dmesg start timestamp was recorded")
    if not orch.sudo_prefix():
        pytest.skip("passwordless sudo unavailable on the hosts; dmesg scan needs sudo")
    globals.error_list = []
    end_time = orch.all.exec(mori_lib.DMESG_DATE_CMD)
    verify_dmesg_for_errors(orch.all, lifecycle.dmesg_start, end_time, till_end_flag=False)
    update_test_result()


def test_teardown(orch, lifecycle):
    if orch.orchestrator_type != "container":
        lifecycle.torn_down = True
        pytest.skip("baremetal orchestrator: no container to tear down")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    orch.teardown_containers()
    lifecycle.torn_down = True
    # no_launch / persistent containers are left running by design.
    if orch.container_config.get("lifetime", "per_run") == "per_run" and orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
