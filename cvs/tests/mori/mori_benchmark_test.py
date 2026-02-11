'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import time
import json

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib import docker_lib
from cvs.lib import mori_lib
from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the cluster JSON file specified via --cluster_file.

    Notes:
      - Ensure your pytest.ini or CLI includes: --cluster_file=/path/to/cluster.json
      - Use module scope so the value is resolved once per test module.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def mori_config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option provided to pytest.

    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load the entire cluster configuration from the provided JSON file.

    Args:
      cluster_file (str): Path to the cluster JSON file.

    Returns:
      dict: Parsed JSON representing the cluster (nodes, credentials, etc.).

    Notes:
      - Logs the loaded structure for visibility; consider using log.debug if verbose.
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def mori_dict(mori_config_file, cluster_dict):
    with open(mori_config_file) as json_file:
        mori_dict = json.load(json_file)
    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    mori_dict = resolve_test_config_placeholders(mori_dict, cluster_dict)
    return mori_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.fixture(scope="module")
def gpu_type(phdl, cluster_dict):
    """
    Returns:
      str: The GPU type (e.g., 'mi300', 'mi300x') used to select model parameters and logic.
    """

    print(phdl)
    head_node = phdl.host_list[0]
    smi_out_dict = phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    return gpu_type


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
# This is appropriate because:
#   - MoriBenchmark setup may be expensive (environment, connections, state)
#   - Tests are logically related and can reuse the same benchmark object
#   - Avoids redundant initialization overhead for each test case
#
# Dependencies:
#   phdl      : Platform/host abstraction used to execute remote commands
#   mori_dict : Configuration dictionary for MORI benchmark parameters
#   gpu_type  : GPU type identifier (used to select tuning/expectations)
#
# Side effects:
#   - Resets global error list at fixture creation time to ensure a clean
#     error state before any tests are executed.
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mori_obj(phdl, mori_dict, gpu_type):
    # Initialize (or reset) global error tracking before tests begin
    globals.error_list = []

    # Create the MORI benchmark object that encapsulates:
    #   - Command execution
    #   - Distributed benchmark orchestration
    #   - Result collection and validation logic
    mori_obj = mori_lib.MoriBenchmark(phdl, mori_dict, gpu_type)

    # Provide the initialized MoriBenchmark instance to all tests
    # in this module that request the `mori_obj` fixture
    return mori_obj


def test_cleanup_stale_containers(phdl, mori_dict):
    """
    Pytest: Clean up potentially stale Docker containers and volumes before tests.

    Running this cleanup as a test guarantees a clean and predictable
    environment before MORI benchmark tests are executed.

    Args:
        phdl      : Platform handler used to execute Docker commands on the host
        mori_dict : Configuration dictionary containing MORI test parameters,
                    including the expected container name
    """
    container_name = mori_dict['container_name']
    docker_lib.kill_docker_container(phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(phdl)


def test_launch_mori_container(phdl, mori_dict):
    log.info('Testcase launch mori containers')
    globals.error_list = []
    container_name = mori_dict['container_name']
    docker_lib.launch_docker_container(
        phdl,
        container_name,
        mori_dict['container_image'],
        mori_dict['container_config']['device_list'],
        mori_dict['container_config']['volume_dict'],
        mori_dict['container_config']['env_dict'],
        shm_size='48G',
        timeout=60 * 20,
    )
    # ADD verifications ..
    time.sleep(30)
    print('Verify if the containers have been launched properly')
    out_dict = phdl.exec('docker ps')
    for node in out_dict.keys():
        if not re.search(f'{container_name}', out_dict[node], re.I):
            fail_test(f'Failed to launch container on node {node}')
    update_test_result()


# Setup the ib MORI devices and ensure they show up in the container
def test_setup_ibv_devices(mori_obj):
    globals.error_list = []
    mori_obj.check_ibv_devices()
    mori_obj.exec_nic_setup_scripts()
    update_test_result()


def test_install_container_packages(mori_obj):
    globals.error_list = []
    mori_obj.install_packages()
    update_test_result()


def test_setup_env(mori_obj):
    globals.error_list = []
    mori_obj.create_env_script()
    update_test_result()



def test_shmem_api(mori_obj):
    globals.error_list = []
    mori_obj.run_shmem_apitest()
    update_test_result()


#def test_dispatch_combine(mori_obj):
#    globals.error_list = []
#    mori_obj.run_dispatch_combine()
#    update_test_result()


#def test_bench_dispatch_combine(mori_obj):
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
    mori_obj.run_ibgda_dist_write(no_of_procs=2,
        min_val=2,
        max_val='64m',
        ctas=2,
        threads=256,
        qp_count=4,
        iters=1
        )

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
