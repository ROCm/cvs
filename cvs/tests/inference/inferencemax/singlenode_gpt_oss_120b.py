'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import sys
import os
import sys
import time
import json
import logging
import time

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib import docker_lib
from cvs.lib import inference_max_lib
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
def training_config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the training config JSON file specified via --config_file.

    Notes:
      - Ensure your pytest.ini or CLI includes: --config_file=/path/to/training_config.json
      - Module scope avoids re-fetching the option across tests in this module.
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
def inference_dict(training_config_file, cluster_dict):
    with open(training_config_file) as json_file:
        inference_dict_t = json.load(json_file)
    inference_dict = inference_dict_t['config']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    inference_dict = resolve_test_config_placeholders(inference_dict, cluster_dict)
    return inference_dict


@pytest.fixture(scope="module")
def benchmark_params_dict(training_config_file, cluster_dict):
    with open(training_config_file) as json_file:
        inference_dict_t = json.load(json_file)
    benchmark_params_dict = inference_dict_t['benchmark_params']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    benchmark_params_dict = resolve_test_config_placeholders(benchmark_params_dict, cluster_dict)

    log.info(benchmark_params_dict)
    return benchmark_params_dict


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    """
    Load the Hugging Face access token from the file path specified in the training config.

    Args:
      inference_dict (dict): Training configuration dict that includes:
        - 'hf_token_file': Path to the file containing the HF token.

    Returns:
      str: The HF token string read from the file.

    Behavior:
      - Reads the token from inference_dict['hf_token_file'] (already resolved for placeholders).
      - Strips the trailing newline from the token.
    """
    hf_token_file = inference_dict['hf_token_file']
    try:
        with open(hf_token_file, 'r') as fp:
            hf_token = fp.read().rstrip("\n")
    except FileNotFoundError:
        print(f"Error: The file '{hf_token_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return hf_token


@pytest.fixture(scope="module")
def s_phdl(cluster_dict):
    """
    Create and return a parallel SSH handle for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster configuration loaded by another fixture. Expected keys:
        - 'node_dict': dict of node_name -> node_details (used to derive the node list)
        - 'username': SSH username for connecting to nodes
        - 'priv_key_file': path to the SSH private key file

    Returns:
      Pssh: An initialized Pssh handle for issuing commands across all nodes.

    Behavior:
      - Prints the full cluster_dict for quick debugging (consider switching to log.debug to reduce noise).
      - Collects all node names from cluster_dict['node_dict'] and constructs a Pssh handle.

    Notes:
      - This fixture has module scope, so a single connection handle is reused for all tests in the module.
    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    s_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return s_phdl


@pytest.fixture(scope="module")
def c_phdl(cluster_dict):
    """
    Create and return a parallel SSH handle for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster configuration loaded by another fixture. Expected keys:
        - 'node_dict': dict of node_name -> node_details (used to derive the node list)
        - 'username': SSH username for connecting to nodes
        - 'priv_key_file': path to the SSH private key file

    Returns:
      Pssh: An initialized Pssh handle for issuing commands across all nodes.

    Behavior:
      - Prints the full cluster_dict for quick debugging (consider switching to log.debug to reduce noise).
      - Collects all node names from cluster_dict['node_dict'] and constructs a Pssh handle.

    Notes:
      - This fixture has module scope, so a single connection handle is reused for all tests in the module.
    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    c_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return c_phdl


@pytest.fixture(scope="module")
def gpu_type(s_phdl, cluster_dict):
    """
    Provide the GPU type string for the test module.

    Args:
      cluster_dict (dict): Cluster configuration that includes the GPU type.

    Returns:
      str: The GPU type (e.g., 'mi300', 'mi300x') used to select model parameters and logic.

    Notes:
      - Module scope ensures this is evaluated once per test module.
      - Consider validating this value against an expected set of GPU types to catch typos early.
    """

    print(s_phdl)
    print(dir(s_phdl))
    head_node = s_phdl.host_list[0]
    smi_out_dict = s_phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    return gpu_type


def test_cleanup_stale_containers(s_phdl, inference_dict):
    """
    Pytest: Clean up potentially stale Docker containers and volumes before tests.

    Args:
      s_phdl: Parallel SSH/process handle used by docker_lib to run commands on nodes.
      inference_dict (dict): Training configuration dict that includes:
        - 'container_name': Name of the container to be killed if running.

    Behavior:
      - Kills the specific container identified by inference_dict['container_name'].
      - Deletes all containers and volumes on the target nodes (broad cleanup).

    Notes:
      - This performs a broad cleanup via delete_all_containers_and_volumes; ensure the
        test environment is isolated so this doesn?t remove unrelated containers/volumes.
      - Consider narrowing cleanup scope if other workloads may be present on the hosts.
    """

    container_name = inference_dict['container_name']
    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)


def test_launch_inference_containers(s_phdl, inference_dict):
    """ """

    log.info('Testcase launch InferenceMax containers')
    globals.error_list = []
    container_name = inference_dict['container_name']
    # Launch the containers ..
    docker_lib.launch_docker_container(
        s_phdl,
        container_name,
        inference_dict['container_image'],
        inference_dict['container_config']['device_list'],
        inference_dict['container_config']['volume_dict'],
        inference_dict['container_config']['env_dict'],
        shm_size='48G',
        timeout=60 * 20,
    )
    # ADD verifications ..
    time.sleep(30)
    print('Verify if the containers have been launched properly')
    out_dict = s_phdl.exec('docker ps')
    for node in out_dict.keys():
        if not re.search(f'{container_name}', out_dict[node], re.I):
            fail_test(f'Failed to launch container on node {node}')
    update_test_result()


def test_gpt_oss_120_single_node(c_phdl, s_phdl, gpu_type, inference_dict, benchmark_params_dict, hf_token):
    globals.error_list = []
    im_obj = inference_max_lib.InferenceMaxJob(
        c_phdl,
        s_phdl,
        'gpt-oss-120b',
        inference_dict,
        benchmark_params_dict,
        hf_token,
        gpu_type,
        distributed_inference=False,
    )
    im_obj.build_server_inference_job_cmd()
    im_obj.start_inference_server_job()
    im_obj.start_inference_client_job()
    im_obj.poll_for_inference_completion()
    im_obj.verify_inference_results()
    update_test_result()
