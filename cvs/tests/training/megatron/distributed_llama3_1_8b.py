'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import json


from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib import docker_lib
from cvs.lib import megatron_training_lib

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
def training_dict(training_config_file, cluster_dict):
    """
    Load the training configuration section ('config') from the training JSON file.

    Args:
      training_config_file (str): Path to the training config JSON file.
      cluster_dict: Cluster configuration (for placeholder resolution)

    Returns:
      dict: The 'config' nested dictionary with training/test parameters.

    Notes:
      - Assumes the JSON root contains a 'config' key.
    """
    with open(training_config_file) as json_file:
        training_dict_t = json.load(json_file)
    training_dict = training_dict_t['config']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    training_dict = resolve_test_config_placeholders(training_dict, cluster_dict)

    return training_dict


@pytest.fixture(scope="module")
def model_params_dict(training_config_file, cluster_dict):
    """
    Load model parameter presets from the training config JSON file.

    Args:
      training_config_file (str): Path to the training config JSON file.
      cluster_dict: Cluster configuration (for placeholder resolution)

    Returns:
      dict: The 'model_params' nested dictionary (e.g., single_node/multi_node presets).
    """
    with open(training_config_file) as json_file:
        training_dict_t = json.load(json_file)
    model_params_dict = training_dict_t['model_params']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    model_params_dict = resolve_test_config_placeholders(model_params_dict, cluster_dict)

    log.info(model_params_dict)
    return model_params_dict


@pytest.fixture(scope="module")
def hf_token(training_dict):
    """
    Load the Hugging Face access token from the file path specified in the training config.

    Args:
      training_dict (dict): Training configuration dict that includes:
        - 'hf_token_file': Path to the file containing the HF token.

    Returns:
      str: The HF token string read from the file.

    Behavior:
      - Reads the token from training_dict['hf_token_file'] (already resolved for placeholders).
      - Strips the trailing newline from the token.
    """
    hf_token_file = training_dict['hf_token_file']
    try:
        with open(hf_token_file, 'r') as fp:
            hf_token = fp.read().rstrip("\n")
    except FileNotFoundError:
        print(f"Error: The file '{hf_token_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return hf_token


@pytest.fixture(scope="module")
def phdl(cluster_dict):
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
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.fixture(scope="module")
def gpu_type(
    phdl,
):
    """
    Provide the GPU type string for the test module.

    Args:
      phdl

    Returns:
      str: The GPU type (e.g., 'mi300', 'mi300x') used to select model parameters and logic.

    Notes:
      - Module scope ensures this is evaluated once per test module.
      - Consider validating this value against an expected set of GPU types to catch typos early.
    """
    head_node = phdl.host_list[0]
    smi_out_dict = phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    return gpu_type


def test_disable_firewall(phdl):
    globals.error_list = []
    # Disable firewall otherwise we may have threads timing out to connect to Rendezvous
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive', out_dict[node], re.I):
            phdl.exec('sudo service ufw stop')
            continue
    out_dict = phdl.exec('sudo ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|disabled', out_dict[node], re.I):
            fail_test(f'Failed to disable firewall on node {node}')
    update_test_result()


def test_cleanup_stale_containers(phdl, training_dict):
    """
    Pytest: Clean up potentially stale Docker containers and volumes before tests.

    Args:
      phdl: Parallel SSH/process handle used by docker_lib to run commands on nodes.
      training_dict (dict): Training configuration dict that includes:
        - 'container_name': Name of the container to be killed if running.

    Behavior:
      - Kills the specific container identified by training_dict['container_name'].
      - Deletes all containers and volumes on the target nodes (broad cleanup).

    Notes:
      - This performs a broad cleanup via delete_all_containers_and_volumes; ensure the
        test environment is isolated so this doesn?t remove unrelated containers/volumes.
      - Consider narrowing cleanup scope if other workloads may be present on the hosts.
    """

    container_name = training_dict['container_name']
    docker_lib.kill_docker_container(phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(phdl)


def test_launch_megatron_containers(phdl, training_dict):
    """
    Pytest: Launch Megatron training containers and verify launch step.

    Args:
      phdl: Cluster handle for executing commands across nodes.
      training_dict (dict): Training configuration including:
        - 'container_name': Name for the container(s)
        - 'container_image': Docker image to use
        - 'container_config': {
            'device_list': device pass-through config (GPUs, RDMA, etc),
            'volume_dict': bind mounts for datasets, logs, etc
          }

    Behavior:
      - Initializes the global error_list for fresh test pass/fail tracking.
      - Launches the container(s) with a large shared memory segment (shm_size='128G').
      - Uses a generous timeout (20 minutes) for image pulls/initialization.
      - Calls update_test_result() to record the outcome based on accumulated errors.

    """

    log.info('Testcase launch Megatron containers')
    globals.error_list = []
    container_name = training_dict['container_name']
    # Launch the containers ..
    docker_lib.launch_docker_container(
        phdl,
        container_name,
        training_dict['container_image'],
        training_dict['container_config']['device_list'],
        training_dict['container_config']['volume_dict'],
        shm_size='128G',
        timeout=60 * 20,
    )
    # ADD verifications ..
    update_test_result()


def test_llama_3_1_fp8_single_node(phdl, gpu_type, training_dict, model_params_dict, hf_token):
    """
    Pytest: Single-node Megatron Llama 3.1 FP8 training lifecycle test.

    Args:
      phdl: Cluster handle used by the training job to execute commands.
      gpu_type (str): GPU identifier (e.g., 'mi300') used for selecting params.
      training_dict (dict): Training configuration, volumes, environment, etc.
      model_params_dict (dict): Model/hyperparameter presets (single_node/multi_node).
      hf_token (str): Access token for Hugging Face datasets/models.

    Behavior:
      - Resets the global error list for a clean test run.
      - Instantiates MegatronLlamaTrainingJob with tuned parameters disabled.
      - Runs NIC setup (may apply vendor-specific workarounds).
      - Builds training command/config inside the container.
      - Starts training, polls for completion, and verifies results.
      - Records final test outcome via update_test_result().

    """
    globals.error_list = []
    mt_obj = megatron_training_lib.MegatronLlamaTrainingJob(
        phdl,
        'llama3_1_8b',
        training_dict,
        model_params_dict,
        hf_token,
        gpu_type,
        distributed_training=True,
        tune_model_params=False,
    )
    mt_obj.exec_nic_setup_scripts()
    mt_obj.build_training_job_cmd()
    mt_obj.start_training_job()
    mt_obj.poll_for_training_completion()
    mt_obj.verify_training_results()
    update_test_result()
