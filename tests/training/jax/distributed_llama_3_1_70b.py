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

sys.path.insert( 0, './lib' )

from parallel_ssh_lib import *
from utils_lib import *
import docker_lib
import jax_training_lib

import globals
log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option value from pytest.

    Args:
      pytestconfig: Built-in pytest fixture providing access to command-line options.

    Returns:
      str: Path to the cluster configuration JSON file (provided via --cluster_file).

    Notes:
      - Ensure your pytest invocation includes: --cluster_file=/path/to/cluster.json
      - Module scope ensures this runs once per test module.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def training_config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option value from pytest.

    Args:
      pytestconfig: Built-in pytest fixture providing access to command-line options.

    Returns:
      str: Path to the training configuration JSON file (provided via --config_file).

    Notes:
      - Ensure your pytest invocation includes: --config_file=/path/to/config.json
      - Module scope ensures this runs once per test module.
    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the full cluster configuration from JSON.

    Args:
      cluster_file (str): Path to a cluster JSON file (from cluster_file fixture).

    Returns:
      dict: Parsed cluster configuration (e.g., node_dict, credentials, gpu_type, etc.).

    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def training_dict(training_config_file):
    """
    Load and return the 'config' section from the training config JSON.

    Args:
      training_config_file (str): Path to the training configuration JSON.

    Returns:
      dict: The training configuration dictionary stored under the 'config' key.

    Notes:
      - Assumes the JSON has a top-level 'config' key.
    """
    with open(training_config_file) as json_file:
        training_dict_t = json.load(json_file)
    training_dict = training_dict_t['config']
    return training_dict


@pytest.fixture(scope="module")
def model_params_dict(training_config_file):
    """
    Load and return the 'model_params' section from the training config JSON.

    Args:
      training_config_file (str): Path to the training configuration JSON.

    Returns:
      dict: Model parameter presets stored under the 'model_params' key.
    """
    with open(training_config_file) as json_file:
        training_dict_t = json.load(json_file)
    model_params_dict = training_dict_t['model_params']
    log.info(model_params_dict)
    return model_params_dict


@pytest.fixture(scope="module")
def hf_token(training_config_file):
    """
    Load a Hugging Face token from the file specified in the training config.

    Args:
      training_config_file (str): Path to the training configuration JSON.

    Returns:
      str: The HF token read from the file path under config.hf_token_file.

    Behavior:
      - Reads the training JSON and extracts config.hf_token_file.
      - Reads and strips the trailing newline from the token file content.
    """
    with open(training_config_file) as json_file:
        training_dict_t = json.load(json_file)
    hf_token_file = training_dict_t['config']['hf_token_file']
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
    Create and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster configuration loaded by another fixture. Expected keys:
        - 'node_dict': dict mapping node_name -> node_details (used to derive the node list)
        - 'username': SSH username to authenticate with
        - 'priv_key_file': path to the SSH private key file

    Returns:
      Pssh: A handle that can execute commands across all nodes in parallel.

    Behavior:
      - Prints the entire cluster_dict for quick debugging (consider switching to log.debug to reduce noise).
      - Extracts the list of nodes from cluster_dict['node_dict'] keys.
      - Initializes a Pssh handle with provided credentials.
    """
 
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl



def test_disable_firewall( phdl ):
    globals.error_list = []
    # Disable firewall otherwise we may have threads timing out to connect to Rendezvous
    phdl.exec('sudo service ufw stop')
    out_dict = phdl.exec('sudo ufw status')
    for node in out_dict.keys():
        if not re.search( 'inactive|disabled', out_dict[node], re.I ):
            fail_test(f'Failed to disable firewall on node {node}')
    update_test_result()




def test_cleanup_stale_containers( phdl, training_dict ):
    """
    Pytest: Clean up any stale Docker containers/volumes before running training tests.

    Args:
      phdl: Parallel SSH/process handle used by docker_lib to execute commands across nodes.
      training_dict (dict): Training configuration that includes:
        - 'container_name': The specific container to stop/kill if present.

    Behavior:
      - Kills the specified container by name if it exists.
      - Performs a broad cleanup of all containers and volumes on target nodes to ensure a clean slate.

    Notes:
      - we do broad cleanup (delete_all_containers_and_volumes); ensure the test environment
    """
    globals.error_list = []
    container_name = training_dict['container_name']
    docker_lib.kill_docker_container( phdl, container_name )
    docker_lib.delete_all_containers_and_volumes( phdl )
    update_test_result()




def test_launch_jax_containers(phdl, training_dict ):

    """
    Pytest: Launch JAX training containers and record the outcome.

    Args:
      phdl: Cluster handle to execute commands across nodes.
      training_dict (dict): Training configuration including:
        - 'container_name': Name to assign to the container(s)
        - 'container_image': Docker image URI
        - 'container_config': {
            'device_list': Device pass-through (GPUs, RDMA, etc.)
            'volume_dict': Host->container mounts for data/logs
            'env_dict': Environment variables to set inside the container
          }

    Behavior:
      - Resets globals.error_list to isolate this test's failures.
      - Launches the container(s) with:
          * a large shared memory segment (shm_size='256G') suited for ML workloads,
          * a generous timeout (20 minutes) to tolerate pulls/initialization.
      - Calls update_test_result() to mark test outcome based on collected errors.

    """

    log.info('Testcase launch JAX containers')
    globals.error_list = []
    container_name = training_dict['container_name']
    # Launch the containers ..
    docker_lib.launch_docker_container( phdl, container_name,
          training_dict['container_image'], 
          training_dict['container_config']['device_list'],
          training_dict['container_config']['volume_dict'],
          training_dict['container_config']['env_dict'],
          shm_size='256G', timeout=60*20 )
    update_test_result()



    

def test_llama_3_1_70b_training(phdl, training_dict, model_params_dict, hf_token ):
    """
    Pytest: End-to-end JAX training test for Llama 3.1 FP8.
    Can be distributed or single node training based on the input json file

    Args:
      phdl: Cluster handle used to execute commands on all nodes.
      training_dict (dict): Training configuration (container, env, paths).
      model_params_dict (dict): Model/hyperparameter presets (multi-node, per GPU type).
      hf_token (str): Hugging Face access token for datasets/models.

    Behavior:
      - Resets global error list for this test.
      - Detects GPU model from rocm-smi on the head node to select the correct GPU type.
      - Creates a JaxTrainingJob for model 'llama3.1-70b' without parameter tuning overrides.
      - Performs NIC setup steps (e.g., vendor-specific tweaks) inside the container.
      - Builds training command/config and environment inside the container.
      - Starts training across nodes, polls for completion (handling errors/NaN/Inf), and verifies results.
      - Calls update_test_result() to record test outcome based on errors collected.

    """
    globals.error_list = []
    head_node = phdl.host_list[0]
    smi_out_dict = phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type=get_model_from_rocm_smi_output(smi_out)
    jx_obj = jax_training_lib.JaxTrainingJob( phdl,
           'llama3.1-70b', training_dict, model_params_dict,
           hf_token, gpu_type, distributed_training=True, tune_model_params=False )
    jx_obj.exec_nic_setup_scripts()
    jx_obj.build_training_job_cmd()
    jx_obj.start_training_job()
    jx_obj.poll_for_training_completion()
    jx_obj.verify_training_results()
    update_test_result()
