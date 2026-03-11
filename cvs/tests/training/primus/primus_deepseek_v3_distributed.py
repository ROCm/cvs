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
from cvs.lib import primus_training_lib

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
        log.error(f"Error: The file '{hf_token_file}' was not found.")
        hf_token = "dummy_token"
    except Exception as e:
        log.error(f"An error occurred reading HF token: {e}")
        hf_token = "dummy_token"
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
    log.info("Cluster configuration:")
    log.info(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.fixture(scope="module")
def gpu_type(phdl):
    """
    Provide the GPU type string for the test module.

    Args:
      phdl: Parallel SSH handle

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
    log.info(f"Detected GPU type: {gpu_type}")
    return gpu_type


@pytest.fixture(scope="module")
def prepared_c4_dataset(phdl, training_dict):
    """
    One-time C4 dataset preparation for DeepSeek training.

    This fixture checks if C4 data preparation is enabled, and if so,
    tokenizes the dataset using Primus's DeepSeekV3Tokenizer.

    The result is cached across all tests in this module.

    Args:
        phdl: Parallel SSH handle
        training_dict: Training configuration

    Returns:
        str or None: Path to tokenized data prefix, or None if data prep disabled
    """
    data_prep_config = training_dict.get('data_prep', {})

    if not data_prep_config.get('enabled', False):
        log.info("Data preparation disabled, will use mock data")
        return None

    log.info("=" * 80)
    log.info("C4 Dataset Preparation")
    log.info("=" * 80)

    tokenized_path = primus_training_lib.prepare_c4_dataset(
        phdl=phdl,
        container_name=training_dict['container_name'],
        num_shards=data_prep_config.get('num_shards', 200),
        data_dir=data_prep_config.get('data_dir', '/shared/c4'),
        tokenizer_type="DeepSeekV3Tokenizer",
        tokenizer_model="deepseek-ai/DeepSeek-V3",
        primus_path=training_dict.get('primus_path', '/workspace/Primus'),
        workers=data_prep_config.get('workers', None),
        timeout=data_prep_config.get('timeout', 3600),
    )

    log.info("=" * 80)
    log.info(f"Dataset prepared: {tokenized_path}")
    log.info("=" * 80)

    return tokenized_path


def test_disable_firewall(phdl):
    """
    Pytest: Disable firewall on all nodes to prevent connection timeouts.

    Distributed training requires all nodes to communicate freely.
    Firewalls can block the rendezvous connections needed for PyTorch distributed.

    Note: This test requires passwordless sudo. If not available, skip with:
          pytest -k "not test_disable_firewall"
    """
    globals.error_list = []

    log.info("Checking firewall status (requires passwordless sudo)...")

    # Check if user has passwordless sudo access
    test_sudo = phdl.exec('sudo -n true 2>&1')
    head_node = phdl.host_list[0]
    if 'password is required' in test_sudo.get(head_node, ''):
        log.warning("Passwordless sudo not available, skipping firewall check")
        log.info("To disable firewall manually, run on all nodes:")
        log.info("  sudo service ufw stop && sudo ufw disable")
        pytest.skip("Passwordless sudo not configured")
        return

    # Check UFW service status
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive', out_dict[node], re.I):
            log.info(f"Stopping firewall on {node}")
            phdl.exec('sudo service ufw stop')
            continue

    # Verify UFW is disabled
    out_dict = phdl.exec('sudo ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|disabled', out_dict[node], re.I):
            fail_test(f'Failed to disable firewall on node {node}')

    log.info("Firewall disabled on all nodes")
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
        test environment is isolated so this doesn't remove unrelated containers/volumes.
      - Consider narrowing cleanup scope if other workloads may be present on the hosts.
    """
    log.info("Cleaning up stale containers...")
    globals.error_list = []

    container_name = training_dict['container_name']
    docker_lib.kill_docker_container(phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(phdl)

    log.info("Cleanup complete")
    update_test_result()


def test_launch_primus_containers(phdl, training_dict):
    """
    Pytest: Launch Primus containers with AINIC networking and proper configuration.

    Args:
      phdl: Cluster handle for executing commands across nodes.
      training_dict (dict): Training configuration including:
        - 'container_name': Name for the container(s)
        - 'container_image': Primus Docker image to use
        - 'container_config': {
            'device_list': device pass-through config (GPUs, RDMA, etc),
            'volume_dict': bind mounts for datasets, logs, etc
          }
        - 'using_ainic': Whether to enable AINIC networking
        - 'nccl_ib_hca': InfiniBand HCA configuration for AINIC
        - etc.

    Behavior:
      - Initializes the global error_list for fresh test pass/fail tracking.
      - Sets up environment variables for AINIC if enabled.
      - Launches the container(s) with a large shared memory segment (shm_size='128G').
      - Uses a generous timeout (20 minutes) for image pulls/initialization.
      - Calls update_test_result() to record the outcome based on accumulated errors.
    """

    log.info('=' * 80)
    log.info('Launching Primus Containers')
    log.info('=' * 80)

    globals.error_list = []
    container_name = training_dict['container_name']
    container_image = training_dict['container_image']

    log.info(f"Container name: {container_name}")
    log.info(f"Container image: {container_image}")

    # Set up environment variables for AINIC if enabled
    env_vars = {}
    if training_dict.get('using_ainic', False):
        log.info("AINIC networking enabled")
        env_vars = {
            'USING_AINIC': '1',
            'NCCL_IB_HCA': training_dict.get('nccl_ib_hca', ''),
            'NCCL_SOCKET_IFNAME': training_dict.get('nccl_socket_ifname', 'ens9np0'),
            'GLOO_SOCKET_IFNAME': training_dict.get('gloo_socket_ifname', 'ens9np0'),
            'NCCL_IB_TIMEOUT': str(training_dict.get('nccl_ib_timeout', 23)),
            'NCCL_DEBUG': training_dict.get('nccl_debug', 'ERROR'),
        }

        # NUMA binding
        if training_dict.get('enable_numa_binding', False):
            env_vars['ENABLE_NUMA_BINDING'] = '1'

        if training_dict.get('hsa_kernarg_pool_size'):
            env_vars['HSA_KERNARG_POOL_SIZE'] = str(training_dict['hsa_kernarg_pool_size'])

    # Launch the containers
    log.info("Launching containers (this may take several minutes to pull image)...")
    docker_lib.launch_docker_container(
        phdl,
        container_name,
        container_image,
        training_dict['container_config']['device_list'],
        training_dict['container_config']['volume_dict'],
        env_dict=env_vars,
        shm_size='128G',
        timeout=60 * 20,
    )

    log.info("Containers launched successfully")
    update_test_result()


def test_deepseek_v3_distributed_training(
    phdl, gpu_type, training_dict, model_params_dict, hf_token, prepared_c4_dataset
):
    """
    Pytest: DeepSeek V3 distributed training lifecycle test using Primus.

    Args:
      phdl: Cluster handle used by the training job to execute commands.
      gpu_type (str): GPU identifier (e.g., 'mi300x') used for selecting params.
      training_dict (dict): Training configuration, volumes, environment, etc.
      model_params_dict (dict): Model/hyperparameter presets (single_node/multi_node).
      hf_token (str): Access token for Hugging Face datasets/models.
      prepared_c4_dataset (str): Path to prepared C4 dataset (or None for mock data).

    Behavior:
      - Resets the global error list for a clean test run.
      - Updates training dict with prepared data path if available.
      - Instantiates PrimusDeepSeekTrainingJob with all configurations.
      - Runs NIC setup (AINIC-specific).
      - Builds training command with MoE parameters.
      - Starts training, polls for completion, and verifies results.
      - Records final test outcome via update_test_result().
    """

    log.info('=' * 80)
    log.info('DeepSeek V3 Distributed Training Test')
    log.info('=' * 80)

    globals.error_list = []

    # Update training dict with prepared data if available
    if prepared_c4_dataset:
        training_dict['tokenized_data_path'] = prepared_c4_dataset
        training_dict['mock_data'] = False
        log.info(f"Using prepared C4 dataset: {prepared_c4_dataset}")
    else:
        training_dict['mock_data'] = True
        log.info("Using mock data for training")

    # Create Primus DeepSeek training job
    log.info("Initializing Primus DeepSeek training job...")
    ds_obj = primus_training_lib.PrimusDeepSeekTrainingJob(
        phdl,
        'deepseek_v3',
        training_dict,
        model_params_dict,
        hf_token,
        gpu_type,
        distributed_training=True,
        tune_model_params=False,
    )

    # Execute training lifecycle
    log.info("Setting up networking...")
    ds_obj.exec_nic_setup_scripts()

    log.info("Building training command...")
    ds_obj.build_training_job_cmd()

    log.info("Starting training job...")
    ds_obj.start_training_job()

    log.info("Polling for training completion...")
    ds_obj.poll_for_training_completion()

    log.info("Verifying training results...")
    ds_obj.verify_training_results()

    log.info('=' * 80)
    log.info('DeepSeek V3 Training Test Complete')
    log.info('=' * 80)

    update_test_result()
