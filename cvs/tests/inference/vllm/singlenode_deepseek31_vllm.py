'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import os
import time
import json

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib import docker_lib
from cvs.lib.inference_lib import InferenceJobFactory
from cvs.lib import globals

log = globals.log

# Model name for this test suite
MODEL_NAME = "deepseek-v31"


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


def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize inference tests based on sequence combinations and concurrency levels
    for the DeepSeek-V3.1 model.

    Behavior:
      - Reads the config file path from pytest's --config_file option.
      - Loads the JSON and extracts deepseek-v31 model configuration.
      - Extracts sequence_combinations (ISL/OSL pairs) and concurrency_levels.
      - Creates test cases for each sequence combination × concurrency level.

    Test Matrix:
      - Test IDs: "combination_name-concX" (e.g., "balanced-conc16")

    Notes:
      - If no config_file is provided, the hook returns without parametrizing.
      - Each combination gets a separate test case with clear ID.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        print(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)

    # Extract deepseek-v31 model config (now directly under benchmark_params, no single_node nesting)
    benchmark_params = cfg.get("benchmark_params", {})
    model_config = benchmark_params.get(MODEL_NAME, {})

    if not model_config:
        print(f'Warning: Model {MODEL_NAME} not found in config')
        return

    # Build test parameters: list of (seq_combo_dict, concurrency, test_id)
    test_params = []

    # Check if model uses sequence_combinations or legacy ISL/OSL
    seq_combos = model_config.get("sequence_combinations", [])

    if seq_combos:
        # New format: multiple combinations per model
        for combo in seq_combos:
            combo_name = combo.get("name", f"isl{combo['isl']}_osl{combo['osl']}")

            # Check if model has concurrency_levels array or single max_concurrency
            conc_levels = model_config.get("concurrency_levels", [])
            if conc_levels:
                # Parametrize across concurrency levels
                for conc in conc_levels:
                    test_id = f"{combo_name}-conc{conc}"
                    test_params.append((combo, conc, test_id))
            else:
                # Backward compatibility: use max_concurrency as single value
                max_conc = int(model_config.get("max_concurrency", "64"))
                test_id = f"{combo_name}"
                test_params.append((combo, max_conc, test_id))
    else:
        # Legacy format: single ISL/OSL values
        isl = model_config.get("input_sequence_length", "1024")
        osl = model_config.get("output_sequence_length", "1024")
        combo = {"isl": isl, "osl": osl, "name": "default"}

        conc_levels = model_config.get("concurrency_levels", [])
        if conc_levels:
            for conc in conc_levels:
                test_id = f"conc{conc}"
                test_params.append((combo, conc, test_id))
        else:
            max_conc = int(model_config.get("max_concurrency", "64"))
            test_params.append((combo, max_conc, "default"))

    # Parametrize if test uses these fixtures
    if "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames:
        if test_params:
            combos, concs, ids = zip(*test_params)
            metafunc.parametrize("seq_combo,concurrency", list(zip(combos, concs)), ids=ids)


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
    Create and return a parallel SSH handle for all cluster nodes (server).

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
    s_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return s_phdl


@pytest.fixture(scope="module")
def c_phdl(cluster_dict):
    """
    Create and return a parallel SSH handle for all cluster nodes (client).

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
    c_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return c_phdl


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
        test environment is isolated so this doesn't remove unrelated containers/volumes.
      - Consider narrowing cleanup scope if other workloads may be present on the hosts.
    """

    container_name = inference_dict['container_name']
    docker_lib.kill_docker_container(s_phdl, container_name)
    docker_lib.delete_all_containers_and_volumes(s_phdl)


def test_launch_inference_containers(s_phdl, inference_dict, benchmark_params_dict):
    """
    Launch vLLM inference containers on all nodes.

    Note: Container image can be model-specific or use global default.
    """

    log.info(f'Testcase launch vLLM containers for {MODEL_NAME}')
    globals.error_list = []
    container_name = inference_dict['container_name']

    # Get model-specific container image or use global default
    container_image = benchmark_params_dict.get(MODEL_NAME, {}).get(
        'container_image', inference_dict['container_image']
    )

    # Launch the containers ..
    docker_lib.launch_docker_container(
        s_phdl,
        container_name,
        container_image,
        inference_dict['container_config']['device_list'],
        inference_dict['container_config']['volume_dict'],
        inference_dict['container_config']['env_dict'],
        shm_size='16G',
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


def test_vllm_inference(c_phdl, s_phdl, inference_dict, benchmark_params_dict, hf_token, seq_combo, concurrency):
    """
    Test vLLM inference for DeepSeek-V3.1 with specific sequence combination and concurrency level.

    This test is parametrized via pytest_generate_tests to run once per:
      - Sequence combination (ISL/OSL pair) defined in model's sequence_combinations
      - Concurrency level defined in model's concurrency_levels

    The factory will automatically create the correct VllmJob instance with
    the model-specific container image and parameters.

    Args:
        seq_combo: Dict with 'isl', 'osl', 'name' keys for this test iteration
        concurrency: Integer concurrency level for this test iteration
    """
    # Since this is a per-GPU-type config file (mi355x), gpu_type is implicit
    gpu_type = "mi355x"

    log.info(
        f"Starting inference test for model: {MODEL_NAME}, GPU: {gpu_type}, combination: {seq_combo['name']} (ISL={seq_combo['isl']}, OSL={seq_combo['osl']}), concurrency: {concurrency}"
    )
    globals.error_list = []

    # Override ISL/OSL and concurrency in benchmark_params for this specific test iteration
    # Config is now fully flattened, so access directly under benchmark_params
    model_params = benchmark_params_dict[MODEL_NAME]
    model_params['input_sequence_length'] = seq_combo['isl']
    model_params['output_sequence_length'] = seq_combo['osl']
    model_params['max_concurrency'] = str(concurrency)

    # Calculate num_prompts based on OSL (matching recipe logic)
    osl = int(seq_combo['osl'])
    if osl == 8192:
        model_params['num_prompts'] = str(concurrency * 20)
    else:
        model_params['num_prompts'] = str(concurrency * 50)

    # Create job using factory - it will auto-detect vLLM from config
    vllm_job = InferenceJobFactory.create_job(
        inference_dict,
        c_phdl,
        s_phdl,
        MODEL_NAME,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        gpu_type,
        distributed_inference=False,
    )

    # Build and execute inference job
    vllm_job.build_server_inference_job_cmd()
    vllm_job.start_inference_server_job()
    vllm_job.start_inference_client_job()
    vllm_job.poll_for_inference_completion()
    vllm_job.verify_inference_results()
    update_test_result()

    log.info(
        f"Completed inference test for model: {MODEL_NAME}, GPU: {gpu_type}, combination: {seq_combo['name']}, concurrency: {concurrency}"
    )
