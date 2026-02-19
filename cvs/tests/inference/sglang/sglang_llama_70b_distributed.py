'''
Copyright 2025 Advanced Micro Devices, Inc.
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
from cvs.lib import sglang_disagg_lib
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
def inference_config_file(pytestconfig):
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
def inference_dict(inference_config_file, cluster_dict):
    with open(inference_config_file) as json_file:
        inference_dict_t = json.load(json_file)
    inference_dict = inference_dict_t['config']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    inference_dict = resolve_test_config_placeholders(inference_dict, cluster_dict)
    return inference_dict


@pytest.fixture(scope="module")
def benchmark_params_dict(inference_config_file, cluster_dict):
    with open(inference_config_file) as json_file:
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
def p_phdl(cluster_dict, inference_dict):
    print(cluster_dict)
    p_phdl = Pssh(
        log, inference_dict['prefill_node_list'], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file']
    )
    return p_phdl


@pytest.fixture(scope="module")
def d_phdl(cluster_dict, inference_dict):
    d_phdl = Pssh(
        log, inference_dict['decode_node_list'], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file']
    )
    return d_phdl


@pytest.fixture(scope="module")
def r_phdl(cluster_dict, inference_dict):
    node_list = []
    node_list.append(inference_dict['proxy_router_node'])
    r_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return r_phdl


@pytest.fixture(scope="module")
def b_phdl(cluster_dict, inference_dict):
    node_list = []
    node_list.append(inference_dict['benchmark_serv_node'])
    b_phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return b_phdl


@pytest.fixture(scope="module")
def gpu_type(p_phdl, cluster_dict):
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

    print(p_phdl)
    head_node = p_phdl.host_list[0]
    smi_out_dict = p_phdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    gpu_type = get_model_from_rocm_smi_output(smi_out)
    return gpu_type


def test_cleanup_stale_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):
    """
    Pytest: Clean up potentially stale Docker containers and volumes before tests.

    Notes:
      - This performs a broad cleanup via delete_all_containers_and_volumes; ensure the
        test environment is isolated so this doesn?t remove unrelated containers/volumes.
      - Consider narrowing cleanup scope if other workloads may be present on the hosts.
    """

    container_name = inference_dict['container_name']
    for a_phdl in [p_phdl, d_phdl, r_phdl, b_phdl]:
        docker_lib.kill_docker_container(a_phdl, container_name)
        docker_lib.delete_all_containers_and_volumes(a_phdl)

    # Cleanup log directory from one of the nodes
    print('Cleaning up log directory')
    r_phdl.exec(f"sudo rm -rf {inference_dict['log_dir']}")
    time.sleep(5)


def test_launch_inference_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):
    log.info('Testcase launch InferenceMax containers')
    globals.error_list = []
    container_name = inference_dict['container_name']
    # Launch the containers ..
    hdl_list = [p_phdl, d_phdl]
    # Users can use the one of the prefill, decode nodes as proxy, benchmark, so
    # check before scheduling
    if inference_dict['proxy_router_node'] == inference_dict['benchmark_serv_node']:
        if (inference_dict['proxy_router_node'] in inference_dict['prefill_node_list']) or (
            inference_dict['proxy_router_node'] in inference_dict['decode_node_list']
        ):
            print('Already part of the handle list, no need to add')
        else:
            hdl_list.extend(r_phdl)
    else:
        if (inference_dict['proxy_router_node'] in inference_dict['prefill_node_list']) or (
            inference_dict['proxy_router_node'] in inference_dict['decode_node_list']
        ):
            print('Already part of the handle list, no need to add')
        else:
            hdl_list.extend(r_phdl)
        if (inference_dict['benchmark_serv_node'] in inference_dict['prefill_node_list']) or (
            inference_dict['benchmark_serv_node'] in inference_dict['decode_node_list']
        ):
            print('Already part of the handle list, no need to add')
        else:
            hdl_list.extend(b_phdl)

    for a_phdl in hdl_list:
        docker_lib.launch_docker_container(
            a_phdl,
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
    for a_phdl in [p_phdl, d_phdl, r_phdl, b_phdl]:
        out_dict = a_phdl.exec('docker ps')
        for node in out_dict.keys():
            if not re.search(f'{container_name}', out_dict[node], re.I):
                fail_test(f'Failed to launch container on node {node}')
    update_test_result()


# Setup the ib devices and ensure they show up in the container
def test_setup_ibv_devices(im_obj):
    """
    Validate that InfiniBand / RDMA devices are:
      - Properly configured on the host
      - Visible inside the inference container
      - Ready for high-performance communication (Prefill/Decode)

    This test is foundational and must pass before any inference tests
    that rely on RDMA-based KV cache transfer.
    """
    globals.error_list = []
    im_obj.check_ibv_devices()
    im_obj.exec_nic_setup_scripts()
    update_test_result()


# Create the SGlang Inference Object Fixture
@pytest.fixture(scope="module")
def im_obj(p_phdl, d_phdl, r_phdl, b_phdl, gpu_type, inference_dict, benchmark_params_dict, hf_token):
    globals.error_list = []
    bp_dict = benchmark_params_dict['llama-70b']
    im_obj = sglang_disagg_lib.SglangDisaggPD(
        bp_dict['model'], inference_dict, bp_dict, hf_token, p_phdl, d_phdl, r_phdl, b_phdl, gpu_type
    )
    return im_obj


def test_rms_norm(im_obj):
    """
    Run RMSNorm operator tests to validate:
      - GPU kernel correctness
      - AITer backend functionality
      - Basic compute stability before inference

    This serves as a low-level sanity check before launching servers.
    """
    globals.error_list = []
    im_obj.run_test_rmsnorm()
    update_test_result()


# Test to start the prefill servers using sglang.launch_server
def test_launch_prefill_servers(im_obj):
    """
    Start SGLang Prefill servers in disaggregated PD mode.

    Prefill servers:
      - Handle prompt processing
      - Generate KV cache
      - Serve as upstream for Decode servers

    This test prepares the Prefill side of the inference pipeline.
    """
    globals.error_list = []
    im_obj.setup_prefill_container_env()
    im_obj.launch_prefill_servers()
    update_test_result()


# Test to start the decode servers using sglang.launch_server
def test_launch_decode_servers(im_obj):
    """
    Start SGLang Decode servers in disaggregated PD mode.

    Decode servers:
      - Consume KV cache from Prefill servers
      - Generate output tokens
      - Drive decode throughput and latency

    This test completes the inference data plane.
    """
    globals.error_list = []
    im_obj.setup_decode_container_env()
    im_obj.launch_decode_servers()
    update_test_result()


# Test to validate the Prefill and Decode servers are ready to serve
# Inference traffic
def test_poll_for_server_ready(im_obj):
    """
    Poll Prefill and Decode server logs to ensure:
      - Servers have fully started
      - Models are loaded
      - HTTP endpoints are responding (200 OK)
      - Systems are stable before inference begins

    This test prevents inference traffic from being sent too early.
    """
    globals.error_list = []
    im_obj.poll_and_check_server_ready()
    update_test_result()


# Start the proxy router serving using sglang_router.launch_router
def test_launch_proxy_router(im_obj):
    """
    Start the SGLang Proxy Router.

    The Proxy Router:
      - Accepts inference requests
      - Routes Prefill and Decode traffic
      - Coordinates disaggregated PD execution

    This test completes the control plane setup for inference serving.
    """
    globals.error_list = []
    im_obj.setup_proxy_router_container_env()
    im_obj.launch_proxy_router()
    update_test_result()


# Test to run the canned gsm8k benchmark packaged with the container
def test_run_gsm8k_benchmark_test(im_obj):
    """
    Execute the GSM8K benchmark using the SGLang inference serving stack.

    Purpose:
    --------
    This test validates:
      - End-to-end inference correctness using a real-world dataset
      - Sustained decode throughput under realistic query patterns
      - Proper interaction between Proxy Router, Prefill, and Decode servers

    GSM8K is a commonly used reasoning benchmark, making it a strong
    signal for both correctness and performance regression detection.
    """
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    im_obj.run_gsm8k_benchmark_test()
    update_test_result()


# Test to run the sglang Benchmarking Testing using bench_serv
def test_run_benchmark_test(im_obj):
    """
    Execute a synthetic serving benchmark using sglang.bench_serving
    with a random dataset.

    Purpose:
    --------
    This test focuses on:
      - Stress-testing the serving infrastructure
      - Evaluating scheduling, batching, and throughput under load
      - Isolating serving performance independent of dataset semantics

    Randomized workloads are useful for detecting performance regressions
    and scaling bottlenecks.
    """
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    im_obj.benchserv_test_random(d_type='auto')
    update_test_result()
