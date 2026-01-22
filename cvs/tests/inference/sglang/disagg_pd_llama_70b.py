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
    p_phdl = Pssh(log, inference_dict['prefill_node_list'], user=cluster_dict['username'], \
            pkey=cluster_dict['priv_key_file'])
    return p_phdl


@pytest.fixture(scope="module")
def d_phdl(cluster_dict, inference_dict):
    d_phdl = Pssh(log, inference_dict['decode_node_list'], user=cluster_dict['username'], \
            pkey=cluster_dict['priv_key_file'])
    return d_phdl


@pytest.fixture(scope="module")
def r_phdl(cluster_dict, inference_dict):
    node_list = []
    node_list.append(inference_dict['proxy_router_node'])
    r_phdl = Pssh(log, node_list, user=cluster_dict['username'], \
            pkey=cluster_dict['priv_key_file'])
    return r_phdl



@pytest.fixture(scope="module")
def b_phdl(cluster_dict, inference_dict):
    node_list = []
    node_list.append(inference_dict['benchmark_serv_node'])
    b_phdl = Pssh(log, node_list, user=cluster_dict['username'], \
            pkey=cluster_dict['priv_key_file'])
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



# Create the SGlang Inference Object Fixture
@pytest.fixture(scope="module")
def im_obj( p_phdl, d_phdl, r_phdl, b_phdl, gpu_type, inference_dict, \
        benchmark_params_dict, hf_token):
    globals.error_list = []
    bp_dict = benchmark_params_dict['llama-70b']
    im_obj = sglang_disagg_lib.SglangDisaggPD(
        bp_dict['model'],
        inference_dict,
        bp_dict,
        hf_token,
        p_phdl,
        d_phdl,
        r_phdl,
        b_phdl,
        gpu_type
    )
    return im_obj


def test_cleanup_stale_containers(p_phdl, d_hdl, r_hdl, inference_dict):
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
    # Cleanup log directory
    print('Cleaning up log directory')
    r_hdl.exec(f'sudo rm -rf {self.log_dir}')



def test_cleanup_stale_containers( p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):

    for a_phdl in [ p_phdl, d_phdl, r_phdl ]:
        container_name = inference_dict['container_name']
        docker_lib.kill_docker_container(a_phdl, container_name)
        docker_lib.delete_all_containers_and_volumes(a_phdl)





def test_launch_inference_containers(p_phdl, d_phdl, r_phdl, b_phdl, inference_dict):
    log.info('Testcase launch InferenceMax containers')
    globals.error_list = []
    container_name = inference_dict['container_name']
    # Launch the containers ..
    if inference_dict['proxy_router_node'] == inference_dict['benchmark_serv_node']:
        hdl_list = [ p_phdl, d_phdl, r_phdl ]
    else:
        hdl_list = [ p_phdl, d_phdl, r_phdl, b_phdl ]

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
    for a_phdl in [ p_phdl, d_phdl, r_phdl, b_phdl ]:
        out_dict = a_phdl.exec('docker ps')
        for node in out_dict.keys():
            if not re.search(f'{container_name}', out_dict[node], re.I):
                fail_test(f'Failed to launch container on node {node}')
    update_test_result()




# Setup the ib devices and ensure they show up in the container
def test_setup_ibv_devices( im_obj ):
    globals.error_list = []
    im_obj.check_ibv_devices()
    im_obj.exec_nic_setup_scripts()
    update_test_result()



def test_rms_norm( im_obj ):
    globals.error_list = []
    im_obj.run_test_rmsnorm()
    update_test_result()



# Test to start the prefill servers using sglang.launch_server
def test_launch_prefill_servers( im_obj ):
    globals.error_list = []
    im_obj.setup_prefill_container_env()
    im_obj.launch_prefill_servers()
    update_test_result()


# Test to start the decode servers using sglang.launch_server
def test_launch_decode_servers( im_obj ):
    globals.error_list = []
    im_obj.setup_decode_container_env()
    im_obj.launch_decode_servers()
    update_test_result()


# Test to validate the Prefill and Decode servers are ready to serve 
# Inference traffic
def test_poll_for_server_ready( im_obj ):
    globals.error_list = []
    im_obj.poll_and_check_server_ready()
    update_test_result()


# Start the proxy router serving using sglang_router.launch_router
def test_launch_proxy_router( im_obj ):
    globals.error_list = []
    im_obj.setup_proxy_router_container_env()
    im_obj.launch_proxy_router()
    update_test_result()


# Test to run the canned gsm8k benchmark packaged with the container
def test_run_gsm8k_benchmark_test( im_obj ):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    im_obj.run_gsm8k_benchmark_test()
    update_test_result()



# Test to run the sglang Benchmarking Testing using bench_serv
def test_run_benchmark_test( im_obj ):
    globals.error_list = []
    im_obj.setup_benchmark_serv_container_env()
    im_obj.benchserv_test_random( d_type='auto' )
    update_test_result()
