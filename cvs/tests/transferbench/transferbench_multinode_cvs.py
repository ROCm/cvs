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
import itertools

from cvs.lib import transferbench_lib
from cvs.lib import html_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


transferbench_res_dict = {}


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --cluster_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the cluster configuration file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration JSON file passed via pytest CLI.

    For multi-node TransferBench testing, use:
    --config_file cvs/input/config_file/transferbench/transferbench_multinode_config.json

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def  cluster_dict(cluster_file):
     """
    Load and expose full cluster configuration for the test module.

    Behavior:
      - Opens the JSON at cluster_file and parses it into a Python dict.
      - Logs the parsed dictionary for visibility and debugging.
      - Returns the entire cluster configuration (node list, credentials, etc.).

    Args:
      cluster_file (str): Path to the cluster configuration JSON.

    Returns:
      dict: Parsed cluster configuration. Expected keys include:
            - 'node_dict': Map of node name -> node metadata
            - 'username': SSH username
            - 'priv_key_file': Path to SSH private key
    """
     with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

     # Resolve path placeholders like {user-id} in cluster config
     cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
     log.info(cluster_dict)
     return cluster_dict




@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and return the TransferBench-specific configuration dictionary for the test module.
    """
    with open(config_file) as json_file:
       config_dict_t = json.load(json_file)
    config_dict = config_dict_t['transferbench']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info(config_dict)
    return config_dict




@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing:
        - node_dict: dict of node_name -> node_details
        - username: SSH username
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: Handle configured for all nodes (for broadcast/parallel operations).

    Notes:
      - Prints the cluster_dict for quick debugging; consider replacing with log.debug.
      - Module-scoped so a single shared handle is used across all tests in the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
      - Assumes Pssh(log, node_list, user=..., pkey=...) is available in scope.
    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return shdl




@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    """
    Collect and return a list of VPC IPs for all nodes in the cluster.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing node_dict with vpc_ip per node.

    Returns:
      list[str]: List of VPC IP addresses in the cluster, ordered by node_dict iteration.

    Notes:
      - Iteration order depends on the insertion order of node_dict.
      - Consider validating that each node entry contains a 'vpc_ip' key.
    """
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])
    return vpc_node_list



def pytest_generate_tests(metafunc):

    """
    Dynamically parametrize TransferBench-related tests based on a JSON config file.

    Behavior:
      - Reads the config file path from pytest's --config_file option.
      - Loads the JSON and extracts transferbench parameters:
          * transfer_scale: list of transfer counts to test
          * executor_scale: list of executor counts to test  
          * source_executor_destination: list of memory transfer paths to test
          * start_num_bytes/end_num_bytes/step_function: for generating byte sizes
      - Generates all combinations of transfer_scale × executor_scale × source_executor_destination
      - Generates num_bytes list from start/end/step parameters
      - If a corresponding fixture name is present in a test function, applies
        parametrize with the built list.

    Notes:
      - If no config_file is provided, the hook returns without parametrizing.
      - Defaults are used when keys are absent under config['transferbench'].
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        print(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)
    transferbench = cfg.get("transferbench", {})

    # Get the three arrays for generating transfer configs
    transfer_scale_list = transferbench.get("transfer_scale", [1, 2, 4, 8])
    executor_scale_list = transferbench.get("executor_scale", [1, 2, 4])
    source_executor_destination_list = transferbench.get("source_executor_destination", [
        "G0->G0->G1",  # GPU-to-GPU via GPU executor
        "C0->D0->G0",  # CPU-to-GPU via DMA executor
        "G0->N0->G1",  # GPU-to-GPU via NIC executor
    ])

    # Generate all combinations of transfer_scale × executor_scale × source_executor_destination
    transfer_config_list = []
    for transfer_scale in transfer_scale_list:
        for executor_scale in executor_scale_list:
            for sed in source_executor_destination_list:
                transfer_config_list.append(f"{transfer_scale} {executor_scale} ({sed})")

    # Generate num_bytes list from start/end/step parameters
    start_num_bytes = transferbench.get("start_num_bytes", "1024")
    end_num_bytes = transferbench.get("end_num_bytes", "16g")
    step_function = int(transferbench.get("step_function", "2"))

    # Convert start and end to bytes
    def parse_size(size_str):
        size_str = size_str.lower()
        if size_str.endswith('g'):
            return int(size_str[:-1]) * 1024**3
        elif size_str.endswith('m'):
            return int(size_str[:-1]) * 1024**2
        elif size_str.endswith('k'):
            return int(size_str[:-1]) * 1024
        else:
            return int(size_str)

    start_bytes = parse_size(start_num_bytes)
    end_bytes = parse_size(end_num_bytes)

    # Generate geometric progression
    num_bytes_list = []
    current = start_bytes
    while current <= end_bytes:
        num_bytes_list.append(str(current))
        current *= step_function

    # Only parametrize fixtures used by this test
    all_keys = ("transfer_config", "num_bytes")

    active = [k for k in all_keys if k in metafunc.fixturenames]
    if not active:
        return

    domain_by_key = {
        "transfer_config": transfer_config_list,
        "num_bytes": num_bytes_list,
    }
    domains = [domain_by_key[k] for k in active]

    params, ids = [], []
    for values in itertools.product(*domains):
        combo = dict(zip(active, values))

        params.append(values)

        ids.append("|".join(f"{k}={combo[k]}" for k in active))
    metafunc.parametrize(",".join(active), params, ids=ids)




# Start of test cases.

def test_collect_hostinfo( phdl ):

    """
    Collect basic ROCm/host info from all nodes.

    Behavior:
      - Executes common ROCm commands to capture version and agent info.
      - Does not parse output; relies on update_test_result to finalize status.

    Notes:
      - globals.error_list is reset before test (pattern used across tests).
    """

    globals.error_list = []
    phdl.exec('cat /opt/rocm/.info/version')
    phdl.exec('hipconfig')
    phdl.exec('rocm_agent_enumerator')
    update_test_result()



def test_collect_networkinfo( phdl ):

    """
    Collect basic RDMA/verbs info from all nodes.

    Behavior:
      - Executes 'rdma link' and 'ibv_devinfo' to snapshot network capabilities.
      - Does not parse output; relies on update_test_result to finalize status.
    """

    globals.error_list = []
    phdl.exec('rdma link')
    phdl.exec('ibv_devinfo')
    update_test_result()



def test_disable_firewall( phdl ):
    globals.error_list = []
    phdl.exec('sudo service ufw stop')
    time.sleep(2)
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search( 'inactive|dead|stopped|disabled', out_dict[node], re.I ):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()



def test_transferbench_perf(phdl, shdl, cluster_dict, config_dict, transfer_config, num_bytes ):

    """
    Execute TransferBench performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
      - phdl: parallel execution handle for nodes (expects exec/exec_cmd_list).
      - shdl: ssh handle to the first node in the cluster.
      - cluster_dict: cluster topology and credentials.
      - config_dict: test configuration with TransferBench/MPI paths, env, and thresholds.
      - transfer_config: TransferBench transfer configuration string (e.g., "1 4 (G0->G0->G1)").
      - num_bytes: Number of bytes to transfer (0 for sweep).

    Flow:
      1) Capture start time to bound dmesg checks later.
      2) Optionally source environment script if provided in config.
      3) Invoke transferbench_lib.transferbench_cluster_test with parameters built from config and fixtures.
      4) Capture end time and verify dmesg for errors between start/end.
      5) Optionally snapshot metrics again and compare before/after.
      6) Call update_test_result() to finalize test status.

    Notes:
      - cluster_snapshot_debug controls whether before/after snapshots are taken.
    """

    # Log a message to Dmesg to create a timestamp record
    phdl.exec( f'sudo echo "Starting TransferBench Test {transfer_config}" | sudo tee /dev/kmsg' )

    start_time = phdl.exec('date +"%a %b %e %H:%M"')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    # Build list of nodes and their VPC IPs
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    #Get cluster snapshot ..
    if re.search( 'True', config_dict['cluster_snapshot_debug'], re.I ):
        cluster_dict_before = create_cluster_metrics_snapshot( phdl )

    # Optionally source environment (e.g., set MPI/ROCm env) before running TransferBench tests
    if not re.search( 'None', config_dict['env_source_script'], re.I ):
        phdl.exec(f"bash {config_dict['env_source_script']}")

    # Execute the TransferBench cluster test with parameters sourced from config_dict
    result_dict = transferbench_lib.transferbench_cluster_test( phdl, shdl, \
       test_name               = transfer_config, \
       cluster_node_list       = node_list, \
       vpc_node_list           = vpc_node_list, \
       user_name               = cluster_dict['username'], \
       transfer_configs        = [transfer_config], \
       num_bytes               = num_bytes, \
       rocm_path_var           = config_dict['rocm_path_var'], \
       mpi_dir                 = config_dict['mpi_dir'], \
       mpi_path_var            = config_dict['mpi_path_var'], \
       transferbench_dir       = config_dict['transferbench_dir'], \
       transferbench_path_var  = config_dict['transferbench_path_var'], \
       transferbench_bin_dir   = config_dict['transferbench_bin_dir'], \
       num_iterations          = config_dict['num_iterations'], \
       num_warmups             = config_dict['num_warmups'], \
       transferbench_result_file = config_dict['transferbench_result_file'], \
       verify_bus_bw           = config_dict['verify_bus_bw'], \
       verify_bw_dip           = config_dict['verify_bw_dip'], \
       verify_lat_dip          = config_dict['verify_lat_dip'], \
       exp_results_dict        = config_dict['results']
    )

    print(result_dict)
    key_name = f'{transfer_config}-{num_bytes}'
    transferbench_res_dict[key_name] = result_dict

    # Scan dmesg between start and end times cluster wide ..
    phdl.exec( f'sudo echo "End of TransferBench Test {transfer_config}" | sudo tee /dev/kmsg' )

    end_time = phdl.exec('date +"%a %b %e %H:%M"')
    verify_dmesg_for_errors( phdl, start_time, end_time, till_end_flag=True )

    # Get new cluster snapshot and compare ..
    if re.search( 'True', config_dict['cluster_snapshot_debug'], re.I ):
        cluster_dict_after = create_cluster_metrics_snapshot( phdl )
        compare_cluster_metrics_snapshots( cluster_dict_before, cluster_dict_after )

    # Update test results based on any failures ..
    update_test_result()





def test_gen_graph():
    print('Final Global TransferBench result dict')
    print(transferbench_res_dict)
    transferbench_graph_dict = transferbench_lib.convert_transferbench_to_graph_dict(transferbench_res_dict)
    print(transferbench_graph_dict)

    proc_id = os.getpid()

    html_file = f'/tmp/transferbench_perf_report_{proc_id}.html'

    html_lib.add_html_begin( html_file )
    html_lib.build_rccl_amcharts_graph( html_file, 'transferbench', transferbench_graph_dict )
    html_lib.insert_chart( html_file, 'transferbench' )
    html_lib.build_rccl_result_table( html_file, transferbench_graph_dict )
    html_lib.add_json_data( html_file, json.dumps(transferbench_graph_dict) )
    html_lib.add_html_end( html_file )

    print(f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view')