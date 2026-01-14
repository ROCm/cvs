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


from cvs.lib import rccl_lib
from cvs.lib import html_lib
from cvs.core import OrchestratorFactory, OrchestratorConfig
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = globals.log


rccl_res_dict = {}


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

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
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
    Load and return the RCCL-specific configuration dictionary for the test module.

    Args:
      config_file (str): Path to a JSON config file provided by another fixture.

    Returns:
      dict: The value of the "rccl" key from the loaded JSON, logged for visibility.

    Notes:
      - Expects the JSON file to contain a top-level key "rccl".
      - Uses module scope so the config is parsed once per test module.
      - Consider adding validation (e.g., assert "rccl" in config) to fail fast on bad configs.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rccl']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info(config_dict)
    return config_dict


@pytest.fixture(scope="module")
def orch(cluster_file, config_file):
    """
    Create and return orchestrator instance for test execution.
    """
    log.info("Creating orchestrator from config files")

    config = OrchestratorConfig.from_configs(cluster_file, config_file)
    orch = OrchestratorFactory.create_orchestrator(log, config)
    yield orch
    # Cleanup after all tests
    orch.cleanup(orch.hosts)


# Start of test cases.


def test_collect_hostinfo(orch):
    """
    Collect basic ROCm/host info from all nodes.

    Behavior:
      - Executes common ROCm commands to capture version and agent info.
      - Does not parse output; relies on update_test_result to finalize status.

    Notes:
      - globals.error_list is reset before test (pattern used across tests).
    """

    globals.error_list = []
    orch.exec('cat /opt/rocm/.info/version')
    orch.exec('hipconfig')
    orch.exec('rocm_agent_enumerator')
    update_test_result()


def test_collect_networkinfo(orch):
    """
    Collect basic RDMA/verbs info from all nodes.

    Behavior:
      - Executes 'rdma link' and 'ibv_devinfo' to snapshot network capabilities.
      - Always executes on baremetal host regardless of orchestrator type.
      - Does not parse output; relies on update_test_result to finalize status.
    """

    globals.error_list = []
    orch.all.exec('rdma link')
    orch.all.exec('ibv_devinfo')
    update_test_result()


def test_disable_firewall(orch):
    globals.error_list = []
    # Firewall operations must run on host, not in container
    orch.all.exec('sudo service ufw stop')
    time.sleep(2)
    out_dict = orch.all.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|dead|stopped|disabled', out_dict[node], re.I):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()


@pytest.mark.parametrize(
    "rccl_collective",
    [
        "all_reduce_perf",
        "all_gather_perf",
        "scatter_perf",
        "gather_perf",
        "reduce_scatter_perf",
        "sendrecv_perf",
        "alltoall_perf",
        "alltoallv_perf",
        "broadcast_perf",
    ],
)
def test_singlenode_perf(orch, cluster_dict, config_dict, rccl_collective):
    """
    Execute RCCL performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
      - orchestrator: orchestrator instance for test execution
      - cluster_dict: cluster topology and credentials (expects node_dict, username, etc.).
      - config_dict: test configuration with RCCL/MPI paths, env, and thresholds.
      - rccl_collective: which RCCL collective test to run (e.g., "all_reduce_perf").

    Flow:
      1) Capture start time to bound dmesg checks later.
      2) Optionally snapshot cluster metrics before the test (for debugging/compare).
      3) Optionally source environment script if provided in config.
      4) Invoke rccl_lib.rccl_cluster_test with parameters built from config and fixtures.
      5) Capture end time and verify dmesg for errors between start/end.
      6) Optionally snapshot metrics again and compare before/after.
      7) Call update_test_result() to finalize test status.

    Notes:
      - cluster_snapshot_debug controls whether before/after snapshots are taken.
    """

    # Log a message to Dmesg to create a timestamp record
    orch.exec(f'sudo echo "Starting Test singlenode {rccl_collective}" | sudo tee /dev/kmsg')

    # start_time = orch.exec('date')
    start_time = orch.exec('date +"%a %b %e %H:%M"')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    # Get cluster snapshot ..
    if re.search('True', config_dict['cluster_snapshot_debug'], re.I):
        cluster_dict_before = create_cluster_metrics_snapshot(orch)

    # Optionally source environment (e.g., set MPI/ROCm env) before running RCCL tests
    if not re.search('None', config_dict['env_source_script'], re.I):
        orch.exec(f"bash {config_dict['env_source_script']}")

    # Execute the RCCL cluster test with parameters sourced from config_dict
    result_dict = rccl_lib.rccl_single_node_test(
        orch,
        test_name=rccl_collective,
        cluster_node_list=node_list,
        rocm_path_var=config_dict['rocm_path_var'],
        rccl_dir=config_dict['rccl_dir'],
        rccl_path_var=config_dict['rccl_path_var'],
        rccl_tests_dir=config_dict['rccl_tests_dir'],
        start_msg_size=config_dict['start_msg_size'],
        end_msg_size=config_dict['end_msg_size'],
        step_function=config_dict['step_function'],
        warmup_iterations=config_dict['warmup_iterations'],
        no_of_iterations=config_dict['no_of_iterations'],
        check_iteration_count=config_dict['check_iteration_count'],
        debug_level=config_dict['debug_level'],
        rccl_result_file=config_dict['rccl_result_file'],
        no_of_local_ranks=config_dict['no_of_local_ranks'],
        verify_bus_bw=config_dict['verify_bus_bw'],
        verify_bw_dip=config_dict['verify_bw_dip'],
        verify_lat_dip=config_dict['verify_lat_dip'],
        exp_results_dict=config_dict['results'],
        env_source_script=config_dict['env_source_script'],
    )

    print(result_dict)
    key_name = f'{rccl_collective}'
    rccl_res_dict[key_name] = result_dict

    # Scan dmesg between start and end times cluster wide ..
    # end_time = orch.exec('date')
    orch.exec(f'sudo echo "End of Test singlenode {rccl_collective}" | sudo tee /dev/kmsg')

    end_time = orch.exec('date +"%a %b %e %H:%M"')
    verify_dmesg_for_errors(orch, start_time, end_time, till_end_flag=True)

    # Get new cluster snapshot and compare ..
    if re.search('True', config_dict['cluster_snapshot_debug'], re.I):
        cluster_dict_after = create_cluster_metrics_snapshot(orch)
        compare_cluster_metrics_snapshots(cluster_dict_before, cluster_dict_after)

    # Update test results based on any failures ..
    update_test_result()


def test_gen_graph():
    print('Final Global result dict')
    print(rccl_res_dict)
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    print(rccl_graph_dict)

    proc_id = os.getpid()

    html_file = f'/tmp/rccl_singlenode_perf_report_{proc_id}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_default_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)

    print(f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view')
