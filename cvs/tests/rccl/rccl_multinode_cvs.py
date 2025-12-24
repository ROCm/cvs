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
import itertools

from cvs.lib import rccl_lib
from cvs.lib import html_lib
from cvs.lib.parallel_ssh_lib import *
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
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
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
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
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
    Dynamically parametrize RCCL-related tests based on a JSON config file.

    Behavior:
      - Reads the config file path from pytest's --config_file option.
      - Loads the JSON and extracts rccl parameters:
          * rccl_collective: list of collectives to test (defaults provided)
          * rccl_algo: list of algorithms (defaults: ["ring", "tree"])
          * rccl_protocol: list of protocols (defaults: ["simple", "LL128", "LL"])
          * qp_scale: list of queue pair scaling values (defaults: ["1", "2"])
      - If a corresponding fixture name is present in a test function, applies
        parametrize with the built list.

    Notes:
      - If no config_file is provided, the hook returns without parametrizing.
      - Defaults are used when keys are absent under config['rccl'].
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        print(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)
    rccl = cfg.get("rccl", {})

    # Defaults (dedup'd)
    rccl_collective_list = rccl.get(
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
    rccl_algo_list = rccl.get("rccl_algo", ["ring", "tree"])
    rccl_protocol_list = rccl.get("rccl_protocol", ["simple", "LL128", "LL"])
    qp_scale_list = rccl.get("qp_scale", ["1", "2"])
    nccl_pxn_disable_list = rccl.get("nccl_pxn_disable", ["1", "0"])

    # Only parametrize fixtures used by this test
    all_keys = ("rccl_collective", "rccl_algo", "rccl_protocol", "qp_scale", "nccl_pxn_disable")

    active = [k for k in all_keys if k in metafunc.fixturenames]
    if not active:
        return

    domain_by_key = {
        "rccl_collective": rccl_collective_list,
        "rccl_algo": rccl_algo_list,
        "rccl_protocol": rccl_protocol_list,
        "qp_scale": qp_scale_list,
        "nccl_pxn_disable": nccl_pxn_disable_list,
    }
    domains = [domain_by_key[k] for k in active]

    params, ids = [], []
    for values in itertools.product(*domains):
        combo = dict(zip(active, values))

        if combo.get("rccl_algo") == "tree":
            if combo.get("rccl_collective") != "all_reduce_perf":
                continue

        params.append(values)

        ids.append("|".join(f"{k}={combo[k]}" for k in active))
    metafunc.parametrize(",".join(active), params, ids=ids)


# Start of test cases.


def test_collect_hostinfo(phdl):
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


def test_collect_networkinfo(phdl):
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


def test_disable_firewall(phdl):
    globals.error_list = []
    phdl.exec('sudo service ufw stop')
    time.sleep(2)
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|dead|stopped|disabled', out_dict[node], re.I):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()


def test_rccl_perf(
    phdl, shdl, cluster_dict, config_dict, rccl_collective, rccl_algo, rccl_protocol, qp_scale, nccl_pxn_disable
):
    """
    Execute RCCL performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
      - phdl: parallel execution handle for nodes (expects exec/exec_cmd_list).
      - shdl: switch or auxiliary handle used by rccl_lib (implementation-specific).
      - cluster_dict: cluster topology and credentials (expects node_dict, username, etc.).
      - config_dict: test configuration with RCCL/MPI paths, env, and thresholds.
      - rccl_collective: which RCCL collective test to run (e.g., "all_reduce_perf").
      - rccl_algo: RCCL algorithm (e.g., "ring", "tree").
      - rccl_protocol: RCCL protocol (e.g., "simple", "LL128", "LL").
      - qp_scale: Queue pair scaling factor as a string (e.g., "1", "2").

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
    phdl.exec(f'sudo echo "Starting Test {rccl_collective} {rccl_algo} {rccl_protocol}" | sudo tee /dev/kmsg')

    # start_time = phdl.exec('date')
    start_time = phdl.exec('date +"%a %b %e %H:%M"')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    # Build list of nodes and their VPC IPs (used by the RCCL test)
    # make sure the VPC IPs are reachable from all nodes for passwordless ssh
    # otherwise use the regular mgmt-ip if that is reachable.
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    # Get cluster snapshot ..
    if re.search('True', config_dict['cluster_snapshot_debug'], re.I):
        cluster_dict_before = create_cluster_metrics_snapshot(phdl)

    # Optionally source environment (e.g., set MPI/ROCm env) before running RCCL tests
    if not re.search('None', config_dict['env_source_script'], re.I):
        phdl.exec(f"bash {config_dict['env_source_script']}")
        shdl.exec(f"bash {config_dict['env_source_script']}")

    # Execute the RCCL cluster test with parameters sourced from config_dict
    result_dict = rccl_lib.rccl_cluster_test(
        phdl,
        shdl,
        test_name=rccl_collective,
        cluster_node_list=node_list,
        vpc_node_list=vpc_node_list,
        user_name=cluster_dict['username'],
        ib_hca_list=config_dict['ib_hca_list'],
        net_dev_list=config_dict['net_dev_list'],
        oob_port=config_dict['oob_port'],
        no_of_global_ranks=config_dict['no_of_global_ranks'],
        rocm_path_var=config_dict['rocm_path_var'],
        mpi_dir=config_dict['mpi_dir'],
        mpi_path_var=config_dict['mpi_path_var'],
        rccl_dir=config_dict['rccl_dir'],
        rccl_path_var=config_dict['rccl_path_var'],
        rccl_tests_dir=config_dict['rccl_tests_dir'],
        nccl_socket_ifname=config_dict.get('nccl_socket_ifname', ''),
        nccl_algo=rccl_algo,
        nccl_proto=rccl_protocol,
        gid_index=config_dict['gid_index'],
        qp_count=qp_scale,
        start_msg_size=config_dict['start_msg_size'],
        end_msg_size=config_dict['end_msg_size'],
        step_function=config_dict['step_function'],
        threads_per_gpu=config_dict['threads_per_gpu'],
        warmup_iterations=config_dict['warmup_iterations'],
        no_of_iterations=config_dict['no_of_iterations'],
        check_iteration_count=config_dict['check_iteration_count'],
        debug_level=config_dict['debug_level'],
        rccl_result_file=config_dict['rccl_result_file'],
        no_of_local_ranks=config_dict['no_of_local_ranks'],
        ib_rx_queue_len=config_dict['ib_rx_queue_len'],
        ucx_tls=config_dict['ucx_tls'],
        hcoll_enable_mcast_all=config_dict['hcoll_enable_mcast_all'],
        nccl_cumem_enable=config_dict['nccl_cumem_enable'],
        nccl_ib_timeout=config_dict['nccl_ib_timeout'],
        nccl_ib_sl=config_dict['nccl_ib_sl'],
        nccl_ib_tc=config_dict['nccl_ib_tc'],
        nccl_ib_split_data_on_qps=config_dict['nccl_ib_split_data_on_qps'],
        nccl_pxn_disable=nccl_pxn_disable,
        nccl_net_plugin=config_dict['nccl_net_plugin'],
        user_key_file=cluster_dict['priv_key_file'],
        verify_bus_bw=config_dict['verify_bus_bw'],
        verify_bw_dip=config_dict['verify_bw_dip'],
        verify_lat_dip=config_dict['verify_lat_dip'],
        exp_results_dict=config_dict['results'],
        env_source_script=config_dict['env_source_script'],
    )

    print(result_dict)
    key_name = f'{rccl_collective}-{rccl_algo}-{rccl_protocol}-{qp_scale}-{nccl_pxn_disable}'
    rccl_res_dict[key_name] = result_dict

    # Scan dmesg between start and end times cluster wide ..
    # end_time = phdl.exec('date')
    phdl.exec(f'sudo echo "End of Test {rccl_collective} {rccl_algo} {rccl_protocol}" | sudo tee /dev/kmsg')

    end_time = phdl.exec('date +"%a %b %e %H:%M"')
    verify_dmesg_for_errors(phdl, start_time, end_time, till_end_flag=True)

    # Get new cluster snapshot and compare ..
    if re.search('True', config_dict['cluster_snapshot_debug'], re.I):
        cluster_dict_after = create_cluster_metrics_snapshot(phdl)
        compare_cluster_metrics_snapshots(cluster_dict_before, cluster_dict_after)

    # Update test results based on any failures ..
    update_test_result()


def test_gen_graph():
    print('Final Global result dict')
    print(rccl_res_dict)
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    print(rccl_graph_dict)

    proc_id = os.getpid()

    html_file = f'/tmp/rccl_perf_report_{proc_id}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)

    print(f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view')
