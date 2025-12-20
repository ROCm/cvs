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
import shutil
import socket

from datetime import datetime


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
        # Consider "not loaded" or "not found" as acceptable (firewall not installed)
        if not re.search('inactive|dead|stopped|disabled|not loaded|could not be found', out_dict[node], re.I):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()


data_type_full_list = [
    "bfloat16",
    "float",
    "uint64",
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "half",
    "double",
    "fp8_e4m3",
    "fp8_e5m2",
]


def pytest_generate_tests(metafunc):
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

    gpu_count_list = rccl.get("gpu_count_list", ["8", "16"])
    data_type_list = rccl.get("data_type_list", ["float", "bfloat16"])
    all_keys = ("rccl_collective", "gpu_count", "data_type")

    active = [k for k in all_keys if k in metafunc.fixturenames]
    if not active:
        return

    domain_by_key = {"rccl_collective": rccl_collective_list, "data_type": data_type_list, "gpu_count": gpu_count_list}
    domains = [domain_by_key[k] for k in active]

    params, ids = [], []
    for values in itertools.product(*domains):
        combo = dict(zip(active, values))

        params.append(values)

        ids.append("|".join(f"{k}={combo[k]}" for k in active))
    metafunc.parametrize(",".join(active), params, ids=ids)


def test_rccl_perf(cluster_dict, config_dict, rccl_collective, gpu_count, data_type):
    """
    Execute RCCL performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
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

    globals.error_list = []
    full_node_list = list(cluster_dict['node_dict'].keys())
    no_of_nodes = int(int(gpu_count) / 8)

    node_list = full_node_list[:no_of_nodes]
    no_of_global_ranks = int(gpu_count)

    # Build list of nodes and their VPC IPs (used by the RCCL test)
    # make sure the VPC IPs are reachable from all nodes for passwordless ssh
    # otherwise use the regular mgmt-ip if that is reachable.
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        if node in node_list:
            vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    # Take the phdl, shdl
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])

    # Log a message to Dmesg to create a timestamp record
    phdl.exec(f'sudo echo "Starting Test {rccl_collective}" | sudo tee /dev/kmsg')
    start_time = phdl.exec('date +"%a %b %e %H:%M"')

    # Get cluster snapshot ..
    if re.search('True', config_dict['cluster_snapshot_debug'], re.I):
        cluster_dict_before = create_cluster_metrics_snapshot(phdl)

    # Optionally source environment (e.g., set MPI/ROCm env) before running RCCL tests
    if not re.search('None', config_dict['env_source_script'], re.I):
        phdl.exec(f'bash {config_dict["env_source_script"]}')

    # Execute the RCCL cluster test with parameters sourced from config_dict
    try:
        result_dict = rccl_lib.rccl_cluster_test_default(
            phdl,
            shdl,
            test_name=rccl_collective,
            cluster_node_list=node_list,
            vpc_node_list=vpc_node_list,
            user_name=cluster_dict['username'],
            ib_hca_list=config_dict['ib_hca_list'],
            net_dev_list=config_dict['net_dev_list'],
            oob_port=config_dict['oob_port'],
            no_of_global_ranks=no_of_global_ranks,
            rocm_path_var=config_dict['rocm_path_var'],
            mpi_dir=config_dict['mpi_dir'],
            mpi_path_var=config_dict['mpi_path_var'],
            rccl_dir=config_dict['rccl_dir'],
            rccl_path_var=config_dict['rccl_path_var'],
            rccl_tests_dir=config_dict['rccl_tests_dir'],
            nccl_socket_ifname=config_dict.get('nccl_socket_ifname', ''),
            gid_index=config_dict['gid_index'],
            start_msg_size=config_dict['start_msg_size'],
            end_msg_size=config_dict['end_msg_size'],
            step_function=config_dict['step_function'],
            threads_per_gpu=config_dict['threads_per_gpu'],
            no_of_cycles=config_dict['no_of_cycles'],
            data_types=[data_type],
            warmup_iterations=config_dict['warmup_iterations'],
            no_of_iterations=config_dict['no_of_iterations'],
            check_iteration_count=config_dict['check_iteration_count'],
            debug_level=config_dict['debug_level'],
            rccl_result_file=f"/tmp/rccl_{rccl_collective}_{data_type}_{gpu_count}.json",
            no_of_local_ranks=config_dict['no_of_local_ranks'],
            ucx_tls=config_dict['ucx_tls'],
            nccl_net_plugin=config_dict['nccl_net_plugin'],
            user_key_file=cluster_dict['priv_key_file'],
            verify_bus_bw=config_dict['verify_bus_bw'],
            verify_bw_dip=config_dict['verify_bw_dip'],
            verify_lat_dip=config_dict['verify_lat_dip'],
            nic_model=config_dict['nic_model'],
            exp_results_dict=config_dict['expected_results'],
        )

        print(result_dict)
        key_name = f'{rccl_collective}-{data_type}-{gpu_count}'
        rccl_res_dict[key_name] = result_dict

        # Scan dmesg between start and end times cluster wide ..
        # end_time = phdl.exec('date')
        phdl.exec(f'sudo echo "End of Test {rccl_collective}" | sudo tee /dev/kmsg')

        end_time = phdl.exec('date +"%a %b %e %H:%M"')
        verify_dmesg_for_errors(phdl, start_time, end_time, till_end_flag=True)
    finally:
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

    current_datetime = datetime.now()
    time_stamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    html_file = f'/tmp/rccl_perf_report_{time_stamp}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_default_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)
    print(f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view')


def test_gen_heatmap(phdl, cluster_dict, config_dict):
    print('Generate Heatmap')
    current_datetime = datetime.now()
    time_stamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    heatmap_file = f'/tmp/rccl_heatmap_{time_stamp}.html'
    # Convert raw results to graph format for HTML reports
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    rccl_res_json_file = f'/tmp/rccl_result_{time_stamp}.json'
    rccl_ref_json_file = config_dict['golden_reference_json_file']
    heatmap_title = config_dict.get('heatmap_title', 'RCCL Performance Heatmap')

    # Collect system metadata from compute nodes
    print('Collecting system metadata...')
    # Environment variables are now automatically extracted from config_dict
    # You can optionally pass env_vars=['PATH', 'LD_LIBRARY_PATH'] to capture shell vars
    metadata = collect_system_metadata(phdl, cluster_dict, config_dict)

    # Create structured output with metadata and results
    structured_output = {'metadata': metadata, 'result': rccl_graph_dict}

    # Save structured output with metadata
    structured_json_file = f'/tmp/rccl_result_with_metadata_{time_stamp}.json'
    with open(structured_json_file, "w") as fp:
        json.dump(structured_output, fp, indent=4)
    print(f'Saved structured results with metadata to {structured_json_file}')

    # Save original format for backward compatibility (used by HTML generation)
    with open(rccl_res_json_file, "w") as fp:
        json.dump(rccl_graph_dict, fp, indent=4)

    # Collect and save aggregated data from individual test runs
    print('Collecting aggregated data from individual test runs...')
    aggregated_data = {}
    for key_name in rccl_res_dict.keys():
        # Parse the key to get test parameters
        parts = key_name.split('-')
        if len(parts) >= 3:
            collective = parts[0]
            data_type = parts[1]
            gpu_count = parts[2]

            # Look for aggregated file
            aggregated_file = f'/tmp/rccl_{collective}_{data_type}_{gpu_count}_aggregated.json'
            try:
                if os.path.exists(aggregated_file):
                    with open(aggregated_file, 'r') as fp:
                        agg_data = json.load(fp)
                        aggregated_data[key_name] = agg_data
                        print(f'Loaded aggregated data from {aggregated_file}')
                else:
                    print(f'Warning: Aggregated file not found: {aggregated_file}')
            except Exception as e:
                print(f'Warning: Failed to load aggregated data from {aggregated_file}: {e}')

    # Save aggregated data with metadata
    if aggregated_data:
        aggregated_output = {'metadata': metadata, 'aggregated_results': aggregated_data}
        aggregated_json_file = f'/tmp/rccl_result_final_aggregated_{time_stamp}.json'
        with open(aggregated_json_file, "w") as fp:
            json.dump(aggregated_output, fp, indent=4)
        print(f'Saved final aggregated results to {aggregated_json_file}')

    # Generate HTML heatmap and reports
    html_lib.add_html_begin(heatmap_file)
    html_lib.build_rccl_heatmap(heatmap_file, 'heatmapdiv', heatmap_title, rccl_res_json_file, rccl_ref_json_file)
    html_lib.build_rccl_heatmap_metadata_table(heatmap_file, structured_json_file, rccl_ref_json_file)
    html_lib.build_rccl_heatmap_table(heatmap_file, 'Heatmap data Table', rccl_res_json_file, rccl_ref_json_file)
    html_lib.add_html_end(heatmap_file)

    # Get management/login node IP from cluster config
    mgmt_node = cluster_dict.get('head_node_dict', {}).get('mgmt_ip', None)

    # Get current hostname/IP to check if we're on the management node
    current_host = socket.gethostname()

    # Get output directory from config
    output_dir = config_dict.get('output_dir', '/tmp')

    # List of files to copy
    files_to_copy = [heatmap_file, structured_json_file, rccl_res_json_file]
    if aggregated_data:
        files_to_copy.append(aggregated_json_file)

    copied_files = {}
    is_remote_copy = False

    # Determine if we're on a compute node (not the management node)
    on_compute_node = False
    if mgmt_node:
        # Check if current host matches management node
        try:
            mgmt_ip = socket.gethostbyname(mgmt_node)
            current_ip = socket.gethostbyname(current_host)
            on_compute_node = mgmt_ip != current_ip and current_host != mgmt_node
        except:
            # If resolution fails, assume different if names don't match
            on_compute_node = current_host != mgmt_node

    # If on compute node, copy files to management node
    if on_compute_node and mgmt_node:
        print(f'Running on compute node: {current_host}')
        print(f'Copying results to management node: {mgmt_node}:{output_dir}/')
        is_remote_copy = True

        # Get SSH key and username from cluster config
        ssh_key = cluster_dict.get('priv_key_file', '')
        username = cluster_dict.get('username', os.getenv('USER', 'root'))
        ssh_options = '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
        if ssh_key:
            ssh_options += f' -i {ssh_key}'

        # Copy each file to the management node
        for src_file in files_to_copy:
            if os.path.exists(src_file):
                dst_path = f'{output_dir}/{os.path.basename(src_file)}'
                scp_cmd = f'scp {ssh_options} {src_file} {username}@{mgmt_node}:{dst_path}'

                try:
                    result = os.system(scp_cmd)
                    if result == 0:
                        copied_files[os.path.basename(src_file)] = dst_path
                        print(f'Copied {os.path.basename(src_file)} to {mgmt_node}:{dst_path}')
                    else:
                        print(f'Failed to copy {os.path.basename(src_file)} (exit code: {result})')
                except Exception as e:
                    print(f'Error copying {src_file}: {e}')
    else:
        # On management node or no mgmt_node configured - copy locally
        print(f'Running on management node: {current_host}')
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f'Warning: Could not create {output_dir}: {e}')
            output_dir = '/tmp'

        for src_file in files_to_copy:
            if os.path.exists(src_file):
                dst_file = os.path.join(output_dir, os.path.basename(src_file))
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_files[os.path.basename(src_file)] = dst_file
                    print(f'Copied {os.path.basename(src_file)} to {dst_file}')
                except Exception as e:
                    print(f'Failed to copy {src_file}: {e}')

    print('\n=== RCCL Test Results ===')
    print('Generated files:')
    print(f'  - Heatmap: {heatmap_file}')
    print(f'  - Structured results: {structured_json_file}')
    print(f'  - Aggregated results: {aggregated_json_file if aggregated_data else "N/A"}')

    if copied_files:
        if is_remote_copy:
            print(f'\n Results copied to management node ({mgmt_node}):')
            print(f'  Location: {output_dir}/')
            print('  Files:')
            for filename in copied_files.keys():
                print(f'    - {filename}')
            print(f'\nAccess results on {mgmt_node} with:')
            print(f'  ls {output_dir}/rccl_*{time_stamp}*')
        else:
            print(f'\n Files available in: {output_dir}/')
            for filename in copied_files.keys():
                print(f'    - {filename}')
    print('=========================\n')
