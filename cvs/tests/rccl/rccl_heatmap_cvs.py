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
from cvs.core.scope import ExecScope, ExecTarget
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


rccl_res_dict = {}


# cluster_file / config_file / cluster_dict / orch_config / orch fixtures
# come from cvs/tests/rccl/conftest.py.


@pytest.fixture(scope="module")
def config_dict(orch_config, cluster_dict):
    """RCCL-specific config block, with path placeholders resolved."""
    config_dict = orch_config.testsuite["rccl"]
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


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

# (test_launch_container removed: orch.setup() runs runtime.setup() AND
# PREPARE_PIPELINE before yielding.)


def test_collect_hostinfo(orch):
    """
    Collect basic ROCm/host info from all nodes using orchestrator.

    Behavior:
      - Executes common ROCm commands to capture version and agent info.
      - If container is enabled, executes inside container to get container ROCm version.
      - Does not parse output; relies on update_test_result to finalize status.
    """

    globals.error_list = []
    orch.exec('cat /opt/rocm/.info/version')
    orch.exec('hipconfig')
    orch.exec('rocm_agent_enumerator')
    update_test_result()


def test_collect_networkinfo(orch):
    """
    Collect basic RDMA/verbs info from all nodes using orchestrator.

    Behavior:
      - Executes 'rdma link' and 'ibv_devinfo' to snapshot network capabilities.
      - Always executes on baremetal host regardless of orchestrator type.
      - Does not parse output; relies on update_test_result to finalize status.
    """

    globals.error_list = []
    # RDMA / verbs queries must run on the host, not inside the runtime.
    orch.exec('rdma link', scope=ExecScope.ALL, target=ExecTarget.HOST)
    orch.exec('ibv_devinfo', scope=ExecScope.ALL, target=ExecTarget.HOST)
    update_test_result()


def test_disable_firewall(orch):
    globals.error_list = []
    # Firewall operations must run on host, not inside the runtime.
    orch.exec('sudo service ufw stop', scope=ExecScope.ALL, target=ExecTarget.HOST)
    time.sleep(2)
    out_dict = orch.exec(
        'sudo service ufw status', scope=ExecScope.ALL, target=ExecTarget.HOST
    )
    for node, result in out_dict.items():
        # Consider "not loaded" or "not found" as acceptable (firewall not installed)
        if not re.search('inactive|dead|stopped|disabled|not loaded|could not be found', result.output, re.I):
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
        log.warning(f'Warning: Missing or invalid config file {config_file}')
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
    channel_config_list = rccl.get("channel_config_list", ["default"])
    all_keys = ("rccl_collective", "gpu_count", "data_type", "channel_config")

    active = [k for k in all_keys if k in metafunc.fixturenames]
    if not active:
        return

    domain_by_key = {
        "rccl_collective": rccl_collective_list,
        "data_type": data_type_list,
        "gpu_count": gpu_count_list,
        "channel_config": channel_config_list,
    }
    domains = [domain_by_key[k] for k in active]

    params, ids = [], []
    for values in itertools.product(*domains):
        combo = dict(zip(active, values))

        params.append(values)

        ids.append("|".join(f"{k}={combo[k]}" for k in active))
    metafunc.parametrize(",".join(active), params, ids=ids)


def test_rccl_perf(cluster_dict, config_dict, orch, rccl_collective, gpu_count, data_type, channel_config):
    """
    Execute RCCL performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
      - cluster_dict: cluster topology and credentials (expects node_dict, username, etc.).
      - config_dict: test configuration with RCCL/MPI paths, env, and thresholds.
      - rccl_collective: which RCCL collective test to run (e.g., "all_reduce_perf").
      - gpu_count: number of GPUs to use for the test.
      - data_type: data type for the test (e.g., "float", "bfloat16").
      - channel_config: NCCL channel configuration in "min-max" format (e.g., "64-64").

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

    # Parse channel configuration (format: "min-max" or "default")
    if str(channel_config).lower() == "default":
        # Use RCCL defaults (don't specify channel parameters)
        min_channels = None
        max_channels = None
    elif "-" in str(channel_config):
        # Parse "min-max" format
        min_channels, max_channels = channel_config.split("-")
        min_channels = int(min_channels)
        max_channels = int(max_channels)
    else:
        # Single value - use for both min and max
        min_channels = int(channel_config)
        max_channels = int(channel_config)

    # Build list of nodes and their VPC IPs (used by the RCCL test)
    # make sure the VPC IPs are reachable from all nodes for passwordless ssh
    # otherwise use the regular mgmt-ip if that is reachable.
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        if node in node_list:
            vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    # Log a message to Dmesg to create a timestamp record
    orch.exec(f'sudo echo "Starting Test {rccl_collective}" | sudo tee /dev/kmsg')
    start_time = orch.exec('date +"%a %b %e %H:%M"')

    # Get cluster snapshot ..
    if re.search('True', config_dict.get('cluster_snapshot_debug', 'False'), re.I):
        cluster_dict_before = create_cluster_metrics_snapshot(orch)

    # Optionally source environment (e.g., set MPI/ROCm env) before running RCCL tests
    if not re.search('None', config_dict['env_source_script'], re.I):
        orch.exec(f'bash {config_dict["env_source_script"]}')

    # Execute the RCCL cluster test with parameters sourced from config_dict
    result_dict = rccl_lib.rccl_cluster_test_default(
        orch,
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
        rccl_result_file=f"/tmp/rccl_{rccl_collective}_{data_type}_{gpu_count}_ch{channel_config}.json",
        no_of_local_ranks=config_dict['no_of_local_ranks'],
        ucx_tls=config_dict['ucx_tls'],
        nccl_net_plugin=config_dict['nccl_net_plugin'],
        mpi_pml=config_dict.get('mpi_pml', 'auto'),
        user_key_file=cluster_dict['priv_key_file'],
        verify_bus_bw=config_dict['verify_bus_bw'],
        verify_bw_dip=config_dict['verify_bw_dip'],
        verify_lat_dip=config_dict['verify_lat_dip'],
        nic_model=config_dict['nic_model'],
        exp_results_dict=config_dict['results'],
        min_channels=min_channels,
        max_channels=max_channels,
        env_source_script=config_dict['env_source_script'],
    )

    log.info("%s", result_dict)
    key_name = f'{rccl_collective}-{data_type}-{gpu_count}-ch{channel_config}'
    rccl_res_dict[key_name] = result_dict

    # Scan dmesg between start and end times cluster wide ..
    # end_time = orch.exec('date')
    orch.exec(f'sudo echo "End of Test {rccl_collective}" | sudo tee /dev/kmsg')

    end_time = orch.exec('date +"%a %b %e %H:%M"')
    verify_dmesg_for_errors(orch, start_time, end_time, till_end_flag=True)

    # Get new cluster snapshot and compare ..
    if re.search('True', config_dict.get('cluster_snapshot_debug', 'False'), re.I):
        cluster_dict_after = create_cluster_metrics_snapshot(orch)
        compare_cluster_metrics_snapshots(cluster_dict_before, cluster_dict_after)

    # Update test results based on any failures ..
    update_test_result()


def test_gen_graph(request):
    log.info('Final Global result dict')
    log.info("%s", rccl_res_dict)
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    log.info("%s", rccl_graph_dict)

    current_datetime = datetime.now()
    time_stamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    html_file = f'/tmp/rccl_perf_report_{time_stamp}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_default_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)

    # Add the HTML file to the report bundle with clickable link
    copied_path = request.config._html_report_manager.add_html_to_report(
        html_file, link_name="RCCL Heatmap Performance Report", request=request
    )

    if copied_path:
        log.info(f'Perf report saved and added to report bundle: {copied_path}')
    else:
        log.info(
            f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view'
        )


def test_gen_heatmap(request, orch, cluster_dict, config_dict):
    log.info('Generate Heatmap')
    current_datetime = datetime.now()
    time_stamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    heatmap_file = f'/tmp/rccl_heatmap_{time_stamp}.html'
    # Convert raw results to graph format for HTML reports
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    rccl_res_json_file = f'/tmp/rccl_result_{time_stamp}.json'
    rccl_ref_json_file = config_dict['golden_reference_json_file']
    heatmap_title = config_dict.get('heatmap_title', 'RCCL Performance Heatmap')

    # Collect system metadata from compute nodes
    log.info('Collecting system metadata...')
    # Environment variables are now automatically extracted from config_dict
    # You can optionally pass env_vars=['PATH', 'LD_LIBRARY_PATH'] to capture shell vars
    metadata = collect_system_metadata(orch, cluster_dict, config_dict)

    # Create structured output with metadata and results
    structured_output = {'metadata': metadata, 'result': rccl_graph_dict}

    # Save structured output with metadata
    structured_json_file = f'/tmp/rccl_result_with_metadata_{time_stamp}.json'
    with open(structured_json_file, "w") as fp:
        json.dump(structured_output, fp, indent=4)
    log.info(f'Saved structured results with metadata to {structured_json_file}')

    # Save original format for backward compatibility (used by HTML generation)
    with open(rccl_res_json_file, "w") as fp:
        json.dump(rccl_graph_dict, fp, indent=4)

    # Collect and save aggregated data from individual test runs
    log.info('Collecting aggregated data from individual test runs...')
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
                        log.info(f'Loaded aggregated data from {aggregated_file}')
                else:
                    log.warning(f'Warning: Aggregated file not found: {aggregated_file}')
            except Exception as e:
                log.warning(f'Warning: Failed to load aggregated data from {aggregated_file}: {e}')

    # Save aggregated data with metadata
    if aggregated_data:
        aggregated_output = {'metadata': metadata, 'aggregated_results': aggregated_data}
        aggregated_json_file = f'/tmp/rccl_result_final_aggregated_{time_stamp}.json'
        with open(aggregated_json_file, "w") as fp:
            json.dump(aggregated_output, fp, indent=4)
        log.info(f'Saved final aggregated results to {aggregated_json_file}')

    # Generate HTML heatmap and reports
    html_lib.add_html_begin(heatmap_file)
    html_lib.build_rccl_heatmap(heatmap_file, 'heatmapdiv', heatmap_title, rccl_res_json_file, rccl_ref_json_file)
    html_lib.build_rccl_heatmap_metadata_table(heatmap_file, structured_json_file, rccl_ref_json_file)
    html_lib.build_rccl_heatmap_table(heatmap_file, 'Heatmap data Table', rccl_res_json_file, rccl_ref_json_file)
    html_lib.add_html_end(heatmap_file)

    # Add the heatmap HTML file to the report bundle with clickable link
    copied_path = request.config._html_report_manager.add_html_to_report(
        heatmap_file, link_name="RCCL Heatmap Visualization", request=request
    )

    if copied_path:
        log.info(f'Heatmap report saved and added to report bundle: {copied_path}')

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
        log.info(f'Running on compute node: {current_host}')
        log.info(f'Copying results to management node: {mgmt_node}:{output_dir}/')
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
                        log.info(f'Copied {os.path.basename(src_file)} to {mgmt_node}:{dst_path}')
                    else:
                        log.warning(f'Failed to copy {os.path.basename(src_file)} (exit code: {result})')
                except Exception as e:
                    log.error(f'Error copying {src_file}: {e}')
    else:
        # On management node or no mgmt_node configured - copy locally
        log.info(f'Running on management node: {current_host}')
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            log.warning(f'Warning: Could not create {output_dir}: {e}')
            output_dir = '/tmp'

        for src_file in files_to_copy:
            if os.path.exists(src_file):
                dst_file = os.path.join(output_dir, os.path.basename(src_file))
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_files[os.path.basename(src_file)] = dst_file
                    log.info(f'Copied {os.path.basename(src_file)} to {dst_file}')
                except Exception as e:
                    log.warning(f'Failed to copy {src_file}: {e}')

    log.info('\n=== RCCL Test Results ===')
    log.info('Generated files:')
    log.info(f'  - Heatmap: {heatmap_file}')
    log.info(f'  - Structured results: {structured_json_file}')
    log.info(f'  - Aggregated results: {aggregated_json_file if aggregated_data else "N/A"}')

    if copied_files:
        if is_remote_copy:
            log.info(f'\n Results copied to management node ({mgmt_node}):')
            log.info(f'  Location: {output_dir}/')
            log.info('  Files:')
            for filename in copied_files.keys():
                log.info(f'    - {filename}')
            log.info(f'\nAccess results on {mgmt_node} with:')
            log.info(f'  ls {output_dir}/rccl_*{time_stamp}*')
        else:
            log.info(f'\n Files available in: {output_dir}/')
            for filename in copied_files.keys():
                log.info(f'    - {filename}')
    log.info('=========================\n')
