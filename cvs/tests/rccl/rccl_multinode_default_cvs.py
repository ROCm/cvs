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
        if not re.search('inactive|dead|stopped|disabled', result.output, re.I):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()


# Change this to choose what collectives to run ..
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
def test_rccl_perf(orch, cluster_dict, config_dict, rccl_collective):
    """
    Execute RCCL performance test across the cluster with given parameters.

    Parameters (from fixtures and config):
      - orch: orchestrator for remote execution and MPI (baremetal, container, etc.).
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
    orch.exec(f'sudo echo "Starting Test {rccl_collective}" | sudo tee /dev/kmsg')

    # start_time = orch.exec('date')
    start_time = orch.exec('date +"%a %b %e %H:%M"')
    globals.error_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    # Build list of nodes and their VPC IPs (used by the RCCL test)
    # make sure the VPC IPs are reachable from all nodes for passwordless ssh
    # otherwise use the regular mgmt-ip if that is reachable.
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    # Get cluster snapshot ..
    if re.search('True', config_dict.get('cluster_snapshot_debug', 'False'), re.I):
        cluster_dict_before = create_cluster_metrics_snapshot(orch)

    # Optionally source environment (e.g., set MPI/ROCm env) before running RCCL tests
    if not re.search('None', config_dict['env_source_script'], re.I):
        orch.exec(f"bash {config_dict['env_source_script']}")

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
        no_of_global_ranks=config_dict['no_of_global_ranks'],
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
        warmup_iterations=config_dict['warmup_iterations'],
        no_of_iterations=config_dict['no_of_iterations'],
        no_of_cycles=config_dict['no_of_cycles'],
        check_iteration_count=config_dict['check_iteration_count'],
        debug_level=config_dict['debug_level'],
        rccl_result_file=config_dict['rccl_result_file'],
        no_of_local_ranks=config_dict['no_of_local_ranks'],
        ucx_tls=config_dict['ucx_tls'],
        nccl_net_plugin=config_dict['nccl_net_plugin'],
        mpi_pml=config_dict.get('mpi_pml', 'auto'),
        user_key_file=cluster_dict['priv_key_file'],
        verify_bus_bw=config_dict['verify_bus_bw'],
        verify_bw_dip=config_dict['verify_bw_dip'],
        verify_lat_dip=config_dict['verify_lat_dip'],
        exp_results_dict=config_dict['results'],
        env_source_script=config_dict['env_source_script'],
    )

    log.info("%s", result_dict)
    key_name = f'{rccl_collective}'
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

    proc_id = os.getpid()

    html_file = f'/tmp/rccl_perf_report_{proc_id}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_default_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)

    # Add the HTML file to the report bundle with clickable link
    copied_path = request.config._html_report_manager.add_html_to_report(
        html_file, link_name="RCCL Performance Report", request=request
    )

    if copied_path:
        log.info(f'Perf report saved and added to report bundle: {copied_path}')
    else:
        log.info(
            f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view'
        )
