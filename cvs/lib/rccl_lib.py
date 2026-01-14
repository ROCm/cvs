'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Standard libraries
import re
import json
from pathlib import Path

# Third party libraries
import pandas as pd
from pydantic import ValidationError

from cvs.lib import globals
from cvs.schema.rccl import RcclTestsAggregated, RcclTestsMultinodeRaw
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


rccl_err_dict = {
    'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
    'nccl': 'NCCL ERROR|Test failure',
    'fs_err': 'No such file or directory',
}


def is_ucx_available_in_mpi(orch, mpi_path, head_node):
    """
    Check if UCX is available in the OpenMPI build.

    Parameters:
      orch: Orchestrator instance for executing commands.
      mpi_path: Path to the MPI installation directory.
      head_node: The head node hostname for retrieving command output.

    Returns:
      bool: True if UCX is available, False otherwise.
    """
    # First, try ompi_info
    check_ucx_cmd = f'{mpi_path}/bin/ompi_info | grep "pml: ucx" | wc -l'
    try:
        ucx_check_out = orch.exec_on_head(check_ucx_cmd)
        ucx_available = int(ucx_check_out[head_node].strip()) > 0
        log.info("UCX available in OpenMPI build" if ucx_available else "UCX not available in OpenMPI build")
        return ucx_available
    except Exception as e:
        log.warning(f"ompi_info check failed: {e}; falling back to ldd check")

    # Fallback: check if libmpi.so is linked to UCX
    libmpi_path = f'{mpi_path}/lib/libmpi.so'
    check_ldd_cmd = f'ldd {libmpi_path} | grep ucx | wc -l'
    try:
        ldd_out = orch.exec_on_head(check_ldd_cmd)
        ucx_available = int(ldd_out[head_node].strip()) > 0
        log.info("UCX linked in libmpi.so" if ucx_available else "UCX not linked in libmpi.so")
        return ucx_available
    except Exception as e:
        log.warning(f"ldd check failed: {e}; assuming UCX is not available")
        return False


def determine_mpi_pml_config(orch, mpi_pml, mpi_path, head_node, net_dev_list, ucx_tls):
    """
    Determine MPI PML (Point-to-Point Messaging Layer) configuration based on user config or auto-detection.

    Parameters:
      orch: Orchestrator instance for executing commands.
      mpi_pml: User-specified PML mode ('auto', 'ucx', or 'ob1').
      mpi_path: Path to the MPI installation directory.
      head_node: The head node hostname for retrieving command output.
      net_dev_list: UCX network device(s) to use (UCX_NET_DEVICES).
      ucx_tls: UCX transport layer to use (UCX_TLS).

    Returns:
      tuple: (pml_param, ucx_params) where:
        - pml_param: MCA parameter string for mpirun (e.g., '--mca pml ob1' or '')
        - ucx_params: UCX environment parameter string for mpirun (e.g., '-x UCX_...' or '')
    """
    if mpi_pml.lower() == "auto":
        # Auto-detect UCX availability
        ucx_available = is_ucx_available_in_mpi(orch, mpi_path, head_node)
        pml_param = "--mca pml ob1" if not ucx_available else ""
    elif mpi_pml.lower() == "ucx":
        # User explicitly requested UCX
        ucx_available = True
        pml_param = ""
        log.info("Using UCX (user-specified)")
    elif mpi_pml.lower() == "ob1":
        # User explicitly requested ob1 fallback
        ucx_available = False
        pml_param = "--mca pml ob1"
        log.info("Using pml ob1 (user-specified)")
    else:
        log.warning(f"Unknown mpi_pml value '{mpi_pml}', defaulting to auto-detection")
        ucx_available = is_ucx_available_in_mpi(orch, mpi_path, head_node)
        pml_param = "--mca pml ob1" if not ucx_available else ""

    ucx_params = (
        f"-x UCX_UNIFIED_MODE=y -x UCX_NET_DEVICES={net_dev_list} -x UCX_TLS={ucx_tls} " if ucx_available else ""
    )

    return pml_param, ucx_params


def scan_rccl_logs(output):
    """
    Scan RCCL test stdout for known error/warning patterns and enforce failure criteria.

    Parameters:
      output (str): Combined stdout/stderr text from an RCCL test run.

    Behavior:
      - Iterates over each line to detect:
        * Errors matching patterns in rccl_err_dict (e.g., ORTE/NCCL/FS errors).
        * NCCL WARN lines, which are collected and printed (but not fatal).
      - Fails the test immediately on the first matched error via fail_test(...).
      - After scanning, if no '# Avg bus bandwidth' marker exists in the entire output,
        fails the test because results are considered incomplete.

    Notes:
      - Expects rccl_err_dict (dict of error_name -> regex pattern) to be defined in scope.
      - Expects fail_test(...) to be available, which should raise/exit the test on failure.
      - Uses simple regex searches; patterns in rccl_err_dict can include alternations.
    """
    error_list = []  # Accumulates lines that match known error patterns (for context/auditing)
    warn_list = []  # Accumulates NCCL warning lines (non-fatal but useful for visibility)

    # Process output line-by-line to catch and act on errors/warnings
    for line in output.split("\n"):
        for err_key in rccl_err_dict.keys():
            # Check each line against all known error signatures
            if re.search(f'{rccl_err_dict[err_key]}', line):
                error_list.append(line)
                fail_test(f'ERROR - {line}')
        # Collect NCCL warnings (do not fail the test)
        if re.search('NCCL WARN', line):
            warn_list.append(line)
    if len(warn_list) > 0:
        print('Following warnings were observed in the RCCL test')
        print('#============#')
        print(warn_list)
        print('#============#')
    if not re.search('#\sAvg bus bandwidth', output):
        fail_test('RCCL test did not complete successfully, no bandwidth numbers printed - pls check')


# Not using the avg bus bandwidth verification currently ..
def check_avg_bus_bw(output, exp_res_dict):
    if re.search('#\sAvg bus bandwidth\s+:\s+[0-9\.]+', output, re.I):
        match = re.search('#\sAvg bus bandwidth\s+:\s+([0-9\.]+)', output, re.I)
        actual_bw = float(match.group(1))
        if actual_bw < float(exp_res_dict['avg_bus_bw']):
            fail_test(f"Actual Avg Bus BW {actual_bw} is less than the expected Avg BW {exp_res_dict['avg_bus_bw']}")


def check_bus_bw(test_name, output, exp_res_dict):
    """
    Validate bus bandwidth results from an RCCL test against expected thresholds.

    Parameters:
      test_name (str): Name of the RCCL test (e.g., alltoall, all_reduce_perf).
                       Determines whether to check in-place or out-of-place results.
      output (str): JSON string (possibly with newlines) produced by the RCCL test,
                    containing a list of result dictionaries. Each entry typically includes:
                      - 'size'   : message size for the measurement
                      - 'busBw'  : measured bus bandwidth
                      - 'inPlace': 0 (out-of-place) or 1 (in-place)
      exp_res_dict (dict): Expected results dictionary with the structure:
                    {
                      <msg_size>: {
                          'bus_bw': <min_expected_bus_bw>, ...
                      }
                    }

    Behavior:
      - Parses the JSON output and iterates over measured entries.
      - For alltoall/all_to_all tests, validates out-of-place measurements (inPlace == 0).
      - For other tests, validates in-place measurements (inPlace == 1).
      - Compares measured busBw to minimum expected thresholds per message size.
      - Calls fail_test(...) if any measurement is at least 5% below expectation.

    Notes:
      - Message sizes are compared as strings to avoid type mismatches between JSON and expectations.
      - Assumes fail_test(...) is available in scope to signal test failure.
      - 5% tolerance is applied: test only fails if actual < expected * 0.95
    """

    print(f'exp_res_dict = {exp_res_dict}')

    tolerance = 0.95  # 5% tolerance

    # New hierarchical structure: {msg_size: {'bus_bw': bw_value}}
    msg_size_list = list(exp_res_dict.keys())

    print(test_name)
    # act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    if re.search('alltoall|all_to_all', test_name, re.I):
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        expected_bw = float(exp_res_dict[msg_size]['bus_bw'])
                        actual_bw = float(act_dict['busBw'])
                        threshold = expected_bw * tolerance
                        print(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
                        if actual_bw < threshold:
                            fail_test(
                                f"The actual out-of-place bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})"
                            )
    else:
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        expected_bw = float(exp_res_dict[msg_size]['bus_bw'])
                        actual_bw = float(act_dict['busBw'])
                        threshold = expected_bw * tolerance
                        print(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
                        if actual_bw < threshold:
                            fail_test(
                                f"The actual in-place bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})"
                            )


def check_bw_dip(test_name, output, exp_res_dict=None):
    """
    Check for bandwidth dips as message size increases.
    Only fails if bandwidth drops by more than 5%.
    Only validates message sizes specified in the reference. If no reference provided, skips validation.
    """
    # act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance

    # Get reference message sizes if provided
    # If no reference data, skip validation entirely
    if not exp_res_dict:
        log.info(f"No reference data provided for BW dip check, skipping validation for {test_name}")
        return

    ref_msg_sizes = set(str(size) for size in exp_res_dict.keys())
    log.info(f"Validating BW dip only for reference message sizes: {ref_msg_sizes}")

    if re.search('alltoall|all_to_all', test_name, re.I):
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                # Skip validation if this message size is not in reference
                if str(act_dict['size']) not in ref_msg_sizes:
                    continue

                current_bw = float(act_dict['busBw'])
                threshold = float(last_bw) * tolerance
                if last_bw > 0 and current_bw < threshold:
                    fail_test(
                        f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})"
                    )
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']
    else:
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                # Skip validation if this message size is not in reference
                if str(act_dict['size']) not in ref_msg_sizes:
                    continue

                current_bw = float(act_dict['busBw'])
                threshold = float(last_bw) * tolerance
                if last_bw > 0 and current_bw < threshold:
                    fail_test(
                        f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})"
                    )
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']


def check_lat_dip(test_name, output, exp_res_dict=None):
    """
    Check for latency decreases as message size increases (which would be unexpected).
    Only fails if latency drops by more than 5%.
    Only validates message sizes specified in the reference. If no reference provided, skips validation.
    """
    # act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance

    # Get reference message sizes if provided
    # If no reference data, skip validation entirely
    if not exp_res_dict:
        log.info(f"No reference data provided for latency dip check, skipping validation for {test_name}")
        return

    ref_msg_sizes = set(str(size) for size in exp_res_dict.keys())
    log.info(f"Validating latency dip only for reference message sizes: {ref_msg_sizes}")

    if re.search('alltoall|all_to_all', test_name, re.I):
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                # Skip validation if this message size is not in reference
                if str(act_dict['size']) not in ref_msg_sizes:
                    continue

                current_time = float(act_dict['time'])
                threshold = float(last_time) * tolerance
                if last_time > 0 and current_time < threshold:
                    fail_test(
                        f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})"
                    )
                last_time = act_dict['time']
                last_msg_size = act_dict['size']
    else:
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                # Skip validation if this message size is not in reference
                if str(act_dict['size']) not in ref_msg_sizes:
                    continue

                current_time = float(act_dict['time'])
                threshold = float(last_time) * tolerance
                if last_time > 0 and current_time < threshold:
                    fail_test(
                        f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})"
                    )
                last_time = act_dict['time']
                last_msg_size = act_dict['size']


def convert_to_graph_dict(result_dict):
    graph_dict = {}
    for graph_series_name in result_dict.keys():
        print(graph_series_name)
        graph_dict[graph_series_name] = {}
        dict_list = result_dict[graph_series_name]
        print(dict_list)
        for dict_item in dict_list:
            msg_size = dict_item['size']
            graph_dict[graph_series_name][msg_size] = {}
            if re.search('alltoall', dict_item['name'], re.I) and dict_item['inPlace'] == 1:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
            else:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
    print(graph_dict)
    return graph_dict


def aggregate_rccl_test_results(validated_results):
    """
    Aggregate multiple rccl-test results into mean/std per (name, size, type, inPlace)
    Args: validated_results: List[RcclTests] - list of validated rccl-test results
    Returns: List[RcclTestsAggregated] - list of aggregated rccl-test results with mean/std per (name, size, type, inPlace)
    """
    if not validated_results:
        raise ValueError("validated_results list cannot be empty")

    # Check if these are multinode results and validate consistency
    multinode_config = None
    if isinstance(validated_results[0], RcclTestsMultinodeRaw):
        # Extract config from first result
        first = validated_results[0]
        multinode_config = {
            'nodes': first.nodes,
            'ranks': first.ranks,
            'ranksPerNode': first.ranksPerNode,
            'gpusPerRank': first.gpusPerRank,
        }

        # Validate all results have same config
        for i, result in enumerate(validated_results):
            if not isinstance(result, RcclTestsMultinodeRaw):
                raise ValueError(f"Mixed single-node and multi-node results at index {i}")
            if (
                result.nodes != multinode_config['nodes']
                or result.ranks != multinode_config['ranks']
                or result.ranksPerNode != multinode_config['ranksPerNode']
                or result.gpusPerRank != multinode_config['gpusPerRank']
            ):
                raise ValueError(
                    f"Inconsistent cluster config at index {i}: "
                    f"expected {multinode_config}, got "
                    f"nodes={result.nodes}, ranks={result.ranks}, "
                    f"ranksPerNode={result.ranksPerNode}, gpusPerRank={result.gpusPerRank}"
                )
        log.info(f"Validated consistent multinode config: {multinode_config}")

    log.info(f"Aggregating {len(validated_results)} RCCL test results")
    data = [result.model_dump() for result in validated_results]
    df = pd.DataFrame(data)

    # Group and aggregate
    agg_df = df.groupby(['name', 'size', 'type', 'inPlace'], as_index=False).agg(
        busBw_mean=('busBw', 'mean'),
        busBw_std=('busBw', 'std'),
        algBw_mean=('algBw', 'mean'),
        algBw_std=('algBw', 'std'),
        time_mean=('time', 'mean'),
        time_std=('time', 'std'),
        num_runs=('numCycle', 'count'),
    )

    # Add multinode config if present
    if multinode_config:
        for key, value in multinode_config.items():
            agg_df[key] = value

    agg_results = []
    errors = []

    for row_dict in agg_df.to_dict('records'):
        try:
            agg_results.append(RcclTestsAggregated.model_validate(row_dict))
        except ValidationError as e:
            error_msg = f"Validation failed for row {row_dict}: {e}"
            log.error(error_msg)
            errors.append(error_msg)

    # Report any validation failures
    if errors:
        error_summary = "\n".join(errors)
        fail_test(f"Aggregation validation failed:\n{error_summary}")

    log.info(f"Successfully validated {len(agg_results)} aggregated results")
    return agg_results


# Main RCCL Test library which gets invoked from cvs/test/rccl tests and accepts most of the
# standard NCCL environment variables ..
#
def rccl_cluster_test(
    orch,
    test_name,
    cluster_node_list,
    vpc_node_list,
    user_name,
    ib_hca_list,
    net_dev_list,
    oob_port,
    no_of_global_ranks,
    rocm_path_var,
    mpi_dir,
    mpi_path_var,
    rccl_dir,
    rccl_path_var,
    rccl_tests_dir,
    nccl_socket_ifname="",
    nccl_algo='ring',
    nccl_proto='simple',
    gid_index=1,
    qp_count=1,
    start_msg_size=1024,
    end_msg_size='16g',
    step_function=2,
    threads_per_gpu=1,
    warmup_iterations=10,
    no_of_iterations=1,
    check_iteration_count=1,
    debug_level='INFO',
    rccl_result_file='/tmp/rccl_result_output.json',
    no_of_local_ranks=8,
    ib_rx_queue_len=8192,
    ucx_tls='tcp',
    hcoll_enable_mcast_all=0,
    nccl_cumem_enable=0,
    nccl_ib_timeout=30,
    nccl_ib_sl=0,
    nccl_ib_tc=41,
    nccl_ib_split_data_on_qps=0,
    nccl_pxn_disable=1,
    nccl_net_plugin=None,
    user_password=None,
    min_channels=None,
    max_channels=None,
    data_type="float",
    mpi_pml="auto",
    user_key_file=None,
    verify_bus_bw=False,
    verify_bw_dip=True,
    verify_lat_dip=True,
    exp_results_dict=None,
    env_source_script=None,
    container_config=None,
):
    """
    Run an RCCL collective test across a cluster via MPI and verify results.

    Arguments:
      orch: Orchestrator instance for test execution.
      test_name: RCCL test binary name (e.g., all_reduce_perf).
      cluster_node_list: List of cluster node hostnames/IPs (first is treated as head node).
      vpc_node_list: List of hostnames/IPs to pass to mpirun -H as hosts - \
         Make sure passwordless ssh works between them
      user_name: Username for remote ops (unused here).
      ib_hca_list: Comma-separated IB HCA devices for NCCL (NCCL_IB_HCA).
      net_dev_list: UCX network device(s) to use (UCX_NET_DEVICES).
      oob_port: Interface for MPI TCP OOB (btl_tcp_if_include).
      no_of_global_ranks: Total MPI ranks to launch across the cluster.
      rocm_path_var, mpi_dir, mpi_path_var, rccl_dir, rccl_path_var, rccl_tests_dir: Installation paths.
      nccl_algo, nccl_proto, gid_index, qp_count, ...: NCCL/UCX/MPI tuning parameters.
      start_msg_size, end_msg_size, step_function: Message size sweep setup.
      threads_per_gpu, warmup_iterations, check_iteration_count: Test execution tuning.
      debug_level: NCCL_DEBUG level.
      rccl_result_file: Path where the RCCL test writes JSON results (-Z json -x file).
      verify_bus_bw: If 'True' (string), compare bus BW vs expected thresholds.
      exp_results_dict: Dict of expected results per test for verification.
      container_config: Container configuration for containerized execution (optional).

    Returns:
      result_out: The raw JSON dict read from rccl_result_file on the head node.
    """

    print(f'Starting RCCL Test ..........................................{test_name}')
    # Base ROCm path as provided by caller
    ROCM_PATH = rocm_path_var

    # Resolve tool/library install locations
    # MPI_PATH=f'{mpi_path}/install/bin'
    MPI_PATH = f'{mpi_path_var}'
    MPI_INSTALL_DIR = f'{mpi_dir}'
    RCCL_PATH = f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR = f'{rccl_tests_dir}'

    # Environment variables exported into the mpirun context
    PATH = f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH = f'{RCCL_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')

    # Use the first cluster node as the head node (source for collected outputs)
    # The -H {host_params} is obsolete in ompi5.0 and greater, so changing to
    # --hostfile option
    head_node = cluster_node_list[0]
    # host_params=''
    # proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    # for node in vpc_node_list:
    #    host_params = f'{host_params}{node}:{proc_per_node},'
    # Compute processes per node and build the -H host mapping string: host:N,host:N,...
    # host_params = host_params.rstrip(',')
    # print(f'RCCL Hosts -H value {host_params}')

    # proc_per_node = int(int(no_of_global_ranks) / len(cluster_node_list))
    proc_per_node = int(no_of_local_ranks)

    # Determine PML (Point-to-Point Messaging Layer) based on user config or auto-detection
    pml_param, ucx_params = determine_mpi_pml_config(orch, mpi_pml, MPI_PATH, head_node, net_dev_list, ucx_tls)

    # Wrap test binary in shell to source env script if provided
    test_cmd = f'env && {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {threads_per_gpu} -c {check_iteration_count} -w {warmup_iterations} \
        -d {data_type} \
        -Z json -x {rccl_result_file}'

    if env_source_script and env_source_script.lower() != 'none':
        test_cmd = f'bash -c "source {env_source_script} && {test_cmd}"'
    else:
        # Always wrap in bash to interpret && shell operator
        test_cmd = f'bash -c "{test_cmd}"'

    # Prepare parameters for distribute_using_mpi
    mpi_extra_args = [
        '--bind-to',
        'numa',
        '--mca',
        'btl',
        '^vader,openib',
        '--mca',
        'btl_tcp_if_include',
        oob_port,
        '--mca',
        'oob_tcp_if_include',
        oob_port,
    ]

    env_vars = {
        'NCCL_DEBUG': debug_level,
        'NCCL_IB_GID_INDEX': str(gid_index),
        'UCX_UNIFIED_MODE': 'y',
        'NCCL_IB_PCI_RELAXED_ORDERING': '1',
        'PATH': PATH,
        'LD_LIBRARY_PATH': LD_LIBRARY_PATH,
        'NCCL_IB_HCA': ib_hca_list,
        'UCX_NET_DEVICES': net_dev_list,
        'NCCL_ALGO': nccl_algo,
        'NCCL_IB_QPS_PER_CONNECTION': str(qp_count),
        'IB_RX_QUEUE_LEN': str(ib_rx_queue_len),
        'UCX_TLS': ucx_tls,
        'HCOLL_ENABLE_MCAST_ALL': str(hcoll_enable_mcast_all),
        'NCCL_CUMEM_ENABLE': str(nccl_cumem_enable),
        'NCCL_IB_TIMEOUT': str(nccl_ib_timeout),
        'NCCL_IB_SL': str(nccl_ib_sl),
        'NCCL_IB_TC': str(nccl_ib_tc),
        'NCCL_IB_SPLIT_DATA_ON_QPS': str(nccl_ib_split_data_on_qps),
        'NCCL_PXN_DISABLE': str(nccl_pxn_disable),
        'NCCL_NET_PLUGIN': nccl_net_plugin,
    }

    if nccl_socket_ifname.strip():
        env_vars['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname

    if min_channels is not None:
        env_vars['NCCL_MIN_NCHANNELS'] = str(min_channels)

    if max_channels is not None:
        env_vars['NCCL_MAX_NCHANNELS'] = str(max_channels)

    # Check if UCX is available in OpenMPI build; if not, use pml ob1 and remove UCX env vars
    ucx_available = is_ucx_available_in_mpi(orch, MPI_PATH, head_node)
    if not ucx_available:
        mpi_extra_args.extend(['--mca', 'pml', 'ob1'])
        # Remove UCX-related environment variables
        ucx_env_vars = ['UCX_UNIFIED_MODE', 'UCX_NET_DEVICES', 'UCX_TLS']
        for var in ucx_env_vars:
            env_vars.pop(var, None)

    rank_cmd = test_cmd
    mpi_install_dir = MPI_INSTALL_DIR

    # Get the full command for logging (simulate what distribute_using_mpi will do)
    # Note: This is just for logging, the actual command construction happens in the orchestrator
    print('%%%%%%%%%%%%%%%%')
    print(f'MPI Hosts: {vpc_node_list}')
    print(f'Ranks per host: {proc_per_node}')
    print(f'Env Vars: {env_vars}')
    print(f'Rank Cmd: {rank_cmd}')
    print(f'MPI Install Dir: {mpi_install_dir}')
    print('%%%%%%%%%%%%%%%%')

    try:
        out_dict = orch.distribute_using_mpi(
            rank_cmd, vpc_node_list, proc_per_node, env_vars, mpi_install_dir, mpi_extra_args
        )
        output = out_dict[head_node]
        # print(output)
        scan_rccl_logs(output)
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd - exception {repr(e)}')
        fail_test(f'Hit Exceptions with rccl cmd - exception {repr(e)}')

    # Read the JSON results emitted by the RCCL test binary
    cat_cmd = f'cat {rccl_result_file}'

    result_dict_out = orch.exec(cat_cmd, hosts=[head_node])
    result_out = json.loads(result_dict_out[head_node].replace('\n', '').replace('\r', ''))

    # Collect basic GPU information via rocm-smi
    smi_cmd = 'rocm-smi -a | head -30'

    smi_out_dict = orch.exec(smi_cmd, hosts=[head_node])
    smi_out = smi_out_dict[head_node]

    get_model_from_rocm_smi_output(smi_out)

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None

    if re.search('True', verify_bus_bw, re.I):
        if test_exp_dict:
            check_bus_bw(test_name, result_out, test_exp_dict)

    if re.search('True', verify_bw_dip, re.I):
        check_bw_dip(test_name, result_out, test_exp_dict)

    if re.search('True', verify_lat_dip, re.I):
        check_lat_dip(test_name, result_out, test_exp_dict)

    return result_out


# Main RCCL Test library which gets invoked from cvs/test/rccl tests and accepts most of the
# standard NCCL environment variables ..
#
def rccl_cluster_test_default(
    orch,
    test_name,
    cluster_node_list,
    vpc_node_list,
    user_name,
    ib_hca_list,
    net_dev_list,
    oob_port,
    no_of_global_ranks,
    rocm_path_var,
    mpi_dir,
    mpi_path_var,
    rccl_dir,
    rccl_path_var,
    rccl_tests_dir,
    nccl_socket_ifname="",
    nccl_algo='ring',
    nccl_proto='simple',
    gid_index=1,
    qp_count=1,
    start_msg_size=1024,
    end_msg_size='16g',
    step_function=2,
    threads_per_gpu=1,
    warmup_iterations=10,
    no_of_iterations=1,
    data_types=['float'],
    no_of_cycles=10,
    check_iteration_count=1,
    debug_level='INFO',
    rccl_result_file='/tmp/rccl_result_output.json',
    no_of_local_ranks=8,
    ib_rx_queue_len=8192,
    ucx_tls='tcp',
    hcoll_enable_mcast_all=0,
    nccl_cumem_enable=0,
    nccl_ib_timeout=30,
    nccl_ib_sl=0,
    nccl_ib_tc=41,
    nccl_ib_split_data_on_qps=0,
    nccl_pxn_disable=1,
    nccl_net_plugin=None,
    user_password=None,
    min_channels=None,
    max_channels=None,
    mpi_pml="auto",
    user_key_file=None,
    verify_bus_bw=False,
    verify_bw_dip=True,
    verify_lat_dip=True,
    nic_model='ainic',
    exp_results_dict=None,
    env_source_script=None,
):
    """
    Run an RCCL collective test across a cluster via MPI and verify results.

    Arguments:
      orch: Orchestrator instance for test execution.
      test_name: RCCL test binary name (e.g., all_reduce_perf).
      cluster_node_list: List of cluster node hostnames/IPs (first is treated as head node).
      vpc_node_list: List of hostnames/IPs to pass to mpirun -H as hosts - \
         Make sure passwordless ssh works between them
      user_name: Username for remote ops (unused here).
      ib_hca_list: Comma-separated IB HCA devices for NCCL (NCCL_IB_HCA).
      net_dev_list: UCX network device(s) to use (UCX_NET_DEVICES).
      oob_port: Interface for MPI TCP OOB (btl_tcp_if_include).
      no_of_global_ranks: Total MPI ranks to launch across the cluster.
      rocm_path_var, mpi_dir, mpi_path_var, rccl_dir, rccl_path_var, rccl_tests_dir: Installation paths.
      nccl_algo, nccl_proto, gid_index, qp_count, ...: NCCL/UCX/MPI tuning parameters.
      start_msg_size, end_msg_size, step_function: Message size sweep setup.
      threads_per_gpu, warmup_iterations, check_iteration_count: Test execution tuning.
      data_types: List of data types to test (e.g., ['float', 'half']).
      no_of_cycles: Number of cycles to run for each data type.
      min_channels: Minimum NCCL channels (NCCL_MIN_NCHANNELS).
      max_channels: Maximum NCCL channels (NCCL_MAX_NCHANNELS).
      debug_level: NCCL_DEBUG level.
      rccl_result_file: Path where the RCCL test writes JSON results (-Z json -x file).
      verify_bus_bw: If 'True' (string), compare bus BW vs expected thresholds.
      exp_results_dict: Dict of expected results per test for verification.

    Returns:
      all_raw_results: List of dictionaries containing all test results from all data types.
    """

    print(f'Starting RCCL Test ..........................................{test_name}')
    if min_channels is not None and max_channels is not None:
        log.info(f'Using NCCL channels: min={min_channels}, max={max_channels}')
    else:
        log.info('Using RCCL default NCCL channel configuration')
    # Base ROCm path as provided by caller
    ROCM_PATH = rocm_path_var

    # Resolve tool/library install locations
    # MPI_PATH=f'{mpi_path}/install/bin'
    MPI_PATH = f'{mpi_path_var}'
    MPI_INSTALL_DIR = f'{mpi_dir}'
    RCCL_PATH = f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR = f'{rccl_tests_dir}'

    # Environment variables exported into the mpirun context
    PATH = f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH = f'{RCCL_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')

    # Use the first cluster node as the head node (source for collected outputs)
    # The -H {host_params} is obsolete in ompi5.0 and greater, so changing to
    # --hostfile option
    head_node = cluster_node_list[0]
    # host_params=''
    # proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    # for node in vpc_node_list:
    #    host_params = f'{host_params}{node}:{proc_per_node},'
    # Compute processes per node and build the -H host mapping string: host:N,host:N,...
    # host_params = host_params.rstrip(',')
    # print(f'RCCL Hosts -H value {host_params}')

    # proc_per_node = int(int(no_of_global_ranks) / len(cluster_node_list))
    proc_per_node = int(no_of_local_ranks)

    # Determine PML (Point-to-Point Messaging Layer) based on user config or auto-detection
    pml_param, ucx_params = determine_mpi_pml_config(orch, mpi_pml, MPI_PATH, head_node, net_dev_list, ucx_tls)

    all_raw_results = []
    all_validated_results = []
    base_path = Path(rccl_result_file)

    # Prepare common parameters for distribute_using_mpi
    mpi_extra_args = [
        '--bind-to',
        'numa',
        '--mca',
        'btl',
        '^vader,openib',
        '--mca',
        'btl_tcp_if_include',
        oob_port,
        '--mca',
        'oob_tcp_if_include',
        oob_port,
    ]

    env_vars = {
        'NCCL_DEBUG': debug_level,
        'NCCL_IB_GID_INDEX': str(gid_index),
        'UCX_UNIFIED_MODE': 'y',
        'NCCL_IB_PCI_RELAXED_ORDERING': '1',
        'PATH': PATH,
        'LD_LIBRARY_PATH': LD_LIBRARY_PATH,
        'NCCL_IB_HCA': ib_hca_list,
        'UCX_NET_DEVICES': net_dev_list,
        'NCCL_ALGO': nccl_algo,
        'NCCL_IB_QPS_PER_CONNECTION': str(qp_count),
        'IB_RX_QUEUE_LEN': str(ib_rx_queue_len),
        'UCX_TLS': ucx_tls,
        'HCOLL_ENABLE_MCAST_ALL': str(hcoll_enable_mcast_all),
        'NCCL_CUMEM_ENABLE': str(nccl_cumem_enable),
        'NCCL_IB_TIMEOUT': str(nccl_ib_timeout),
        'NCCL_IB_SL': str(nccl_ib_sl),
        'NCCL_IB_TC': str(nccl_ib_tc),
        'NCCL_IB_SPLIT_DATA_ON_QPS': str(nccl_ib_split_data_on_qps),
        'NCCL_PXN_DISABLE': str(nccl_pxn_disable),
        'NCCL_NET_PLUGIN': nccl_net_plugin,
    }

    if nccl_socket_ifname.strip():
        env_vars['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname

    if min_channels is not None:
        env_vars['NCCL_MIN_NCHANNELS'] = str(min_channels)

    if max_channels is not None:
        env_vars['NCCL_MAX_NCHANNELS'] = str(max_channels)

    # Check if UCX is available in OpenMPI build; if not, use pml ob1 and remove UCX env vars
    ucx_available = is_ucx_available_in_mpi(orch, MPI_PATH, head_node)
    if not ucx_available:
        mpi_extra_args.extend(['--mca', 'pml', 'ob1'])
        # Remove UCX-related environment variables
        ucx_env_vars = ['UCX_UNIFIED_MODE', 'UCX_NET_DEVICES', 'UCX_TLS']
        for var in ucx_env_vars:
            env_vars.pop(var, None)

    mpi_install_dir = MPI_INSTALL_DIR

    for dtype in data_types:
        # Create a unique result file for each data type
        dtype_result_file = f'{base_path.parent}/{base_path.stem}_{dtype}.json'
        log.info(f'Running {test_name} with dtype={dtype}')

        # Wrap test binary in shell to source env script if provided
        test_cmd = (
            f'env && {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
            -g {threads_per_gpu} -c {check_iteration_count} -w {warmup_iterations} \
            -d {dtype} -N {no_of_cycles} -Z json -x {dtype_result_file}'
        )

        if env_source_script and env_source_script.lower() != 'none':
            test_cmd = f'bash -c "source {env_source_script} && {test_cmd}"'
        else:
            # Always wrap in bash to interpret && shell operator
            test_cmd = f'bash -c "{test_cmd}"'

        rank_cmd = test_cmd

        # Get the full command for logging (simulate what distribute_using_mpi will do)
        # Note: This is just for logging, the actual command construction happens in the orchestrator
        print('%%%%%%%%%%%%%%%%')
        print(f'MPI Hosts: {vpc_node_list}')
        print(f'Ranks per host: {proc_per_node}')
        print(f'Env Vars: {env_vars}')
        print(f'Rank Cmd: {rank_cmd}')
        print(f'MPI Install Dir: {mpi_install_dir}')
        print('%%%%%%%%%%%%%%%%')

        try:
            out_dict = orch.distribute_using_mpi(
                rank_cmd, vpc_node_list, proc_per_node, env_vars, mpi_install_dir, mpi_extra_args
            )
            output = out_dict[head_node]
            # print(output)
            scan_rccl_logs(output)
        except Exception as e:
            log.error(f'Hit Exceptions with rccl cmd - exception {repr(e)}')
            fail_test(f'Hit Exceptions with rccl cmd - exception {repr(e)}')

        # Read the JSON results emitted by the RCCL test binary
        result_dict_out = orch.exec(f'cat {dtype_result_file}')
        dtype_result_out = json.loads(result_dict_out[head_node].replace('\n', '').replace('\r', ''))
        # Validate the results against the schema fail if results are not valid
        try:
            validated = [RcclTestsMultinodeRaw.model_validate(test_result) for test_result in dtype_result_out]
            log.info(f'Validation passed: {len(validated)} RcclTests schema validation passed')
            all_validated_results.extend(validated)
            all_raw_results.extend(dtype_result_out)
        except ValidationError as e:
            log.error(f'Validation Failed: {e}')
            fail_test(f'RCCL Test {dtype} schema validation failed: {e}')

    # Save the results to a main result file
    with open(rccl_result_file, 'w') as f:
        json.dump(all_raw_results, f, indent=2)
    log.info(f'Saved combined results from all data types to {rccl_result_file}')

    # Validate the results against the schema and aggregate if multiple results are found, fail if results are not valid
    aggregated_rccl_tests = None
    try:
        if len(all_validated_results) >= 1:
            aggregated_rccl_tests = aggregate_rccl_test_results(all_validated_results)
            log.info(f'Aggregation passed: {len(aggregated_rccl_tests)} RcclTestsAggregated schema validation passed')
            # Note: currently we are saving the aggregated results, but we could instead use this for final report generation
            aggregated_path = f'{base_path.parent}/{base_path.stem}_aggregated.json'
            with open(aggregated_path, 'w') as f:
                json.dump([result.model_dump() for result in aggregated_rccl_tests], f, indent=2)
            log.info(f'Saved aggregated results to {aggregated_path}')
        else:
            log.info('Aggregation skipped: only one run found')
    except ValidationError as e:
        log.error(f'Validation Failed: {e}')
        fail_test(f'RCCL Test schema validation failed: {e}')
    except ValueError as e:
        log.error(f'Aggregation failed: {e}')
        fail_test(f'RCCL Test aggregation failed: {e}')

    # Collect basic GPU information via rocm-smi
    smi_out_dict = orch.exec_on_head('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    get_model_from_rocm_smi_output(smi_out)

    # Determine NIC type from nic_model parameter
    if re.search('ainic|pensando|amd', nic_model, re.I):
        nic_type = 'ainic'
    elif re.search('broadcom|thor|bnxt', nic_model, re.I):
        nic_type = 'thor'
    elif re.search('mellanox|cx|nvidia', nic_model, re.I):
        nic_type = 'connectx'
    else:
        nic_type = 'ainic'
    log.info(f'Detected NIC type: {nic_type} from nic_model: {nic_model}')

    # Convert aggregated results to format compatible with verification functions (using mean values)
    results_for_verification = []
    if aggregated_rccl_tests:
        for agg_result in aggregated_rccl_tests:
            results_for_verification.append(
                {
                    'name': agg_result.name,
                    'size': agg_result.size,
                    'type': agg_result.type,
                    'inPlace': agg_result.inPlace,
                    'busBw': agg_result.busBw_mean,
                    'algBw': agg_result.algBw_mean,
                    'time': agg_result.time_mean,
                }
            )
        log.info(f'Converted {len(results_for_verification)} aggregated results for verification')
    else:
        # Fallback to raw results if aggregation wasn't performed
        results_for_verification = all_raw_results
        log.info('Using raw results for verification (no aggregation performed)')

    # Build result key in format: test_name-data_types-global_ranks
    # Join all data types with underscores for the key
    dtypes_str = '_'.join(data_types)
    result_key = f'{test_name}-{dtypes_str}-{no_of_global_ranks}'
    log.info(f'Looking up results with key: {result_key} in nic_type: {nic_type}')

    # Get test-specific expected results from hierarchical structure
    test_exp_dict = None
    if exp_results_dict and isinstance(exp_results_dict, dict) and nic_type in exp_results_dict:
        if result_key in exp_results_dict[nic_type]:
            test_exp_dict = exp_results_dict[nic_type][result_key]
            log.info(f'Found expected results: {nic_type}/{result_key}')

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    if re.search('True', verify_bus_bw, re.I):
        if test_exp_dict:
            check_bus_bw(test_name, results_for_verification, test_exp_dict)
        else:
            log.warning(f'verify_bus_bw enabled but no expected results found for {result_key}')

    if re.search('True', verify_bw_dip, re.I):
        check_bw_dip(test_name, results_for_verification, test_exp_dict)

    if re.search('True', verify_lat_dip, re.I):
        check_lat_dip(test_name, results_for_verification, test_exp_dict)

    return all_raw_results


# Single node RCCL
#
def rccl_single_node_test(
    orch,
    test_name,
    cluster_node_list,
    rocm_path_var,
    rccl_dir,
    rccl_path_var,
    rccl_tests_dir,
    start_msg_size=1024,
    end_msg_size='16g',
    step_function=2,
    warmup_iterations=10,
    no_of_iterations=1,
    check_iteration_count=1,
    debug_level='INFO',
    rccl_result_file='/tmp/rccl_result_output.json',
    no_of_local_ranks=8,
    verify_bus_bw=False,
    verify_bw_dip=True,
    verify_lat_dip=True,
    exp_results_dict=None,
    env_source_script=None,
):
    """
    Run an Single Node RCCL collective test

    Arguments:
      orch: Orchestrator instance for test execution.
      test_name: RCCL test binary name (e.g., all_reduce_perf).
      cluster_node_list: List of cluster node hostnames/IPs
      rocm_path_var, rccl_dir, rccl_path_var, rccl_tests_dir: Installation paths.
      start_msg_size, end_msg_size, step_function: Message size sweep setup.
      threads_per_gpu, warmup_iterations, check_iteration_count: Test execution tuning.
      debug_level: NCCL_DEBUG level.
      rccl_result_file: Path where the RCCL test writes JSON results (-Z json -x file).
      verify_bus_bw: If 'True' (string), compare bus BW vs expected thresholds.
      exp_results_dict: Dict of expected results per test for verification.

    Returns:
      result_out: The raw JSON string read from rccl_result_file on all nodes
    """

    print(f'Starting RCCL Test ..........................................{test_name}')
    # Base ROCm path as provided by caller
    ROCM_PATH = rocm_path_var

    RCCL_PATH = f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR = f'{rccl_tests_dir}'

    head_node = cluster_node_list[0]

    # Environment variables exported into the mpirun context
    PATH = f'{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH = f'{RCCL_PATH}:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    # Build the test command
    # Wrap test binary in shell to source env script if provided
    test_cmd = f'env && {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {no_of_local_ranks} -c {check_iteration_count} -w {warmup_iterations} \
        -Z json -x {rccl_result_file}'

    if env_source_script and env_source_script.lower() != 'none':
        test_cmd = f'bash -c "source {env_source_script} && {test_cmd}"'
    else:
        # Always wrap in bash to interpret && shell operator
        test_cmd = f'bash -c "{test_cmd}"'

    cmd = f'''export NCCL_DEBUG={debug_level};  \
           export PATH={PATH}; \
           export LD_LIBRARY_PATH={LD_LIBRARY_PATH}; \
           {test_cmd}'''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')
    try:
        out_dict = orch.exec(cmd, hosts=[head_node], timeout=500)
        for node in out_dict.keys():
            scan_rccl_logs(out_dict[node])
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {repr(e)}')
        fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {repr(e)}')

    # Read the JSON results emitted by the RCCL test binary
    result_dict_out = orch.exec(f'cat {rccl_result_file}', hosts=[head_node])
    result_out = json.loads(result_dict_out[head_node].replace('\n', '').replace('\r', ''))

    # Collect basic GPU information via rocm-smi
    orch.exec('rocm-smi -a | head -30', hosts=[head_node])

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None

    if re.search('True', verify_bus_bw, re.I):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace('\n', '').replace('\r', ''))
            if test_exp_dict:
                check_bus_bw(test_name, result_out, test_exp_dict)

    if re.search('True', verify_bw_dip, re.I):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace('\n', '').replace('\r', ''))
            check_bw_dip(test_name, result_out, test_exp_dict)

    if re.search('True', verify_lat_dip, re.I):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace('\n', '').replace('\r', ''))
            check_lat_dip(test_name, result_out, test_exp_dict)

    return result_out
