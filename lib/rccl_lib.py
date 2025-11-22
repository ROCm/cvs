'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

#Standard libraries
import re
import sys
import os
import json
from typing import List, Dict
from pathlib import Path

#Third party libraries
import pandas as pd
from pydantic import ValidationError

import globals
from models.rccl import RcclTests,  RcclTestsAggregated, RcclTestsMultinodeRaw
from utils_lib import *
from verify_lib import *

log = globals.log


rccl_err_dict = {
   'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
   'nccl': 'NCCL ERROR|Test failure',
   'fs_err': 'No such file or directory'
}


def scan_rccl_logs( output ):
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
    error_list = []    # Accumulates lines that match known error patterns (for context/auditing)
    warn_list = []     # Accumulates NCCL warning lines (non-fatal but useful for visibility)

    # Process output line-by-line to catch and act on errors/warnings
    for line in output.split("\n"):
        for err_key in rccl_err_dict.keys():
            # Check each line against all known error signatures
            if re.search( f'{rccl_err_dict[err_key]}', line ):
                error_list.append(line)
                fail_test(f'ERROR - {line}')
        # Collect NCCL warnings (do not fail the test)
        if re.search('NCCL WARN', line ):
            warn_list.append(line)
    if len(warn_list) > 0:
        print('Following warnings were observed in the RCCL test')
        print('#============#')
        print(warn_list)
        print('#============#')
    if not re.search('#\sAvg bus bandwidth', output ):
        fail_test('RCCL test did not complete successfully, no bandwidth numbers printed - pls check')
  




# Not using the avg bus bandwidth verification currently ..
def check_avg_bus_bw( output, exp_res_dict ):
    if re.search('#\sAvg bus bandwidth\s+:\s+[0-9\.]+', output, re.I ):
        match = re.search('#\sAvg bus bandwidth\s+:\s+([0-9\.]+)', output, re.I )
        actual_bw = float(match.group(1))
        if actual_bw < float(exp_res_dict['avg_bus_bw']):
            fail_test(f"Actual Avg Bus BW {actual_bw} is less than the expected Avg BW {exp_res_dict['avg_bus_bw']}") 




def check_bus_bw( test_name, output, exp_res_dict ):
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
                      'bus_bw': {
                          <msg_size>: <min_expected_bus_bw>, ...
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
    actual_bw_dict = {}
    msg_size_list = list(exp_res_dict['bus_bw'].keys())
    tolerance = 0.95  # 5% tolerance
    print(test_name)
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        expected_bw = float(exp_res_dict['bus_bw'][msg_size])
                        actual_bw = float(act_dict['busBw'])
                        threshold = expected_bw * tolerance
                        if actual_bw < threshold:
                            fail_test(f"The actual out-of-place bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})")
    else:
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        expected_bw = float(exp_res_dict['bus_bw'][msg_size])
                        actual_bw = float(act_dict['busBw'])
                        threshold = expected_bw * tolerance
                        if actual_bw < threshold:
                            fail_test(f"The actual in-place bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})")

 



def check_bw_dip( test_name, output, exp_res_dict=None ):
    """
    Check for bandwidth dips as message size increases.
    Only fails if bandwidth drops by more than 5%.
    If exp_res_dict is provided, only validates message sizes specified in the reference.
    """
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance
    
    # Get reference message sizes if provided
    ref_msg_sizes = None
    if exp_res_dict and 'bus_bw' in exp_res_dict:
        ref_msg_sizes = set(str(size) for size in exp_res_dict['bus_bw'].keys())
        log.info(f"Validating BW dip only for reference message sizes: {ref_msg_sizes}")
    
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                # Skip validation if this message size is not in reference
                if ref_msg_sizes and str(act_dict['size']) not in ref_msg_sizes:
                    continue
                    
                current_bw = float(act_dict['busBw'])
                threshold = float(last_bw) * tolerance
                if last_bw > 0 and current_bw < threshold:
                    fail_test(f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})")
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']
    else:
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                # Skip validation if this message size is not in reference
                if ref_msg_sizes and str(act_dict['size']) not in ref_msg_sizes:
                    continue
                    
                current_bw = float(act_dict['busBw'])
                threshold = float(last_bw) * tolerance
                if last_bw > 0 and current_bw < threshold:
                    fail_test(f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})")
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']



def check_lat_dip( test_name, output, exp_res_dict=None ):
    """
    Check for latency decreases as message size increases (which would be unexpected).
    Only fails if latency drops by more than 5%.
    If exp_res_dict is provided, only validates message sizes specified in the reference.
    """
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance
    
    # Get reference message sizes if provided
    ref_msg_sizes = None
    if exp_res_dict and 'bus_bw' in exp_res_dict:
        ref_msg_sizes = set(str(size) for size in exp_res_dict['bus_bw'].keys())
        log.info(f"Validating latency dip only for reference message sizes: {ref_msg_sizes}")
    
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                # Skip validation if this message size is not in reference
                if ref_msg_sizes and str(act_dict['size']) not in ref_msg_sizes:
                    continue
                    
                current_time = float(act_dict['time'])
                threshold = float(last_time) * tolerance
                if last_time > 0 and current_time < threshold:
                    fail_test(f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})")
                last_time = act_dict['time']
                last_msg_size = act_dict['size']
    else:
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                # Skip validation if this message size is not in reference
                if ref_msg_sizes and str(act_dict['size']) not in ref_msg_sizes:
                    continue
                    
                current_time = float(act_dict['time'])
                threshold = float(last_time) * tolerance
                if last_time > 0 and current_time < threshold:
                    fail_test(f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})")
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
            if re.search( 'alltoall', dict_item['name'], re.I) and dict_item['inPlace'] == 1:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
            else:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
    print(graph_dict)
    return graph_dict




def aggregate_rccl_test_results(validated_results: List[RcclTests]) -> List[RcclTestsAggregated]:
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
            'gpusPerRank': first.gpusPerRank
        }
        
        # Validate all results have same config
        for i, result in enumerate(validated_results):
            if not isinstance(result, RcclTestsMultinodeRaw):
                raise ValueError(
                    f"Mixed single-node and multi-node results at index {i}"
                )
            if (result.nodes != multinode_config['nodes'] or
                result.ranks != multinode_config['ranks'] or
                result.ranksPerNode != multinode_config['ranksPerNode'] or
                result.gpusPerRank != multinode_config['gpusPerRank']):
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
        num_runs=('numCycle', 'count')
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
def rccl_cluster_test( phdl, shdl, test_name, cluster_node_list, vpc_node_list, user_name, ib_hca_list, \
        net_dev_list, oob_port, no_of_global_ranks, rocm_path_var, mpi_dir, mpi_path_var, \
        rccl_dir, rccl_path_var, rccl_tests_dir, nccl_algo='ring', \
        nccl_proto='simple', gid_index=1, qp_count=1, \
        start_msg_size=1024, end_msg_size='16g', \
        step_function=2, threads_per_gpu=1, warmup_iterations=10, no_of_iterations=1, \
        check_iteration_count=1, debug_level='INFO', \
        rccl_result_file='/tmp/rccl_result_output.json', no_of_local_ranks=8, \
        ib_rx_queue_len=8192, ucx_tls='tcp', hcoll_enable_mcast_all=0, \
        nccl_cumem_enable=0, nccl_ib_timeout=30, nccl_ib_sl=0, \
        nccl_ib_tc=41, nccl_ib_split_data_on_qps=0, nccl_pxn_disable=1, \
        nccl_net_plugin=None, user_password=None, \
        min_channels=64, max_channels=64, \
        data_type="float", \
        user_key_file=None, verify_bus_bw=False, \
        verify_bw_dip=True, verify_lat_dip=True, exp_results_dict=None ):


    """
    Run an RCCL collective test across a cluster via MPI and verify results.

    Arguments:
      phdl: Parallel ssh handle to run commands on all nodes.
      shdl: ssh handle to the first node in the cluster.
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

    Returns:
      result_out: The raw JSON string read from rccl_result_file on the head node.
    """

    print(f'Starting RCCL Test ..........................................{test_name}')
    # Base ROCm path as provided by caller
    ROCM_PATH=rocm_path_var

    # Resolve tool/library install locations
    #MPI_PATH=f'{mpi_path}/install/bin'
    MPI_PATH=f'{mpi_path_var}'
    MPI_INSTALL_DIR=f'{mpi_dir}'
    RCCL_INSTALL_DIR=f'{rccl_dir}'
    RCCL_PATH=f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR=f'{rccl_tests_dir}'


    # Environment variables exported into the mpirun context
    PATH=f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH=f'{RCCL_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')


    # Use the first cluster node as the head node (source for collected outputs)
    # The -H {host_params} is obsolete in ompi5.0 and greater, so changing to
    # --hostfile option
    head_node = cluster_node_list[0]
    #host_params=''
    #proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    #for node in vpc_node_list:
    #    host_params = f'{host_params}{node}:{proc_per_node},'
    # Compute processes per node and build the -H host mapping string: host:N,host:N,...
    #host_params = host_params.rstrip(',')
    #print(f'RCCL Hosts -H value {host_params}')

    host_file_params=''
    proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    for node in vpc_node_list:
        host_file_params = f'{host_file_params}' + f'{node} slots={proc_per_node}\n'

    cmd = 'sudo rm -f /tmp/rccl_hosts_file.txt'
    shdl.exec(cmd)

    cmd = f'echo "{host_file_params}" > /tmp/rccl_hosts_file.txt'
    shdl.exec(cmd)

        
    cmd = f'''{MPI_INSTALL_DIR}/mpirun --np {no_of_global_ranks} \
        --allow-run-as-root \
        --hostfile /tmp/rccl_hosts_file.txt \
        -x NCCL_DEBUG={debug_level} \
        --bind-to numa \
        -x NCCL_IB_GID_INDEX={gid_index} \
        -x UCX_UNIFIED_MODE=y \
        -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
        -x PATH={PATH} \
        -x LD_LIBRARY_PATH={LD_LIBRARY_PATH} \
        -x NCCL_IB_HCA={ib_hca_list} \
        --mca btl ^vader,openib \
        --mca btl_tcp_if_include {oob_port}\
        -x UCX_NET_DEVICES={net_dev_list} \
        -x NCCL_ALGO={nccl_algo} \
        -x NCCL_MIN_NCHANNELS={min_channels} \
        -x NCCL_MAX_NCHANNELS={max_channels} \
        -x NCCL_IB_QPS_PER_CONNECTION={qp_count} \
        -x IB_RX_QUEUE_LEN={ib_rx_queue_len} \
        -x UCX_TLS={ucx_tls} \
        -x HCOLL_ENABLE_MCAST_ALL={hcoll_enable_mcast_all} \
        -x NCCL_CUMEM_ENABLE={nccl_cumem_enable} \
        -x NCCL_IB_TIMEOUT={nccl_ib_timeout} \
        -x NCCL_IB_SL={nccl_ib_sl} \
        -x NCCL_IB_TC={nccl_ib_tc} \
        -x NCCL_IB_SPLIT_DATA_ON_QPS={nccl_ib_split_data_on_qps} \
        -x NCCL_PXN_DISABLE={nccl_pxn_disable} \
        -x NCCL_NET_PLUGIN={nccl_net_plugin} \
        {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {threads_per_gpu} -c {check_iteration_count} -w {warmup_iterations} \
        -d {data_type} \
        -Z json -x {rccl_result_file}
        '''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')
    try:
        out_dict = shdl.exec(cmd, timeout=500)
        output = out_dict[head_node]
        #print(output)
        scan_rccl_logs(output)
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')
        fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')

    # Read the JSON results emitted by the RCCL test binary
    result_dict_out = shdl.exec(f'cat {rccl_result_file}')
    result_out = json.loads(result_dict_out[head_node].replace( '\n', '').replace( '\r', ''))


    # Collect basic GPU information via rocm-smi
    smi_out_dict = shdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    model=get_model_from_rocm_smi_output(smi_out)

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None
    
    if re.search( 'True', verify_bus_bw, re.I ):
        if test_exp_dict:
            check_bus_bw( test_name, result_out, test_exp_dict )

    if re.search( 'True', verify_bw_dip, re.I ):
        check_bw_dip( test_name, result_out, test_exp_dict )

    if re.search( 'True', verify_lat_dip, re.I ):
        check_lat_dip( test_name, result_out, test_exp_dict )

    return result_out








# Main RCCL Test library which gets invoked from cvs/test/rccl tests and accepts most of the 
# standard NCCL environment variables ..
#
def rccl_cluster_test_default( phdl, shdl, test_name, cluster_node_list, vpc_node_list, user_name, ib_hca_list, \
        net_dev_list, oob_port, no_of_global_ranks, rocm_path_var, mpi_dir, mpi_path_var, \
        rccl_dir, rccl_path_var, rccl_tests_dir, nccl_algo='ring', \
        nccl_proto='simple', gid_index=1, qp_count=1, \
        start_msg_size=1024, end_msg_size='16g', \
        step_function=2, threads_per_gpu=1, warmup_iterations=10, no_of_iterations=1, \
        data_types=['float'], no_of_cycles=10, \
        check_iteration_count=1, debug_level='INFO', \
        rccl_result_file='/tmp/rccl_result_output.json', no_of_local_ranks=8, \
        ib_rx_queue_len=8192, ucx_tls='tcp', hcoll_enable_mcast_all=0, \
        nccl_cumem_enable=0, nccl_ib_timeout=30, nccl_ib_sl=0, \
        nccl_ib_tc=41, nccl_ib_split_data_on_qps=0, nccl_pxn_disable=1, \
        nccl_net_plugin=None, user_password=None, \
        min_channels=64, max_channels=64, \
        user_key_file=None, verify_bus_bw=False, \
        verify_bw_dip=True, verify_lat_dip=True, exp_results_dict=None ):


    """
    Run an RCCL collective test across a cluster via MPI and verify results.

    Arguments:
      phdl: Parallel ssh handle to run commands on all nodes.
      shdl: ssh handle to the first node in the cluster.
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
      debug_level: NCCL_DEBUG level.
      rccl_result_file: Path where the RCCL test writes JSON results (-Z json -x file).
      verify_bus_bw: If 'True' (string), compare bus BW vs expected thresholds.
      exp_results_dict: Dict of expected results per test for verification.

    Returns:
      all_raw_results: List of dictionaries containing all test results from all data types.
    """

    print(f'Starting RCCL Test ..........................................{test_name}')
    # Base ROCm path as provided by caller
    ROCM_PATH=rocm_path_var

    # Resolve tool/library install locations
    #MPI_PATH=f'{mpi_path}/install/bin'
    MPI_PATH=f'{mpi_path_var}'
    MPI_INSTALL_DIR=f'{mpi_dir}'
    RCCL_INSTALL_DIR=f'{rccl_dir}'
    RCCL_PATH=f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR=f'{rccl_tests_dir}'


    # Environment variables exported into the mpirun context
    PATH=f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH=f'{RCCL_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')


    # Use the first cluster node as the head node (source for collected outputs)
    # The -H {host_params} is obsolete in ompi5.0 and greater, so changing to
    # --hostfile option
    head_node = cluster_node_list[0]
    #host_params=''
    #proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    #for node in vpc_node_list:
    #    host_params = f'{host_params}{node}:{proc_per_node},'
    # Compute processes per node and build the -H host mapping string: host:N,host:N,...
    #host_params = host_params.rstrip(',')
    #print(f'RCCL Hosts -H value {host_params}')

    host_file_params=''
    proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    for node in vpc_node_list:
        host_file_params = f'{host_file_params}' + f'{node} slots={proc_per_node}\n'

    cmd = 'sudo rm -f /tmp/rccl_hosts_file.txt'
    shdl.exec(cmd)

    cmd = f'echo "{host_file_params}" > /tmp/rccl_hosts_file.txt'
    shdl.exec(cmd)

        
    all_raw_results = []
    all_validated_results = []
    base_path = Path(rccl_result_file)
    for dtype in data_types:
        # Create a unique result file for each data type
        dtype_result_file = f'{base_path.parent}/{base_path.stem}_{dtype}.json'
        cmd = f'''{MPI_INSTALL_DIR}/mpirun --np {no_of_global_ranks} \
        --allow-run-as-root \
        --hostfile /tmp/rccl_hosts_file.txt \
        -x NCCL_DEBUG={debug_level} \
        --bind-to numa \
        -x NCCL_IB_GID_INDEX={gid_index} \
        -x UCX_UNIFIED_MODE=y \
        -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
        -x PATH={PATH} \
        -x LD_LIBRARY_PATH={LD_LIBRARY_PATH} \
        -x NCCL_IB_HCA={ib_hca_list} \
        -x NCCL_SOCKET_IFNAME={oob_port} \
        --mca btl self,tcp \
        --mca btl_tcp_if_exclude lo,docker0,usb0 \
        -x UCX_NET_DEVICES={net_dev_list} \
        -x UCX_TLS={ucx_tls} \
        -x NCCL_NET_PLUGIN={nccl_net_plugin} \
        {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
        -g {threads_per_gpu} -c {check_iteration_count} -w {warmup_iterations} \
        -d {dtype} -N {no_of_cycles} -Z json -x {dtype_result_file}
        '''

        print('%%%%%%%%%%%%%%%%')
        print(cmd)
        print('%%%%%%%%%%%%%%%%')
        try:
            out_dict = shdl.exec(cmd, timeout=500)
            output = out_dict[head_node]
            #print(output)
            scan_rccl_logs(output)
        except Exception as e:
            log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')
            fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')

        # Read the JSON results emitted by the RCCL test binary
        result_dict_out = shdl.exec(f'cat {dtype_result_file}')
        dtype_result_out = json.loads(result_dict_out[head_node].replace( '\n', '').replace( '\r', ''))
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
            log.info(f'Aggregation skipped: only one run found')
    except ValidationError as e:
        log.error(f'Validation Failed: {e}')
        fail_test(f'RCCL Test schema validation failed: {e}')
    except ValueError as e:
        log.error(f'Aggregation failed: {e}')
        fail_test(f'RCCL Test aggregation failed: {e}')


    # Collect basic GPU information via rocm-smi
    smi_out_dict = shdl.exec('rocm-smi -a | head -30')
    smi_out = smi_out_dict[head_node]
    model=get_model_from_rocm_smi_output(smi_out)

    # Convert aggregated results to format compatible with verification functions (using mean values)
    results_for_verification = []
    if aggregated_rccl_tests:
        for agg_result in aggregated_rccl_tests:
            results_for_verification.append({
                'name': agg_result.name,
                'size': agg_result.size,
                'type': agg_result.type,
                'inPlace': agg_result.inPlace,
                'busBw': agg_result.busBw_mean,
                'algBw': agg_result.algBw_mean,
                'time': agg_result.time_mean
            })
        log.info(f'Converted {len(results_for_verification)} aggregated results for verification')
    else:
        # Fallback to raw results if aggregation wasn't performed
        results_for_verification = all_raw_results
        log.info('Using raw results for verification (no aggregation performed)')

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None
    
    if re.search( 'True', verify_bus_bw, re.I ):
        if test_exp_dict:
            check_bus_bw( test_name, results_for_verification, test_exp_dict )

    if re.search( 'True', verify_bw_dip, re.I ):
        check_bw_dip( test_name, results_for_verification, test_exp_dict )

    if re.search( 'True', verify_lat_dip, re.I ):
        check_lat_dip( test_name, results_for_verification, test_exp_dict )

    return all_raw_results






# Single node RCCL
#
def rccl_single_node_test( phdl, test_name, cluster_node_list, \
        rocm_path_var, rccl_dir, rccl_path_var, rccl_tests_dir, \
        start_msg_size=1024, end_msg_size='16g', \
        step_function=2, warmup_iterations=10, no_of_iterations=1, \
        check_iteration_count=1, debug_level='INFO', \
        rccl_result_file='/tmp/rccl_result_output.json', no_of_local_ranks=8, \
        verify_bus_bw=False, verify_bw_dip=True, verify_lat_dip=True, exp_results_dict=None ):

    """
    Run an Single Node RCCL collective test

    Arguments:
      phdl: Parallel ssh handle to run commands on all nodes.
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
    ROCM_PATH=rocm_path_var

    RCCL_INSTALL_DIR=f'{rccl_dir}'
    RCCL_PATH=f'{rccl_path_var}'
    RCCL_TESTS_INSTALL_DIR=f'{rccl_tests_dir}'

    head_node = cluster_node_list[0]

    # Environment variables exported into the mpirun context
    PATH=f'{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH=f'{RCCL_PATH}:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'


        
    cmd = f'''export NCCL_DEBUG={debug_level};  \
           export PATH={PATH}; \
           export LD_LIBRARY_PATH={LD_LIBRARY_PATH}; \
           {RCCL_TESTS_INSTALL_DIR}/{test_name} -b {start_msg_size} -e {end_msg_size} -f {step_function} \
           -g {no_of_local_ranks} -c {check_iteration_count} -w {warmup_iterations} \
           -Z json -x {rccl_result_file}'''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')
    try:
        out_dict = phdl.exec(cmd, timeout=500)
        for node in out_dict.keys():
            scan_rccl_logs(out_dict[node])
    except Exception as e:
        log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')
        fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {e}')

    # Read the JSON results emitted by the RCCL test binary
    result_dict_out = phdl.exec(f'cat {rccl_result_file}')
    result_out = json.loads(result_dict_out[head_node].replace( '\n', '').replace( '\r', ''))

    # Collect basic GPU information via rocm-smi
    smi_out_dict = phdl.exec('rocm-smi -a | head -30')

    # If requested, verify measured bus bandwidths against provided expected Bandwidth
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None
    
    if re.search( 'True', verify_bus_bw, re.I ):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace( '\n', '').replace( '\r', ''))
            if test_exp_dict:
                check_bus_bw( test_name, result_out, test_exp_dict )

    if re.search( 'True', verify_bw_dip, re.I ):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace( '\n', '').replace( '\r', ''))
            check_bw_dip( test_name, result_out, test_exp_dict )

    if re.search( 'True', verify_lat_dip, re.I ):
        for node in result_dict_out.keys():
            result_out = json.loads(result_dict_out[node].replace( '\n', '').replace( '\r', ''))
            check_lat_dip( test_name, result_out, test_exp_dict )

    return result_out
