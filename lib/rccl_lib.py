'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import sys
import os

import globals

log = globals.log

from utils_lib import *
from verify_lib import *



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
      - Calls fail_test(...) if any measurement is below expectation.

    Notes:
      - Message sizes are compared as strings to avoid type mismatches between JSON and expectations.
      - Assumes fail_test(...) is available in scope to signal test failure.
    """
    actual_bw_dict = {}
    msg_size_list = list(exp_res_dict['bus_bw'].keys())
    print(test_name)
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        if float(act_dict['busBw']) < float(exp_res_dict['bus_bw'][msg_size]):
                            fail_test(f"The actual out-of-place bus BW {act_dict['busBw']} for msg size {act_dict['size']} is lower than expected bus BW {exp_res_dict['bus_bw'][msg_size]}")
    else:
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                for msg_size in msg_size_list:
                    if str(msg_size) == str(act_dict['size']):
                        if float(act_dict['busBw']) < float(exp_res_dict['bus_bw'][msg_size]):
                            fail_test(f"The actual out-of-place bus BW {act_dict['busBw']} for msg size {act_dict['size']} is lower than expected bus BW {exp_res_dict['bus_bw'][msg_size]}")

 



def check_bw_dip( test_name, output, exp_res_dict ):
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                if float(act_dict['busBw']) < float(last_bw):
                    fail_test(f"The BusBW for msg size {act_dict['size']} = {act_dict['busBw']} is less than the earlier msg size {last_msg_size} = BW {last_bw}")
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']
    else:
        last_bw = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                if float(act_dict['busBw']) < float(last_bw):
                    fail_test(f"The BusBW for msg size {act_dict['size']} = {act_dict['busBw']} is less than the earlier msg size {last_msg_size} = BW {last_bw}")
                last_bw = act_dict['busBw']
                last_msg_size = act_dict['size']



def check_lat_dip( test_name, output, exp_res_dict ):
    #act_res_dict = json.loads(output.replace( '\n', '').replace( '\r', ''))
    act_res_dict = output
    if re.search( 'alltoall|all_to_all', test_name, re.I ):
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 0:
                if float(act_dict['time']) < float(last_time):
                    fail_test(f"The latency for msg size {act_dict['size']} = {act_dict['time']} is less than the earlier msg size {last_msg_size} = BW {last_time}")
                last_time = act_dict['time']
                last_msg_size = act_dict['size']
    else:
        last_time = 0.0
        last_msg_size = act_res_dict[0]['size']
        for act_dict in act_res_dict:
            if act_dict['inPlace'] == 1:
                if float(act_dict['time']) < float(last_time):
                    fail_test(f"The latency for msg size {act_dict['size']} = {act_dict['time']} is less than the earlier msg size {last_msg_size} = BW {last_time}")
                last_time = act_dict['time']
                last_msg_size = act_dict['size']






def convert_to_graph_dict(result_dict):
    graph_dict = {}
    for graph_series_name in result_dict.keys():
        graph_dict[graph_series_name] = {}
        dict_list = result_dict[graph_series_name]
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
    head_node = cluster_node_list[0]
    host_params=''
    proc_per_node = int(int(no_of_global_ranks)/len(cluster_node_list))
    for node in vpc_node_list:
        host_params = f'{host_params}{node}:{proc_per_node},'


    # Compute processes per node and build the -H host mapping string: host:N,host:N,...
    host_params = host_params.rstrip(',')
    print(f'RCCL Hosts -H value {host_params}')

        
    cmd = f'''{MPI_INSTALL_DIR}/mpirun --np {no_of_global_ranks} \
        --allow-run-as-root \
        -H {host_params} \
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
    if re.search( 'True', verify_bus_bw, re.I ):
        check_bus_bw( test_name, result_out, exp_results_dict[test_name] )

    if re.search( 'True', verify_bw_dip, re.I ):
        check_bw_dip( test_name, result_out, exp_results_dict[test_name] )

    if re.search( 'True', verify_lat_dip, re.I ):
        check_lat_dip( test_name, result_out, exp_results_dict[test_name] )

    return result_out
