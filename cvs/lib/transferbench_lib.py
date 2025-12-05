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
import csv
import io
from typing import List, Dict
from pathlib import Path

#Third party libraries
import pandas as pd
from pydantic import ValidationError

from cvs.lib import globals
from cvs.models.rccl import RcclTests, RcclTestsAggregated, RcclTestsMultinodeRaw
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


transferbench_err_dict = {
   'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
   'transferbench': 'ERROR|Test failure',
   'fs_err': 'No such file or directory'
}


def scan_transferbench_logs( output ):
    """
    Scan TransferBench test stdout for known error/warning patterns and enforce failure criteria.

    Parameters:
      output (str): Combined stdout/stderr text from a TransferBench test run.

    Behavior:
      - Iterates over each line to detect:
        * Errors matching patterns in transferbench_err_dict (e.g., ORTE/TransferBench/FS errors).
        * TransferBench WARN lines, which are collected and printed (but not fatal).
      - Fails the test immediately on the first matched error via fail_test(...).
      - After scanning, if no CSV output marker exists in the entire output,
        fails the test because results are considered incomplete.

    Notes:
      - Similar to scan_rccl_logs but adapted for TransferBench output patterns.
    """
    error_list = []
    warn_list = []

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for errors
        for err_type, pattern in transferbench_err_dict.items():
            if re.search(pattern, line, re.I):
                error_list.append(f"[{err_type.upper()}] {line}")
                break

        # Check for warnings (non-fatal)
        if re.search(r'WARN', line, re.I):
            warn_list.append(line)

    # Report warnings but don't fail
    if warn_list:
        log.warning(f"TransferBench warnings detected: {warn_list}")

    # Fail on errors
    if error_list:
        fail_test(f"TransferBench test failed with errors: {'; '.join(error_list)}")

    # Check for successful completion (CSV header should be present)
    if 'Test#,Transfer#' not in output:
        fail_test("TransferBench test did not produce expected CSV output - test may have failed")


def parse_transferbench_csv_to_json(csv_output: str) -> List[Dict]:
    """
    Parse TransferBench CSV output and convert to RCCL-compatible JSON format.

    TransferBench CSV format: Test#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),SrcAddr,DstAddr

    Args:
        csv_output (str): Raw CSV output from TransferBench

    Returns:
        List[Dict]: List of dictionaries in RCCL JSON format
    """
    results = []

    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(csv_output))

    for row in csv_reader:
        try:
            # Map TransferBench CSV fields to RCCL JSON structure
            result = {
                'numCycle': 1,  # TransferBench doesn't provide cycle count, default to 1
                'name': derive_transfer_name(row['Src'], row['Exe'], row['Dst']),
                'size': int(row['NumBytes']),
                'type': 'float',  # TransferBench typically uses float, adjust if needed
                'redop': 'sum',   # Default for bandwidth tests
                'inPlace': 0,     # TransferBench transfers are typically out-of-place
                'time': float(row['Time(ms)']) / 1000.0,  # Convert ms to seconds
                'algBw': float(row['BW(GB/s)']),  # TransferBench reports algorithmic bandwidth
                'busBw': float(row['BW(GB/s)']),  # For TransferBench, algBw ≈ busBw
                'wrong': 0  # TransferBench doesn't report errors in this format
            }
            results.append(result)
        except (ValueError, KeyError) as e:
            log.warning(f"Failed to parse TransferBench CSV row: {row}, error: {e}")
            continue

    return results


def derive_transfer_name(src: str, exe: str, dst: str) -> str:
    """
    Derive a human-readable transfer name from TransferBench Src/Exe/Dst fields.

    Args:
        src: Source memory type (e.g., 'G0', 'C0', 'N0')
        exe: Executor type (e.g., 'G', 'D', 'N')
        dst: Destination memory type (e.g., 'G1', 'C0', 'N0')

    Returns:
        str: Transfer name (e.g., 'gpu_to_gpu', 'cpu_to_gpu', 'nic_to_gpu')
    """
    # Extract memory types (first character indicates type)
    src_type = src[0].upper() if src else 'U'  # G=GPU, C=CPU, N=NIC, U=Unknown
    dst_type = dst[0].upper() if dst else 'U'

    # Map to readable names
    type_map = {
        'G': 'gpu',
        'C': 'cpu',
        'N': 'nic',
        'U': 'unknown'
    }

    src_name = type_map.get(src_type, 'unknown')
    dst_name = type_map.get(dst_type, 'unknown')

    # Handle same-type transfers
    if src_name == dst_name:
        return f"{src_name}_to_{dst_name}"
    else:
        return f"{src_name}_to_{dst_name}"


def check_bus_bw_transferbench( test_name, output, exp_res_dict ):
    """
    Check TransferBench bus bandwidth against expected thresholds.
    Adapted from RCCL check_bus_bw function for TransferBench CSV output.

    Parameters:
      test_name (str): Name of the transfer test being validated.
      output (list): List of dictionaries containing TransferBench results (already parsed from CSV).
      exp_res_dict (dict): Expected results dictionary with the structure:
                    {
                      <msg_size>: {
                          'bus_bw': <min_expected_bus_bw>, ...
                      }
                    }

    Behavior:
      - Parses the JSON output and iterates over measured entries.
      - Compares measured busBw to minimum expected thresholds per message size.
      - Calls fail_test(...) if any measurement is at least 5% below expectation.

    Notes:
      - Message sizes are compared as strings to avoid type mismatches between JSON and expectations.
      - Assumes fail_test(...) is available in scope to signal test failure.
      - 5% tolerance is applied: test only fails if actual < expected * 0.95
    """

    print( f'exp_res_dict = {exp_res_dict}')

    actual_bw_dict = {}
    tolerance = 0.95  # 5% tolerance

    # New hierarchical structure: {msg_size: {'bus_bw': bw_value}}
    msg_size_list = list(exp_res_dict.keys())

    print(test_name)
    act_res_dict = output
    for act_dict in act_res_dict:
        for msg_size in msg_size_list:
            if str(msg_size) == str(act_dict['size']):
                expected_bw = float(exp_res_dict[msg_size]['bus_bw'])
                actual_bw = float(act_dict['busBw'])
                threshold = expected_bw * tolerance
                print(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
                if actual_bw < threshold:
                    fail_test(f"The actual bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})")


def check_bw_dip_transferbench( test_name, output, exp_res_dict=None ):
    """
    Check for bandwidth dips in TransferBench results as message size increases.
    Only fails if bandwidth drops by more than 5%.
    Only validates message sizes specified in the reference. If no reference provided, skips validation.
    """
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance

    # Get reference message sizes if provided
    # If no reference data, skip validation entirely
    if not exp_res_dict:
        log.info(f"No reference data provided for BW dip check, skipping validation for {test_name}")
        return

    ref_msg_sizes = set(str(size) for size in exp_res_dict.keys())
    log.info(f"Validating BW dip only for reference message sizes: {ref_msg_sizes}")

    last_bw = 0.0
    last_msg_size = act_res_dict[0]['size']
    for act_dict in act_res_dict:
        # Skip validation if this message size is not in reference
        if str(act_dict['size']) not in ref_msg_sizes:
            continue

        current_bw = float(act_dict['busBw'])
        threshold = float(last_bw) * tolerance
        if last_bw > 0 and current_bw < threshold:
            fail_test(f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})")
        last_bw = act_dict['busBw']
        last_msg_size = act_dict['size']


def check_lat_dip_transferbench( test_name, output, exp_res_dict=None ):
    """
    Check for latency decreases in TransferBench results as message size increases (which would be unexpected).
    Only fails if latency drops by more than 5%.
    Only validates message sizes specified in the reference. If no reference provided, skips validation.
    """
    act_res_dict = output
    tolerance = 0.95  # 5% tolerance

    # Get reference message sizes if provided
    # If no reference data, skip validation entirely
    if not exp_res_dict:
        log.info(f"No reference data provided for latency dip check, skipping validation for {test_name}")
        return

    ref_msg_sizes = set(str(size) for size in exp_res_dict.keys())
    log.info(f"Validating latency dip only for reference message sizes: {ref_msg_sizes}")

    last_time = 0.0
    last_msg_size = act_res_dict[0]['size']
    for act_dict in act_res_dict:
        # Skip validation if this message size is not in reference
        if str(act_dict['size']) not in ref_msg_sizes:
            continue

        current_time = float(act_dict['time'])
        threshold = float(last_time) * tolerance
        if last_time > 0 and current_time < threshold:
            fail_test(f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})")
        last_time = act_dict['time']
        last_msg_size = act_dict['size']


def convert_transferbench_to_graph_dict(result_dict):
    """
    Convert TransferBench results to graph dictionary format for HTML reporting.
    Similar to convert_to_graph_dict but adapted for TransferBench data structure.
    """
    graph_dict = {}
    for graph_series_name in result_dict.keys():
        print(graph_series_name)
        graph_dict[graph_series_name] = {}
        dict_list = result_dict[graph_series_name]
        print(dict_list)
        for dict_item in dict_list:
            msg_size = dict_item['size']
            graph_dict[graph_series_name][msg_size] = {}
            graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
            graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
            graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
    print(graph_dict)
    return graph_dict


def aggregate_transferbench_test_results(validated_results: List[RcclTests]) -> List[RcclTestsAggregated]:
    """
    Aggregate multiple TransferBench test results into mean/std per (name, size, type, inPlace)
    Reuses RCCL aggregation logic since we convert TransferBench results to RCCL format.
    """
    return aggregate_rccl_test_results(validated_results)


# Main TransferBench Test library which gets invoked from cvs/test/transferbench tests
#
def transferbench_cluster_test( phdl, shdl, test_name, cluster_node_list, vpc_node_list, user_name, \
        transfer_configs, num_bytes, rocm_path_var, mpi_dir, mpi_path_var, \
        transferbench_dir, transferbench_path_var, transferbench_bin_dir, \
        num_iterations=10, num_warmups=5, \
        transferbench_result_file='/tmp/transferbench_result_output.json', \
        verify_bus_bw=False, verify_bw_dip=True, verify_lat_dip=True, exp_results_dict=None ):

    """
    Run TransferBench performance test across a cluster via MPI and verify results.

    Arguments:
      phdl: Parallel ssh handle to run commands on all nodes.
      shdl: ssh handle to the first node in the cluster.
      test_name: TransferBench test configuration name.
      cluster_node_list: List of cluster node hostnames/IPs (first is treated as head node).
      vpc_node_list: List of hostnames/IPs to pass to mpirun -H as hosts.
      user_name: Username for remote ops (unused here).
      transfer_configs: List of transfer configuration strings (e.g., ["1 4 (G0->G0->G1)"]).
      num_bytes: Number of bytes to transfer (0 for range sweep).
      rocm_path_var, mpi_dir, mpi_path_var: Installation paths.
      transferbench_dir, transferbench_path_var, transferbench_bin_dir: TransferBench installation paths.
      num_iterations, num_warmups: Test execution parameters.
      transferbench_result_file: Path where results will be saved in JSON format.
      verify_bus_bw: If 'True', compare bus BW vs expected thresholds.
      exp_results_dict: Dict of expected results per test for verification.

    Returns:
      result_out: List of dictionaries containing all test results.
    """

    print(f'Starting TransferBench Test ..........................................{test_name}')

    # Base ROCm path
    ROCM_PATH = rocm_path_var

    # Resolve tool/library install locations
    MPI_PATH = mpi_path_var
    MPI_INSTALL_DIR = mpi_dir
    TRANSFERBENCH_PATH = transferbench_path_var
    TRANSFERBENCH_BIN_DIR = transferbench_bin_dir

    # Environment variables exported into the mpirun context
    PATH = f'{MPI_PATH}/bin:{ROCM_PATH}/bin:$PATH'
    LD_LIBRARY_PATH = f'{TRANSFERBENCH_PATH}:{MPI_PATH}/lib:{ROCM_PATH}/lib:$LD_LIBRARY_PATH'

    print(f'%% VPC Node IPs {vpc_node_list}')

    # Use the first cluster node as the head node
    head_node = cluster_node_list[0]

    # Create hostfile for mpirun
    host_file_params = ''
    proc_per_node = 1  # TransferBench typically runs 1 process per node
    for node in vpc_node_list:
        host_file_params = f'{host_file_params}' + f'{node} slots={proc_per_node}\n'

    cmd = 'sudo rm -f /tmp/transferbench_hosts_file.txt'
    shdl.exec(cmd)

    cmd = f'echo "{host_file_params}" > /tmp/transferbench_hosts_file.txt'
    shdl.exec(cmd)

    # Create temporary config file with transfer configurations
    config_content = '\n'.join(transfer_configs)
    cmd = f'echo "{config_content}" > /tmp/transferbench_config.txt'
    shdl.exec(cmd)

    # Build TransferBench command
    cmd = f'''{MPI_INSTALL_DIR}/mpirun --np {len(cluster_node_list)} \
        --allow-run-as-root \
        --hostfile /tmp/transferbench_hosts_file.txt \
        -x PATH={PATH} \
        -x LD_LIBRARY_PATH={LD_LIBRARY_PATH} \
        -x OUTPUT_TO_CSV=1 \
        -x NUM_ITERATIONS={num_iterations} \
        -x NUM_WARMUPS={num_warmups} \
        {TRANSFERBENCH_BIN_DIR}/TransferBench /tmp/transferbench_config.txt {num_bytes}
        '''

    print('%%%%%%%%%%%%%%%%')
    print(cmd)
    print('%%%%%%%%%%%%%%%%')

    try:
        out_dict = shdl.exec(cmd, timeout=600)  # Longer timeout for TransferBench
        output = out_dict[head_node]
        print(output)
        scan_transferbench_logs(output)
    except Exception as e:
        log.error(f'Hit Exceptions with TransferBench cmd {cmd} - exception {repr(e)}')
        fail_test(f'Hit Exceptions with TransferBench cmd {cmd} - exception {repr(e)}')

    # Parse CSV output and convert to JSON
    result_out = parse_transferbench_csv_to_json(output)

    # Save the results to a JSON file for consistency with RCCL
    with open(transferbench_result_file, 'w') as f:
        json.dump(result_out, f, indent=2)
    log.info(f'Saved TransferBench results to {transferbench_result_file}')

    # Validate results if expected results provided
    if re.search( 'True', verify_bus_bw, re.I ):
        if exp_results_dict:
            check_bus_bw_transferbench( test_name, result_out, exp_results_dict )

    if re.search( 'True', verify_bw_dip, re.I ):
        check_bw_dip_transferbench( test_name, result_out, exp_results_dict )

    if re.search( 'True', verify_lat_dip, re.I ):
        check_lat_dip_transferbench( test_name, result_out, exp_results_dict )

    return result_out