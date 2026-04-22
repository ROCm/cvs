'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Standard libraries
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Third party libraries
import pandas as pd
from pydantic import ValidationError

from cvs.lib import globals
from cvs.schema.rccl import RcclTests, RcclTestsAggregated, RcclTestsMultinodeRaw
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log


rccl_err_dict = {
    'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
    'nccl': 'NCCL ERROR|Test failure',
    'fs_err': 'No such file or directory',
}


def _is_severe_wrong_corruption_error(err: ValidationError) -> bool:
    """
    Detect the rccl-tests '#wrong' corruption failure from a pydantic ValidationError.
    This is used to provide explicit, high-signal feedback to the user.
    """
    # Prefer structured parsing when available
    try:
        for item in err.errors():
            msg = item.get('msg', '') or ''
            if 'SEVERE DATA CORRUPTION' in msg or "'#wrong'" in msg or 'wrong=' in msg:
                return True
    except Exception:
        pass

    # Fallback to string search
    s = str(err)
    return 'SEVERE DATA CORRUPTION' in s or "'#wrong'" in s


def is_ucx_available_in_mpi(shdl, mpi_path, head_node):
    """
    Check if UCX is available in the OpenMPI build.

    Parameters:
      shdl: SSH handle to execute commands on the remote node.
      mpi_path: Path to the MPI installation directory.
      head_node: The head node hostname for retrieving command output.

    Returns:
      bool: True if UCX is available, False otherwise.
    """
    # First, try ompi_info
    check_ucx_cmd = f'{mpi_path}/bin/ompi_info | grep "pml: ucx" | wc -l'
    try:
        ucx_check_out = shdl.exec(check_ucx_cmd)
        ucx_available = int(ucx_check_out[head_node].strip()) > 0
        log.info("UCX available in OpenMPI build" if ucx_available else "UCX not available in OpenMPI build")
        return ucx_available
    except Exception as e:
        log.warning(f"ompi_info check failed: {e}; falling back to ldd check")

    # Fallback: check if libmpi.so is linked to UCX
    libmpi_path = f'{mpi_path}/lib/libmpi.so'
    check_ldd_cmd = f'ldd {libmpi_path} | grep ucx | wc -l'
    try:
        ldd_out = shdl.exec(check_ldd_cmd)
        ucx_available = int(ldd_out[head_node].strip()) > 0
        log.info("UCX linked in libmpi.so" if ucx_available else "UCX not linked in libmpi.so")
        return ucx_available
    except Exception as e:
        log.warning(f"ldd check failed: {e}; assuming UCX is not available")
        return False


def determine_mpi_pml_config(mpi_pml, shdl, mpi_path, head_node, net_dev_list, ucx_tls):
    """
    Determine MPI PML (Point-to-Point Messaging Layer) configuration based on user config or auto-detection.

    Parameters:
      mpi_pml: User-specified PML mode ('auto', 'ucx', or 'ob1').
      shdl: SSH handle to execute commands on the remote node.
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
        ucx_available = is_ucx_available_in_mpi(shdl, mpi_path, head_node)
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
        ucx_available = is_ucx_available_in_mpi(shdl, mpi_path, head_node)
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
        log.warning('Following warnings were observed in the RCCL test')
        log.warning('#============#')
        log.warning('%s', warn_list)
        log.warning('#============#')
    if not re.search(r'#\sAvg bus bandwidth', output):
        fail_test('RCCL test did not complete successfully, no bandwidth numbers printed - pls check')


# Not using the avg bus bandwidth verification currently ..
def check_avg_bus_bw(output, exp_res_dict):
    if re.search(r'#\sAvg bus bandwidth\s+:\s+[0-9\.]+', output, re.I):
        match = re.search(r'#\sAvg bus bandwidth\s+:\s+([0-9\.]+)', output, re.I)
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

    log.info(f'exp_res_dict = {exp_res_dict}')

    tolerance = 0.95  # 5% tolerance

    # New hierarchical structure: {msg_size: {'bus_bw': bw_value}}
    msg_size_list = list(exp_res_dict.keys())

    log.info("%s", test_name)
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
                        log.info(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
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
                        log.info(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
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
        log.warning(f"No reference data provided for BW dip check, skipping validation for {test_name}")
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
        log.warning(f"No reference data provided for latency dip check, skipping validation for {test_name}")
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
        log.info("%s", graph_series_name)
        graph_dict[graph_series_name] = {}
        dict_list = result_dict[graph_series_name]
        log.info("%s", dict_list)
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
    log.info("%s", graph_dict)
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
            log.error("%s", error_msg)
            errors.append(error_msg)

    # Report any validation failures
    if errors:
        error_summary = "\n".join(errors)
        fail_test(f"Aggregation validation failed:\n{error_summary}")

    log.info(f"Successfully validated {len(agg_results)} aggregated results")
    return agg_results


def _is_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value) and re.search('true', str(value), re.I) is not None


def _has_env_source_script(env_source_script) -> bool:
    return bool(env_source_script) and not re.search('none', str(env_source_script), re.I)


def _build_rccl_test_cmd(
    rccl_tests_dir,
    test_name,
    result_file,
    *,
    start_msg_size,
    end_msg_size,
    step_function,
    gpu_count,
    check_iteration_count,
    warmup_iterations,
    no_of_iterations,
    no_of_cycles,
    env_source_script,
    data_type=None,
):
    cmd_parts = [
        f'{rccl_tests_dir}/{test_name}',
        f'-b {start_msg_size}',
        f'-e {end_msg_size}',
        f'-f {step_function}',
        f'-g {gpu_count}',
        f'-c {check_iteration_count}',
        f'-w {warmup_iterations}',
    ]
    if data_type is not None:
        cmd_parts.append(f'-d {data_type}')
    cmd_parts.extend(
        [
            f'-n {no_of_iterations}',
            f'-N {no_of_cycles}',
            '-Z json',
            f'-X {result_file}',
        ]
    )

    command = f"env && {' '.join(cmd_parts)}"
    if _has_env_source_script(env_source_script):
        command = f'source {shlex.quote(env_source_script)} && {command}'
    return f'bash -lc {shlex.quote(command)}'


def _write_rccl_hostfile(shdl, vpc_node_list, cluster_node_list, no_of_global_ranks):
    host_file = '/tmp/rccl_hosts_file.txt'
    proc_per_node = int(int(no_of_global_ranks) / len(cluster_node_list))
    host_entries = ''.join(f'{node} slots={proc_per_node}\n' for node in vpc_node_list)

    shdl.exec(f'sudo rm -f {host_file}')
    shdl.exec(f'printf %s {shlex.quote(host_entries)} > {host_file}')
    return host_file


def _build_cluster_cmd(
    *,
    mpi_dir,
    no_of_global_ranks,
    host_file,
    debug_level,
    gid_index,
    ucx_params,
    path_env,
    ld_library_path,
    ib_hca_list,
    nccl_socket_ifname,
    oob_port,
    pml_param,
    nccl_net_plugin,
    min_channels,
    max_channels,
    test_cmd,
    use_explicit_tuning,
    nccl_algo,
    qp_count,
    ib_rx_queue_len,
    hcoll_enable_mcast_all,
    nccl_cumem_enable,
    nccl_ib_timeout,
    nccl_ib_sl,
    nccl_ib_tc,
    nccl_ib_split_data_on_qps,
    nccl_pxn_disable,
):
    cmd_parts = [
        f'{mpi_dir}/mpirun',
        f'--np {no_of_global_ranks}',
        '--allow-run-as-root',
        f'--hostfile {host_file}',
        f'-x NCCL_DEBUG={debug_level}',
        '--bind-to numa',
        f'-x NCCL_IB_GID_INDEX={gid_index}',
        ucx_params,
        '-x NCCL_IB_PCI_RELAXED_ORDERING=1',
        f'-x PATH={path_env}',
        f'-x LD_LIBRARY_PATH={ld_library_path}',
        f'-x NCCL_IB_HCA={ib_hca_list}',
        f'-x NCCL_SOCKET_IFNAME={nccl_socket_ifname}' if str(nccl_socket_ifname).strip() else '',
        '--mca btl ^vader,openib',
        f'--mca btl_tcp_if_include {oob_port}',
        f'--mca oob_tcp_if_include {oob_port}',
        pml_param,
        f'-x NCCL_NET_PLUGIN={nccl_net_plugin}',
        f'-x NCCL_MIN_NCHANNELS={min_channels}' if min_channels is not None else '',
        f'-x NCCL_MAX_NCHANNELS={max_channels}' if max_channels is not None else '',
    ]

    if use_explicit_tuning:
        cmd_parts.extend(
            [
                f'-x NCCL_ALGO={nccl_algo}',
                f'-x NCCL_IB_QPS_PER_CONNECTION={qp_count}',
                f'-x IB_RX_QUEUE_LEN={ib_rx_queue_len}',
                f'-x HCOLL_ENABLE_MCAST_ALL={hcoll_enable_mcast_all}',
                f'-x NCCL_CUMEM_ENABLE={nccl_cumem_enable}',
                f'-x NCCL_IB_TIMEOUT={nccl_ib_timeout}',
                f'-x NCCL_IB_SL={nccl_ib_sl}',
                f'-x NCCL_IB_TC={nccl_ib_tc}',
                f'-x NCCL_IB_SPLIT_DATA_ON_QPS={nccl_ib_split_data_on_qps}',
                f'-x NCCL_PXN_DISABLE={nccl_pxn_disable}',
            ]
        )

    cmd_parts.append(test_cmd)
    return ' '.join(part for part in cmd_parts if part)


def _scan_outputs(out_dict, *, head_node=None):
    if head_node:
        scan_rccl_logs(out_dict[head_node])
        return

    for output in out_dict.values():
        scan_rccl_logs(output)


def _load_result_map(exec_handle, result_file):
    result_dict_out = exec_handle.exec(f'cat {result_file}')
    return {node: json.loads(output.strip()) for node, output in result_dict_out.items()}


def _validate_multinode_results(raw_results, dtype, result_file):
    try:
        validated = [RcclTestsMultinodeRaw.model_validate(test_result) for test_result in raw_results]
        log.info(f'Validation passed: {len(validated)} RcclTests schema validation passed')
        return validated
    except ValidationError as e:
        if _is_severe_wrong_corruption_error(e):
            msg = (
                "\n"
                "==================== SEVERE DATA CORRUPTION ====================\n"
                "RCCL rccl-tests JSON schema validation failed due to '#wrong' > 0.\n"
                "This indicates invalid/corrupted rccl-tests results.\n"
                "\n"
                f"data_type: {dtype}\n"
                f"result_file: {result_file}\n"
                "\n"
                "Action: aborting further RCCL iterations/data types.\n"
                "Please inspect the rccl-tests stdout/stderr and re-run.\n"
                "================================================================\n"
            )
            log.error("%s", msg)
            fail_test(msg)
        else:
            log.error(f'Validation Failed: {e}')
            fail_test(f'RCCL Test {dtype} schema validation failed: {e}')

        raise RuntimeError(f'RCCL Test {dtype} schema validation failed') from e


def _resolve_nic_type(nic_model):
    if re.search('ainic|pensando|amd', nic_model, re.I):
        return 'ainic'
    if re.search('broadcom|thor|bnxt', nic_model, re.I):
        return 'thor'
    if re.search('mellanox|cx|nvidia', nic_model, re.I):
        return 'connectx'
    return 'ainic'


def _aggregate_results(all_raw_results, all_validated_results, rccl_result_file):
    base_path = Path(rccl_result_file)
    with open(rccl_result_file, 'w') as result_file:
        json.dump(all_raw_results, result_file, indent=2)
    log.info(f'Saved combined results from all data types to {rccl_result_file}')

    if not all_validated_results:
        log.warning('Aggregation skipped: no validated results found')
        return None

    try:
        aggregated_rccl_tests = aggregate_rccl_test_results(all_validated_results)
        log.info(f'Aggregation passed: {len(aggregated_rccl_tests)} RcclTestsAggregated schema validation passed')
        aggregated_path = f'{base_path.parent}/{base_path.stem}_aggregated.json'
        with open(aggregated_path, 'w') as aggregated_file:
            json.dump([result.model_dump() for result in aggregated_rccl_tests], aggregated_file, indent=2)
        log.info(f'Saved aggregated results to {aggregated_path}')
        return aggregated_rccl_tests
    except ValidationError as e:
        log.error(f'Validation Failed: {e}')
        fail_test(f'RCCL Test schema validation failed: {e}')
    except ValueError as e:
        log.error(f'Aggregation failed: {e}')
        fail_test(f'RCCL Test aggregation failed: {e}')

    return None


def _verify_flat_results(test_name, results, exp_results_dict, *, verify_bus_bw, verify_bw_dip, verify_lat_dip):
    test_exp_dict = exp_results_dict.get(test_name) if exp_results_dict else None

    if _is_enabled(verify_bus_bw) and test_exp_dict:
        check_bus_bw(test_name, results, test_exp_dict)

    if _is_enabled(verify_bw_dip):
        check_bw_dip(test_name, results, test_exp_dict)

    if _is_enabled(verify_lat_dip):
        check_lat_dip(test_name, results, test_exp_dict)


def _verify_aggregated_results(
    test_name,
    aggregated_results,
    raw_results,
    exp_results_dict,
    *,
    data_types,
    no_of_global_ranks,
    nic_model,
    verify_bus_bw,
    verify_bw_dip,
    verify_lat_dip,
):
    if aggregated_results:
        results_for_verification = [
            {
                'name': result.name,
                'size': result.size,
                'type': result.type,
                'inPlace': result.inPlace,
                'busBw': result.busBw_mean,
                'algBw': result.algBw_mean,
                'time': result.time_mean,
            }
            for result in aggregated_results
        ]
        log.info(f'Converted {len(results_for_verification)} aggregated results for verification')
    else:
        results_for_verification = raw_results
        log.info('Using raw results for verification (no aggregation performed)')

    nic_type = _resolve_nic_type(nic_model)
    log.info(f'Detected NIC type: {nic_type} from nic_model: {nic_model}')

    result_key = f"{test_name}-{'_'.join(data_types)}-{no_of_global_ranks}"
    log.info(f'Looking up results with key: {result_key} in nic_type: {nic_type}')

    test_exp_dict = None
    if exp_results_dict and isinstance(exp_results_dict, dict):
        test_exp_dict = exp_results_dict.get(nic_type, {}).get(result_key)
        if test_exp_dict:
            log.info(f'Found expected results: {nic_type}/{result_key}')

    if _is_enabled(verify_bus_bw):
        if test_exp_dict:
            check_bus_bw(test_name, results_for_verification, test_exp_dict)
        else:
            log.warning(f'verify_bus_bw enabled but no expected results found for {result_key}')

    if _is_enabled(verify_bw_dip):
        check_bw_dip(test_name, results_for_verification, test_exp_dict)

    if _is_enabled(verify_lat_dip):
        check_lat_dip(test_name, results_for_verification, test_exp_dict)


@dataclass(frozen=True)
class MpiRuntime:
    path_env: str
    ld_library_path: str
    host_file: str
    pml_param: str
    ucx_params: str


@dataclass(frozen=True)
class RcclMpiRunSpec:
    data_types: tuple[str, ...] | None = None
    result_file: str | None = None
    aggregate: bool = False
    min_channels: int | None = None
    max_channels: int | None = None
    nccl_algo: str | None = None
    qp_count: int | None = None
    nccl_pxn_disable: int | None = None
    nic_model: str | None = None


class RcclTestRunner:
    """
    Small RCCL MPI runner used by the RCCL pytest modules.
    """

    def __init__(
        self,
        *,
        config,
        cluster_node_list,
        shdl=None,
        vpc_node_list=None,
        no_of_global_ranks=None,
    ):
        self.shdl = shdl
        self.config = config
        self.cluster_node_list = cluster_node_list
        self.vpc_node_list = list(vpc_node_list or [])
        self.no_of_global_ranks = no_of_global_ranks if no_of_global_ranks is not None else config.get('no_of_global_ranks')
        self.head_node = cluster_node_list[0]

    def run(self, test_name, spec: RcclMpiRunSpec | None = None):
        spec = spec or RcclMpiRunSpec()
        log.info(f'Starting RCCL Test ..........................................{test_name}')
        if spec.min_channels is not None and spec.max_channels is not None:
            log.info(f'Using NCCL channels: min={spec.min_channels}, max={spec.max_channels}')
        elif spec.aggregate:
            log.info('Using RCCL default NCCL channel configuration')

        runtime = self._prepare_mpi_runtime()
        result_file = self._result_file(spec.result_file)
        active_data_types = self._data_types(spec)

        all_raw_results = []
        all_validated_results = []
        for data_type in active_data_types:
            dtype_result_file = self._dtype_result_file(result_file, active_data_types, data_type)
            raw_results = self._run_mpi_case(
                runtime,
                test_name,
                result_file=dtype_result_file,
                data_type=data_type,
                spec=spec,
            )
            all_raw_results.extend(raw_results)
            all_validated_results.extend(_validate_multinode_results(raw_results, data_type, dtype_result_file))

        self._collect_gpu_info()
        if spec.aggregate:
            aggregated_results = _aggregate_results(all_raw_results, all_validated_results, result_file)
            _verify_aggregated_results(
                test_name,
                aggregated_results,
                all_raw_results,
                self.config.get('results'),
                data_types=active_data_types,
                no_of_global_ranks=self.no_of_global_ranks,
                nic_model=spec.nic_model or self.config.get('nic_model', 'ainic'),
                verify_bus_bw=self.config.get('verify_bus_bw', False),
                verify_bw_dip=self.config.get('verify_bw_dip', True),
                verify_lat_dip=self.config.get('verify_lat_dip', True),
            )
            return all_raw_results

        self._verify_flat(test_name, all_raw_results)
        return all_raw_results

    def _result_file(self, override=None):
        return override or self.config.get('rccl_result_file', '/tmp/rccl_result_output.json')

    def _configured_data_types(self):
        for key in ('data_types', 'data_type_list'):
            data_types = self.config.get(key)
            if data_types:
                return list(data_types)
        if self.config.get('data_type'):
            return [self.config['data_type']]
        return ['float']

    def _data_types(self, spec: RcclMpiRunSpec):
        if spec.data_types:
            return list(spec.data_types)
        configured_data_types = self._configured_data_types()
        return configured_data_types if spec.aggregate else [configured_data_types[0]]

    def _dtype_result_file(self, result_file, data_types, data_type):
        if len(data_types) == 1:
            return result_file
        base_path = Path(result_file)
        return f'{base_path.parent}/{base_path.stem}_{data_type}.json'

    def _uses_explicit_tuning(self, spec: RcclMpiRunSpec):
        return any(value is not None for value in (spec.nccl_algo, spec.qp_count, spec.nccl_pxn_disable))

    def _base_env(self):
        rocm_path = self.config['rocm_path_var']
        rccl_path = self.config['rccl_path_var']
        mpi_path = self.config['mpi_path_var']
        path_env = f'{mpi_path}/bin:{rocm_path}/bin:$PATH'
        ld_library_path = f'{rccl_path}:{mpi_path}/lib:{rocm_path}/lib:{rocm_path}/lib64:{rocm_path}/hip/lib:$LD_LIBRARY_PATH'
        return path_env, ld_library_path

    def _prepare_mpi_runtime(self):
        if self.shdl is None:
            raise ValueError('MPI RCCL tests require shdl')
        if not self.vpc_node_list:
            raise ValueError('MPI RCCL tests require vpc_node_list')
        if self.no_of_global_ranks is None:
            raise ValueError('MPI RCCL tests require no_of_global_ranks')

        log.info(f'%% VPC Node IPs {self.vpc_node_list}')
        path_env, ld_library_path = self._base_env()
        host_file = _write_rccl_hostfile(self.shdl, self.vpc_node_list, self.cluster_node_list, self.no_of_global_ranks)
        pml_param, ucx_params = determine_mpi_pml_config(
            self.config.get('mpi_pml', 'auto'),
            self.shdl,
            self.config['mpi_path_var'],
            self.head_node,
            self.config['net_dev_list'],
            self.config.get('ucx_tls', 'tcp'),
        )
        return MpiRuntime(
            path_env=path_env,
            ld_library_path=ld_library_path,
            host_file=host_file,
            pml_param=pml_param,
            ucx_params=ucx_params,
        )

    def _test_command(self, test_name, result_file, *, gpu_count, data_type=None):
        return _build_rccl_test_cmd(
            self.config['rccl_tests_dir'],
            test_name,
            result_file,
            start_msg_size=self.config.get('start_msg_size', 1024),
            end_msg_size=self.config.get('end_msg_size', '16g'),
            step_function=self.config.get('step_function', 2),
            gpu_count=gpu_count,
            check_iteration_count=self.config.get('check_iteration_count', 1),
            warmup_iterations=self.config.get('warmup_iterations', 10),
            no_of_iterations=self.config.get('no_of_iterations', 20),
            no_of_cycles=self.config.get('no_of_cycles', 1),
            env_source_script=self.config.get('env_source_script'),
            data_type=data_type,
        )

    def _execute(self, exec_handle, cmd, *, head_node_only):
        log.info('%%%%%%%%%%%%%%%%')
        log.info("%s", cmd)
        log.info('%%%%%%%%%%%%%%%%')
        try:
            out_dict = exec_handle.exec(cmd, timeout=500)
            _scan_outputs(out_dict, head_node=self.head_node if head_node_only else None)
        except Exception as e:
            log.error(f'Hit Exceptions with rccl cmd {cmd} - exception {repr(e)}')
            fail_test(f'Hit Exceptions with rccl cmd {cmd} - exception {repr(e)}')

    def _run_mpi_case(
        self,
        runtime,
        test_name,
        *,
        result_file,
        data_type,
        spec: RcclMpiRunSpec,
    ):
        test_cmd = self._test_command(
            test_name,
            result_file,
            gpu_count=self.config.get('threads_per_gpu', 1),
            data_type=data_type,
        )
        cmd = _build_cluster_cmd(
            mpi_dir=self.config['mpi_dir'],
            no_of_global_ranks=self.no_of_global_ranks,
            host_file=runtime.host_file,
            debug_level=self.config.get('debug_level', 'INFO'),
            gid_index=self.config.get('gid_index', 1),
            ucx_params=runtime.ucx_params,
            path_env=runtime.path_env,
            ld_library_path=runtime.ld_library_path,
            ib_hca_list=self.config['ib_hca_list'],
            nccl_socket_ifname=self.config.get('nccl_socket_ifname', ''),
            oob_port=self.config['oob_port'],
            pml_param=runtime.pml_param,
            nccl_net_plugin=self.config.get('nccl_net_plugin'),
            min_channels=spec.min_channels,
            max_channels=spec.max_channels,
            test_cmd=test_cmd,
            use_explicit_tuning=self._uses_explicit_tuning(spec),
            nccl_algo=spec.nccl_algo or 'ring',
            qp_count=spec.qp_count or 1,
            ib_rx_queue_len=self.config.get('ib_rx_queue_len', 8192),
            hcoll_enable_mcast_all=self.config.get('hcoll_enable_mcast_all', 0),
            nccl_cumem_enable=self.config.get('nccl_cumem_enable', 0),
            nccl_ib_timeout=self.config.get('nccl_ib_timeout', 30),
            nccl_ib_sl=self.config.get('nccl_ib_sl', 0),
            nccl_ib_tc=self.config.get('nccl_ib_tc', 41),
            nccl_ib_split_data_on_qps=self.config.get('nccl_ib_split_data_on_qps', 0),
            nccl_pxn_disable=(
                self.config.get('nccl_pxn_disable', 1)
                if spec.nccl_pxn_disable is None
                else spec.nccl_pxn_disable
            ),
        )
        self._execute(self.shdl, cmd, head_node_only=True)
        return _load_result_map(self.shdl, result_file)[self.head_node]

    def _collect_gpu_info(self):
        smi_out_dict = self.shdl.exec('rocm-smi -a | head -30')
        get_model_from_rocm_smi_output(smi_out_dict[self.head_node])

    def _verify_flat(self, test_name, results):
        _verify_flat_results(
            test_name,
            results,
            self.config.get('results'),
            verify_bus_bw=self.config.get('verify_bus_bw', False),
            verify_bw_dip=self.config.get('verify_bw_dip', True),
            verify_lat_dip=self.config.get('verify_lat_dip', True),
        )
