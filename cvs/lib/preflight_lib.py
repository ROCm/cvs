'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import time
import json
import random
from pathlib import Path
from datetime import datetime

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import linux_utils
from cvs.lib import globals

log = globals.log


def check_gid_consistency(phdl, gid_index="3", expected_interfaces=None):
    """
    Verify GID index exists on specified RDMA interfaces across the cluster.

    Args:
        phdl: Parallel SSH handle for cluster nodes
        gid_index: GID index to check (default: "3")
        expected_interfaces: List of specific interfaces to check (if None, checks all)

    Returns:
        dict: Results with per-node GID status
    """
    if expected_interfaces:
        interface_filter = " ".join([f"/sys/class/infiniband/{iface}" for iface in expected_interfaces])
        log.info(f"Checking GID consistency for index {gid_index} on specific interfaces: {expected_interfaces}")
        cmd = f"""
        for dev in {interface_filter}; do 
            if [ -d "$dev" ]; then
                dev_name=$(basename "$dev")
                echo "DEVICE:$dev_name"
                if [ -f "$dev/ports/1/gids/{gid_index}" ]; then
                    gid_value=$(cat "$dev/ports/1/gids/{gid_index}" 2>/dev/null)
                    if [ -n "$gid_value" ] && [ "$gid_value" != "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
                        echo "GID_OK:$gid_value"
                    else
                        echo "GID_EMPTY:$gid_value"
                    fi
                else
                    echo "GID_MISSING:No GID file"
                fi
            else
                dev_name=$(basename "$dev")
                echo "DEVICE:$dev_name"
                echo "DEVICE_MISSING:Interface not found"
            fi
        done
        """
    else:
        log.info(f"Checking GID consistency for index {gid_index} on all interfaces")
        cmd = f"""
        for dev in /sys/class/infiniband/*/; do 
            if [ -d "$dev" ]; then
                dev_name=$(basename "$dev")
                echo "DEVICE:$dev_name"
                if [ -f "$dev/ports/1/gids/{gid_index}" ]; then
                    gid_value=$(cat "$dev/ports/1/gids/{gid_index}" 2>/dev/null)
                    if [ -n "$gid_value" ] && [ "$gid_value" != "0000:0000:0000:0000:0000:0000:0000:0000" ]; then
                        echo "GID_OK:$gid_value"
                    else
                        echo "GID_EMPTY:$gid_value"
                    fi
                else
                    echo "GID_MISSING:No GID file"
                fi
            fi
        done
        """

    results = {}
    out_dict = phdl.exec(cmd)

    for node, output in out_dict.items():
        results[node] = {'status': 'PASS', 'interfaces': {}, 'errors': []}

        current_device = None
        for line in output.strip().split('\n'):
            if line.startswith('DEVICE:'):
                current_device = line.split(':', 1)[1]
                results[node]['interfaces'][current_device] = {}
            elif line.startswith('GID_OK:'):
                gid_value = line.split(':', 1)[1]
                results[node]['interfaces'][current_device] = {'status': 'OK', 'gid_value': gid_value}
            elif line.startswith('GID_EMPTY:'):
                gid_value = line.split(':', 1)[1]
                results[node]['interfaces'][current_device] = {'status': 'EMPTY', 'gid_value': gid_value}
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"GID index {gid_index} is empty on {current_device}")
            elif line.startswith('GID_MISSING:'):
                error_msg = line.split(':', 1)[1]
                results[node]['interfaces'][current_device] = {'status': 'MISSING', 'error': error_msg}
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"GID index {gid_index} missing on {current_device}: {error_msg}")
            elif line.startswith('DEVICE_MISSING:'):
                error_msg = line.split(':', 1)[1]
                results[node]['interfaces'][current_device] = {'status': 'DEVICE_MISSING', 'error': error_msg}
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"Interface {current_device} not found: {error_msg}")

    return results


def check_rocm_versions(phdl, expected_version):
    """
    Verify ROCm version consistency across cluster nodes.

    Args:
        phdl: Parallel SSH handle for cluster nodes
        expected_version: Expected ROCm version string (e.g., "6.2.0")

    Returns:
        dict: Results with per-node ROCm version status
    """
    log.info(f"Checking ROCm version consistency (expected: {expected_version})")

    # Try multiple methods to get ROCm version
    cmd = """
    # Method 1: amd-smi (most reliable for newer ROCm)
    if command -v amd-smi >/dev/null 2>&1; then
        amd-smi version 2>/dev/null | sed -n 's/.*ROCm version: \\([0-9.]*\\).*/\\1/p'
    fi | head -1 | grep -v '^$' || 
    # Method 2: ROCm info files (fallback)
    (cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm*/share/doc/rocm/version 2>/dev/null) | head -1 | grep -v '^$' ||
    echo 'NOT_FOUND'
    """

    results = {}
    out_dict = phdl.exec(cmd)

    for node, output in out_dict.items():
        version = output.strip()
        results[node] = {
            'detected_version': version,
            'expected_version': expected_version,
            'status': 'PASS' if version == expected_version else 'FAIL',
            'errors': [],
        }

        if version == 'NOT_FOUND':
            results[node]['errors'].append("ROCm version not found - neither amd-smi nor ROCm info files available")
        elif version != expected_version:
            results[node]['errors'].append(f"Version mismatch: expected {expected_version}, found {version}")

    return results


def check_interface_names(phdl, expected_interfaces=None):
    """
    Verify RDMA interface presence and consistency across cluster nodes.

    Uses existing CVS linux_utils.get_rdma_nic_dict() for robust interface detection.

    Args:
        phdl: Parallel SSH handle for cluster nodes
        expected_interfaces: List of expected interface names

    Returns:
        dict: Results with per-node interface status
    """
    if expected_interfaces is None:
        expected_interfaces = ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]

    log.info(f"Checking RDMA interface presence (expected: {expected_interfaces})")

    # Use existing CVS utility to get RDMA interface information
    rdma_dict = linux_utils.get_rdma_nic_dict(phdl)

    results = {}

    for node in phdl.host_list:
        node_rdma_info = rdma_dict.get(node, {})
        found_interfaces = list(node_rdma_info.keys())

        results[node] = {
            'interfaces': [],
            'status': 'PASS',
            'errors': [],
            'expected_interfaces': expected_interfaces,
            'found_interfaces': found_interfaces,
            'missing_interfaces': [],
            'unexpected_interfaces': [],
            'inactive_interfaces': [],
            'down_interfaces': [],
        }

        if not found_interfaces:
            results[node]['status'] = 'FAIL'
            results[node]['errors'].append("No RDMA interfaces found")
            results[node]['missing_interfaces'] = expected_interfaces.copy()
        else:
            # Check for missing expected interfaces and validate states
            missing = []
            inactive_expected = []
            down_expected = []

            for expected_iface in expected_interfaces:
                if expected_iface not in found_interfaces:
                    missing.append(expected_iface)
                else:
                    # Check interface state for expected interfaces
                    iface_info = node_rdma_info[expected_iface]
                    device_status = iface_info.get('device_status', 'UNKNOWN')
                    link_status = iface_info.get('link_status', 'UNKNOWN')

                    if device_status != 'ACTIVE':
                        inactive_expected.append(f"{expected_iface} (state: {device_status})")
                    if link_status not in ['LINK_UP', 'LinkUp']:
                        down_expected.append(f"{expected_iface} (physical_state: {link_status})")

            results[node]['missing_interfaces'] = missing
            results[node]['inactive_interfaces'] = inactive_expected
            results[node]['down_interfaces'] = down_expected

            # Check for unexpected interfaces (informational)
            unexpected = []
            for found_iface in found_interfaces:
                if found_iface not in expected_interfaces:
                    unexpected.append(found_iface)

            results[node]['unexpected_interfaces'] = unexpected

            # Build interface details for compatibility
            for interface in found_interfaces:
                iface_info = node_rdma_info.get(interface, {})
                device_status = iface_info.get('device_status', 'UNKNOWN')
                link_status = iface_info.get('link_status', 'UNKNOWN')

                is_expected = interface in expected_interfaces
                is_functional = device_status == 'ACTIVE' and link_status in ['LINK_UP', 'LinkUp']

                results[node]['interfaces'].append(
                    {
                        'name': interface,
                        'expected': is_expected,
                        'device_status': device_status,
                        'link_status': link_status,
                        'functional': is_functional,
                    }
                )

            # Determine status - fail if missing, inactive, or down expected interfaces
            if missing:
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"Missing expected interfaces: {', '.join(missing)}")

            if inactive_expected:
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"Expected interfaces not ACTIVE: {', '.join(inactive_expected)}")

            if down_expected:
                results[node]['status'] = 'FAIL'
                results[node]['errors'].append(f"Expected interfaces not LINK_UP: {', '.join(down_expected)}")

            # Note: Unexpected interfaces don't cause failure, just logged for info
            if unexpected:
                log.info(f"Node {node} has unexpected interfaces (not an error): {', '.join(unexpected)}")

    return results


def generate_node_pairs(node_list, mode="basic"):
    """
    Generate node pairs for connectivity testing based on mode.

    Args:
        node_list: List of node names
        mode: "basic", "full_mesh", "sample", or "skip"

    Returns:
        list: List of tuples representing node pairs
    """
    if mode == "skip":
        return []
    elif mode == "basic":
        # Adjacent pairs like current IB tests
        pairs = []
        for i in range(0, len(node_list) - 1, 2):
            if i + 1 < len(node_list):
                pairs.append((node_list[i], node_list[i + 1]))
        return pairs

    elif mode == "full_mesh":
        # All possible pairs
        pairs = []
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                pairs.append((node_list[i], node_list[j]))
        return pairs

    elif mode == "sample":
        # Random 20% of all possible pairs
        all_pairs = []
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                all_pairs.append((node_list[i], node_list[j]))

        sample_size = max(1, len(all_pairs) // 5)  # 20%
        return random.sample(all_pairs, sample_size)

    else:
        raise ValueError(f"Unknown connectivity mode: {mode}")


# Removed generate_full_mesh_batches - replaced with parallel group-based algorithm


def partition_nodes_into_groups(node_list, group_size):
    """
    Partition nodes into groups based on configurable group_size.

    Args:
        node_list: List of all cluster nodes
        group_size: Configured group size from preflight config

    Returns:
        dict: {group_id: [node_list]} mapping
    """
    import math

    groups = {}
    num_groups = math.ceil(len(node_list) / group_size)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min(start_idx + group_size, len(node_list))
        groups[f"group_{i + 1}"] = node_list[start_idx:end_idx]

    return groups


def calculate_resource_requirements(group_size, num_interfaces=8):
    """
    Calculate per-node resource requirements based on group size.

    Args:
        group_size: Number of nodes per group
        num_interfaces: Number of RDMA interfaces (default 8)

    Returns:
        dict: Resource requirements per node
    """
    intra_group_fds = (group_size - 1) * (num_interfaces**2)
    inter_group_fds = group_size * (num_interfaces**2)

    return {
        'intra_group_fds_per_node': intra_group_fds,
        'inter_group_fds_per_node': inter_group_fds,
        'max_fds_per_node': inter_group_fds,
        'script_size_kb': (inter_group_fds * 85) // 1024,  # ~85 chars per command
    }


def find_host_group(host, groups):
    """Find which group a host belongs to."""
    for group_id, group_nodes in groups.items():
        if host in group_nodes:
            return group_id
    return None


def check_rdma_connectivity(
    phdl,
    node_list,
    mode="basic",
    port_range="9000-9999",
    timeout=10,
    expected_interfaces=None,
    gid_index="3",
    parallel_group_size=128,
    config_dict=None,
):
    """
    Test RDMA connectivity using ibv_rc_pingpong (direct IB verbs) based on specified mode.
    This provides accurate testing that matches RCCL behavior.

    Args:
        phdl: Parallel SSH handle for cluster nodes
        node_list: List of node names
        mode: "basic", "full_mesh", "sample", or "skip"
        port_range: Port range for ibv_rc_pingpong tests (e.g., "9000-9999")
        timeout: Timeout in seconds for each ibv_rc_pingpong test
        expected_interfaces: List of RDMA interfaces to test (if None, uses first available)
        gid_index: GID index to use for connections (default "3" matches RCCL)

    Returns:
        dict: Results with connectivity test status
    """
    log.info(f"Checking RDMA connectivity using ibv_rc_pingpong (mode: {mode}, GID index: {gid_index})")

    # Handle skip mode
    if mode == "skip":
        log.info("RDMA connectivity test skipped by configuration")
        return {
            'mode': 'skip',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'pair_results': {},
            'node_status': {},
            'skipped': True,
        }

    # Parse port range
    port_start, port_end = map(int, port_range.split('-'))

    results = {
        'mode': mode,
        'total_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'pair_results': {},
        'node_status': {},
        'gid_index': gid_index,
        'port_range': port_range,
        'timeout': timeout,
    }

    # Initialize node status
    for node in node_list:
        results['node_status'][node] = {'server_tests': 0, 'client_tests': 0, 'successful_tests': 0, 'failed_tests': 0}

    if mode == "full_mesh":
        # Use new parallel group-based algorithm
        all_results = execute_parallel_full_mesh_connectivity(
            phdl, node_list, port_start, timeout, expected_interfaces, gid_index, parallel_group_size, config_dict
        )

        # Process results from parallel algorithm
        for pair_key, pair_result in all_results.items():
            results['pair_results'][pair_key] = pair_result

            if pair_result['status'] == 'PASS':
                results['successful_pairs'] += 1
            else:
                results['failed_pairs'] += 1

            # Update node statistics
            if '(' in pair_key:
                base_pair = pair_key.split(' (')[0]
                server_node, client_node = base_pair.split(' ↔ ')
            else:
                server_node, client_node = pair_key.split(' ↔ ')

            results['node_status'][server_node]['server_tests'] += 1
            results['node_status'][client_node]['client_tests'] += 1

            if pair_result['status'] == 'PASS':
                results['node_status'][server_node]['successful_tests'] += 1
                results['node_status'][client_node]['successful_tests'] += 1
            else:
                results['node_status'][server_node]['failed_tests'] += 1
                results['node_status'][client_node]['failed_tests'] += 1

        # Total tests calculated in execute_parallel_full_mesh_connectivity
        results['total_pairs'] = len(all_results)

    else:
        # Single batch for basic or sample mode
        pairs = generate_node_pairs(node_list, mode)
        batch_results = _run_ibv_rc_pingpong_batch(phdl, pairs, port_start, timeout, expected_interfaces, gid_index)

        # Calculate total tests (pairs × interface_combinations)
        # Cross-interface testing: each node pair tests all server_iface → client_iface combinations
        num_interfaces = len(expected_interfaces) if expected_interfaces else 1
        results['total_pairs'] = len(pairs) * num_interfaces * num_interfaces
        results['pair_results'] = batch_results

        for pair_key, pair_result in batch_results.items():
            if pair_result['status'] == 'PASS':
                results['successful_pairs'] += 1
            else:
                results['failed_pairs'] += 1

            # Update node statistics
            # Handle new format: "node1 ↔ node2 (interface)" or old format: "node1 ↔ node2"
            if '(' in pair_key:
                base_pair = pair_key.split(' (')[0]
                server_node, client_node = base_pair.split(' ↔ ')
            else:
                server_node, client_node = pair_key.split(' ↔ ')
            results['node_status'][server_node]['server_tests'] += 1
            results['node_status'][client_node]['client_tests'] += 1

            if pair_result['status'] == 'PASS':
                results['node_status'][server_node]['successful_tests'] += 1
                results['node_status'][client_node]['successful_tests'] += 1
            else:
                results['node_status'][server_node]['failed_tests'] += 1
                results['node_status'][client_node]['failed_tests'] += 1

    return results


def execute_parallel_full_mesh_connectivity(
    phdl, node_list, port_start, timeout, expected_interfaces, gid_index, parallel_group_size, config_dict=None
):
    """
    Execute full mesh connectivity testing using parallel group-based algorithm with script copy approach.

    Args:
        phdl: Parallel SSH handle for cluster nodes
        node_list: List of all cluster nodes
        port_start: Starting port number
        timeout: Timeout for tests
        expected_interfaces: List of RDMA interfaces
        gid_index: GID index to use
        parallel_group_size: Group size for parallel testing

    Returns:
        dict: All connectivity test results
    """
    log.info(f"Starting parallel full mesh connectivity testing with group size {parallel_group_size}")

    # Partition nodes into groups (use only reachable nodes)
    reachable_node_list = list(phdl.reachable_hosts)
    groups = partition_nodes_into_groups(reachable_node_list, parallel_group_size)
    num_groups = len(groups)

    log.info(f"Partitioned {len(reachable_node_list)} reachable nodes into {num_groups} groups")
    for group_id, group_nodes in groups.items():
        log.info(f"{group_id}: {len(group_nodes)} nodes")

    all_results = {}

    # Round 1: Intra-group testing (all groups in parallel)
    log.info("=== Round 1: Intra-group parallel testing ===")
    round1_results = execute_round_with_script_coordination(
        phdl, groups, "intra_group", port_start, timeout, expected_interfaces, gid_index, config_dict
    )
    all_results.update(round1_results)

    # Rounds 2-N: Inter-group testing
    group_ids = list(groups.keys())
    for round_num in range(2, num_groups + 1):
        log.info(f"=== Round {round_num}: Inter-group parallel testing ===")

        # Generate inter-group pairs for this round
        inter_group_tests = {}
        for i, group1_id in enumerate(group_ids):
            group2_idx = (i + round_num - 2) % num_groups
            if group2_idx != i:  # Don't test group with itself
                group2_id = group_ids[group2_idx]
                inter_group_tests[f"{group1_id}_to_{group2_id}"] = {
                    'group1': groups[group1_id],
                    'group2': groups[group2_id],
                }

        if inter_group_tests:
            round_results = execute_round_with_script_coordination(
                phdl,
                inter_group_tests,
                "inter_group",
                port_start + round_num * 1000,
                timeout,
                expected_interfaces,
                gid_index,
                config_dict,
            )
            all_results.update(round_results)

    log.info(f"Parallel full mesh connectivity testing completed: {len(all_results)} total tests")
    return all_results


def execute_round_with_script_coordination(
    phdl, test_groups, round_type, port_start, timeout, expected_interfaces, gid_index, config_dict=None
):
    """
    Execute a round of testing using ScriptLet for optimal performance and reliability.

    Three-phase execution:
    1. Server Phase: Start all ibv_rc_pingpong servers, wait for confirmation
    2. Client Phase: Start all clients, wait for completion
    3. Result Phase: Collect and analyze results with minimal output

    Args:
        phdl: Parallel SSH handle
        test_groups: Groups or group pairs to test
        round_type: "intra_group" or "inter_group"
        port_start: Starting port number
        timeout: Test timeout
        expected_interfaces: RDMA interfaces
        gid_index: GID index

    Returns:
        dict: Round results
    """
    from .scriptlet import ScriptLet

    # Get ScriptLet configuration from config
    scriptlet_debug = config_dict.get('scriptlet_debug', False) if config_dict else False
    temp_dir = "/tmp/preflight"

    log.info(f"Starting ScriptLet-based {round_type} round execution")

    with ScriptLet(phdl, debug=scriptlet_debug, temp_dir=temp_dir, cleanup_on_init=True) as scriptlet:
        # Phase 1: Server Execution - Start servers and wait for confirmation
        log.info("Phase 1: Starting ibv_rc_pingpong servers")

        for host in phdl.reachable_hosts:
            server_commands = generate_server_commands_for_round(
                host, test_groups, round_type, port_start, expected_interfaces, gid_index, temp_dir
            )

            if server_commands:
                # Create server script that starts all servers in background
                script_content = f"""#!/bin/bash
# Start all ibv_rc_pingpong servers in background
{chr(10).join(server_commands)}

# Brief pause to let servers bind to ports
sleep 1

echo "All servers started on {host}"
exit 0
"""
                script_id = f"servers_{host}_{round_type}"
                scriptlet.create_script(script_id, script_content)

        # Copy and execute all server scripts
        server_script_mapping = {
            host: f"servers_{host}_{round_type}"
            for host in phdl.reachable_hosts
            if f"servers_{host}_{round_type}" in scriptlet.local_scripts
        }

        if server_script_mapping:
            # Copy all server scripts to their respective nodes
            scriptlet.copy_script_list(server_script_mapping)

            # Execute all server scripts in parallel using single phdl.exec_cmd_list call
            log.info(f"Starting {len(server_script_mapping)} server scripts in parallel")
            server_results = scriptlet.run_parallel_group(server_script_mapping, timeout=30)

            # Verify all servers started successfully
            failed_servers = []
            for node, output in server_results.items():
                if "All servers started" not in output:
                    failed_servers.append(f"{node}: {output}")

            if failed_servers:
                log.error(f"Server startup failed on {len(failed_servers)} nodes:")
                for failure in failed_servers:
                    log.error(f"  {failure}")
                raise RuntimeError(f"Server startup failures: {failed_servers}")

            log.info(f"All servers started successfully on {len(server_script_mapping)} nodes")

        # Phase 2: Client Execution - Start clients and wait for completion
        log.info("Phase 2: Starting ibv_rc_pingpong clients")

        for host in phdl.reachable_hosts:
            client_commands = generate_client_commands_for_round(
                host, test_groups, round_type, port_start, expected_interfaces, gid_index, temp_dir
            )

            if client_commands:
                # Create client script that runs all clients in parallel with proper synchronization
                script_content = f"""#!/bin/bash
# Run all ibv_rc_pingpong clients in parallel
{chr(10).join(client_commands)}

# Wait for all background jobs to complete
wait

echo "All clients completed on {host}"
exit 0
"""
                script_id = f"clients_{host}_{round_type}"
                scriptlet.create_script(script_id, script_content)

        # Copy and execute all client scripts
        client_script_mapping = {
            host: f"clients_{host}_{round_type}"
            for host in phdl.reachable_hosts
            if f"clients_{host}_{round_type}" in scriptlet.local_scripts
        }

        client_results = {}
        if client_script_mapping:
            # Copy all client scripts to their respective nodes
            scriptlet.copy_script_list(client_script_mapping)

            # Execute all client scripts in parallel using single phdl.exec_cmd_list call
            log.info(f"Starting {len(client_script_mapping)} client scripts in parallel")
            client_execution_results = scriptlet.run_parallel_group(client_script_mapping, timeout=timeout + 30)

            # Verify all clients completed
            failed_clients = []
            for node, output in client_execution_results.items():
                if "All clients completed" not in output:
                    failed_clients.append(f"{node}: {output}")

            if failed_clients:
                log.warning(f"Client execution issues on {len(failed_clients)} nodes:")
                for failure in failed_clients[:5]:  # Log first 5 failures
                    log.warning(f"  {failure}")

            log.info(f"Client execution completed on {len(client_script_mapping)} nodes")

        # Phase 3: Result Collection - Analyze logs and collect failures
        log.info("Phase 3: Collecting and analyzing test results")

        # Build test metadata for result collection using consistent port assignments
        assignments = _calculate_test_port_assignments(test_groups, round_type, port_start, expected_interfaces)

        test_metadata = []
        for assignment in assignments:
            server_node = assignment['server_node']
            client_node = assignment['client_node']
            server_iface = assignment['server_iface']
            client_iface = assignment['client_iface']
            port = assignment['port']

            combo_name = f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
            pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

            test_metadata.append(
                {
                    'pair_key': pair_key,
                    'server_node': server_node,
                    'client_node': client_node,
                    'server_iface': server_iface,
                    'client_iface': client_iface,
                    'combo_name': combo_name,
                    'port': port,
                    'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                    'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                }
            )

        # Use ScriptLet for efficient result collection
        client_results = collect_test_results_with_scriptlet(
            phdl, test_metadata, expected_interfaces, gid_index, config_dict
        )

    log.info(f"ScriptLet-based {round_type} round completed with {len(client_results)} results")
    return client_results


def _calculate_test_port_assignments(test_groups, round_type, port_start, expected_interfaces):
    """
    Calculate port assignments for all tests in a consistent manner.

    Uses per-node sequential port allocation to avoid port overflow while
    allowing port reuse across different nodes.

    Returns a list of test assignments with port numbers that can be used
    by both command generation and metadata creation.

    Args:
        test_groups: Test group configuration
        round_type: "intra_group" or "inter_group"
        port_start: Starting port number
        expected_interfaces: List of RDMA interfaces

    Returns:
        List of dicts with: server_node, client_node, server_iface, client_iface, port
    """
    assignments = []

    # Track port usage per server node (each node can reuse the same port range)
    node_port_counters = {}

    if round_type == "intra_group":
        for group_id, group_nodes in test_groups.items():
            for server_node in group_nodes:
                # Initialize port counter for this server node
                if server_node not in node_port_counters:
                    node_port_counters[server_node] = 0

                for client_node in group_nodes:
                    if server_node != client_node:
                        for server_iface in expected_interfaces:
                            for client_iface in expected_interfaces:
                                # Assign sequential port for this specific server node
                                port = port_start + node_port_counters[server_node]
                                assignments.append(
                                    {
                                        'server_node': server_node,
                                        'client_node': client_node,
                                        'server_iface': server_iface,
                                        'client_iface': client_iface,
                                        'port': port,
                                    }
                                )
                                # Increment port counter only for this server node
                                node_port_counters[server_node] += 1

    elif round_type == "inter_group":
        for test_name, test_config in test_groups.items():
            group1_nodes = test_config['group1']
            group2_nodes = test_config['group2']

            for server_node in group1_nodes:
                # Initialize port counter for this server node
                if server_node not in node_port_counters:
                    node_port_counters[server_node] = 0

                for client_node in group2_nodes:
                    for server_iface in expected_interfaces:
                        for client_iface in expected_interfaces:
                            # Assign sequential port for this specific server node
                            port = port_start + node_port_counters[server_node]
                            assignments.append(
                                {
                                    'server_node': server_node,
                                    'client_node': client_node,
                                    'server_iface': server_iface,
                                    'client_iface': client_iface,
                                    'port': port,
                                }
                            )
                            # Increment port counter only for this server node
                            node_port_counters[server_node] += 1

    return assignments


def generate_server_commands_for_round(
    host, test_groups, round_type, port_start, expected_interfaces, gid_index, temp_dir="/tmp/preflight"
):
    """Generate ibv_rc_pingpong server commands for a specific host and round."""
    # Get consistent port assignments
    assignments = _calculate_test_port_assignments(test_groups, round_type, port_start, expected_interfaces)

    commands = []
    for assignment in assignments:
        if assignment['server_node'] == host:
            server_iface = assignment['server_iface']
            port = assignment['port']
            cmd = f"timeout 120 ibv_rc_pingpong -d {server_iface} -g {gid_index} -p {port} > {temp_dir}/server_{server_iface}_{port}.log 2>&1 &"
            commands.append(cmd)

    return commands


def generate_client_commands_for_round(
    host, test_groups, round_type, port_start, expected_interfaces, gid_index, temp_dir="/tmp/preflight"
):
    """Generate ibv_rc_pingpong client commands for a specific host and round."""
    # Get consistent port assignments
    assignments = _calculate_test_port_assignments(test_groups, round_type, port_start, expected_interfaces)

    commands = []
    for assignment in assignments:
        if assignment['client_node'] == host:
            client_iface = assignment['client_iface']
            server_node = assignment['server_node']
            port = assignment['port']
            cmd = f"timeout 30 ibv_rc_pingpong -d {client_iface} -g {gid_index} -p {port} {server_node} > {temp_dir}/client_{client_iface}_{port}.log 2>&1 &"
            commands.append(cmd)

    return commands


def create_ibv_result_collection_script(test_metadata_for_node):
    """
    Generate an optimized shell script that analyzes ibv_rc_pingpong results
    and emits minimal key=value output for failed tests only.

    This approach dramatically reduces output size by:
    1. Only reporting failed tests (ignoring successful ones)
    2. Using compact key=value format instead of full log content
    3. Performing analysis on the remote node to minimize data transfer

    Args:
        test_metadata_for_node: List of test metadata for this specific node

    Returns:
        str: Shell script content that analyzes logs and reports failures
    """

    script_lines = [
        "#!/bin/bash",
        "# Optimized ibv_rc_pingpong result collection script",
        "# Only reports failed tests in key=value format",
        "",
        "# Function to analyze ibv_rc_pingpong output for success/failure",
        "analyze_ibv_output() {",
        "    local log_file=\"$1\"",
        "    local test_key=\"$2\"",
        "    ",
        "    if [[ ! -f \"$log_file\" ]]; then",
        "        echo \"${test_key}=LOG_MISSING\"",
        "        return",
        "    fi",
        "    ",
        "    local content=$(cat \"$log_file\" 2>/dev/null)",
        "    ",
        "    # Check for success patterns",
        "    if echo \"$content\" | grep -qE '[0-9]+ bytes in .* seconds|local address:.*GID.*remote address:.*GID'; then",
        "        # Success - don't report (minimal output)",
        "        return",
        "    fi",
        "    ",
        "    # Check for specific failure patterns",
        "    if echo \"$content\" | grep -qiE 'Failed to modify QP.*to RTR'; then",
        "        echo \"${test_key}=QP_RTR_FAILED\"",
        "    elif echo \"$content\" | grep -qiE 'Failed to modify QP.*to RTS'; then",
        "        echo \"${test_key}=QP_RTS_FAILED\"",
        "    elif echo \"$content\" | grep -qiE \"Couldn.*t connect to|Unable to Connect\"; then",
        "        echo \"${test_key}=CONNECTION_FAILED\"",
        "    elif echo \"$content\" | grep -qiE 'Failed status transport retry counter exceeded'; then",
        "        echo \"${test_key}=TRANSPORT_RETRY_EXCEEDED\"",
        "    elif echo \"$content\" | grep -qiE 'parse WC failed'; then",
        "        echo \"${test_key}=WC_PARSE_FAILED\"",
        "    elif echo \"$content\" | grep -qiE 'No space left on device'; then",
        "        echo \"${test_key}=NO_SPACE_LEFT\"",
        "    elif [[ -z \"$content\" ]]; then",
        "        echo \"${test_key}=EMPTY_LOG\"",
        "    else",
        "        echo \"${test_key}=UNKNOWN_FAILURE\"",
        "    fi",
        "}",
        "",
        "# Analyze test results",
    ]

    # Add analysis calls for each test involving this node
    for test in test_metadata_for_node:
        # Create a compact test identifier
        test_key = (
            f"{test['server_node']}-{test['client_node']}-{test['server_iface']}-{test['client_iface']}-{test['port']}"
        )

        # Check if this node has client logs to analyze
        if 'client_log_path' in test and test.get('node_role') == 'client':
            client_log = test['client_log_path']
            script_lines.append(f"analyze_ibv_output \"{client_log}\" \"CLIENT_{test_key}\"")

        # Check if this node has server logs to analyze
        if 'server_log_path' in test and test.get('node_role') == 'server':
            server_log = test['server_log_path']
            script_lines.append(f"analyze_ibv_output \"{server_log}\" \"SERVER_{test_key}\"")

    script_lines.extend(
        [
            "",
            "# Cleanup log files to free space",
            "rm -f /tmp/client_*.log /tmp/server_*.log 2>/dev/null || true",
            "",
            "exit 0",
        ]
    )

    return "\n".join(script_lines)


def collect_test_results_with_scriptlet(phdl, test_metadata, expected_interfaces, gid_index, config_dict=None):
    """
    Efficient result collection using ScriptLet with minimal output approach.

    This function replaces the previous exponential SSH approach with:
    1. One script per node that analyzes all logs locally
    2. Minimal key=value output for failed tests only
    3. Parallel execution across all nodes
    4. Dramatic reduction in network traffic and execution time

    Args:
        phdl: Parallel SSH handle
        test_metadata: List of test metadata dictionaries
        expected_interfaces: List of RDMA interfaces
        gid_index: GID index used in tests

    Returns:
        Dict: {pair_key: result_dict} - same format as original function
    """
    from .scriptlet import ScriptLet

    log.info(f"Starting ScriptLet-based result collection for {len(test_metadata)} tests")

    # Phase 1: Group test metadata by node (both client and server roles)
    node_tests = {}

    for test in test_metadata:
        client_node = test['client_node']
        server_node = test['server_node']

        # Add test to client node's list
        if client_node not in node_tests:
            node_tests[client_node] = []
        # Mark this test as involving this node as client
        test_copy = test.copy()
        test_copy['node_role'] = 'client'
        node_tests[client_node].append(test_copy)

        # Add test to server node's list (if different from client)
        if server_node != client_node:
            if server_node not in node_tests:
                node_tests[server_node] = []
            # Mark this test as involving this node as server
            test_copy = test.copy()
            test_copy['node_role'] = 'server'
            node_tests[server_node].append(test_copy)

    log.info(f"Distributing result collection across {len(node_tests)} nodes")

    # Get ScriptLet configuration from config
    scriptlet_debug = config_dict.get('scriptlet_debug', False) if config_dict else False
    temp_dir = "/tmp/preflight"

    # Phase 2: Create ScriptLet and generate optimized collection scripts
    with ScriptLet(phdl, debug=scriptlet_debug, temp_dir=temp_dir) as scriptlet:
        node_script_mapping = {}

        for node, node_test_list in node_tests.items():
            script_content = create_ibv_result_collection_script(node_test_list)
            script_id = f"collect_results_{node}"

            scriptlet.create_script(script_id, script_content)
            node_script_mapping[node] = script_id

        # Phase 3: Copy and execute collection scripts in parallel
        log.info("Copying result collection scripts to nodes")
        copy_results = scriptlet.copy_script_list(node_script_mapping)

        # Check for copy failures
        failed_copies = [result for result in copy_results.values() if "FAILED" in result]
        if failed_copies:
            log.warning(f"Failed to copy scripts to {len(failed_copies)} nodes")
            for failure in failed_copies[:3]:  # Log first 3 failures
                log.warning(f"Copy failure: {failure}")

        log.info("Executing result collection scripts in parallel")

        # Filter out nodes where script copy failed
        successful_nodes = [node for node, status in copy_results.items() if "SUCCESS" in status]
        filtered_script_mapping = {
            node: script_id for node, script_id in node_script_mapping.items() if node in successful_nodes
        }

        failed_copy_nodes = set(node_script_mapping.keys()) - set(successful_nodes)
        if failed_copy_nodes:
            log.warning(
                f"Skipping execution on {len(failed_copy_nodes)} nodes due to copy failures: {list(failed_copy_nodes)[:3]}..."
            )

        # Execute all collection scripts in parallel with longer timeout for analysis
        log.info(f"Starting {len(filtered_script_mapping)} result collection scripts in parallel")
        execution_results = scriptlet.run_parallel_group(filtered_script_mapping, timeout=120)

    # Phase 4: Parse minimal results and build full result structure
    log.info("Processing collected failure reports")

    # Parse the key=value results from all nodes
    failure_reports = {}
    for node, output in execution_results.items():
        if not output or output == 'NO_OUTPUT':
            continue

        # Parse key=value lines
        for line in output.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                failure_reports[key] = value

    log.info(f"Collected {len(failure_reports)} failure reports from nodes")

    # Phase 5: Build final results with success/failure status
    final_results = {}

    for test in test_metadata:
        pair_key = test['pair_key']
        server_node = test['server_node']
        client_node = test['client_node']

        # Create test key to match script output
        test_key = f"{server_node}-{client_node}-{test['server_iface']}-{test['client_iface']}-{test['port']}"

        # Check for failures in either client or server
        client_key = f"CLIENT_{test_key}"
        server_key = f"SERVER_{test_key}"

        client_failure = failure_reports.get(client_key)
        server_failure = failure_reports.get(server_key)

        # Determine overall status
        if client_failure or server_failure:
            status = 'FAIL'
            error_details = []

            if client_failure:
                error_details.append(f"Client: {client_failure}")
            if server_failure:
                error_details.append(f"Server: {server_failure}")

            # Provide minimal but useful output for failed tests
            client_output = f"FAILED: {client_failure}" if client_failure else "PASSED"
            server_output = f"FAILED: {server_failure}" if server_failure else "PASSED"
        else:
            # No failure reported = success
            status = 'PASS'
            error_details = []
            client_output = "PASSED"
            server_output = "PASSED"

        # Build final result entry (maintaining compatibility with existing code)
        final_results[pair_key] = {
            'status': status,
            'server_node': server_node,
            'client_node': client_node,
            'interface': test['combo_name'],
            'port': test['port'],
            'client_output': client_output,
            'server_output': server_output,
            'error_details': error_details,
            'server_cmd': test.get(
                'server_cmd', f"ibv_rc_pingpong -d {test['server_iface']} -g {gid_index} -p {test['port']}"
            ),
            'client_cmd': test.get(
                'client_cmd',
                f"ibv_rc_pingpong -d {test['client_iface']} -g {gid_index} -p {test['port']} {server_node}",
            ),
            'gid_index': gid_index,
        }

    log.info(f"Result collection completed: {len(final_results)} tests processed")

    # Log summary statistics
    passed_tests = len([r for r in final_results.values() if r['status'] == 'PASS'])
    failed_tests = len([r for r in final_results.values() if r['status'] == 'FAIL'])
    log.info(f"Test summary: {passed_tests} passed, {failed_tests} failed")

    return final_results


def collect_test_results(phdl, test_groups, round_type, port_start, expected_interfaces, gid_index):
    """
    Collect and analyze test results using ScriptLet for optimal performance.

    Eliminates exponential SSH calls by using ScriptLet's minimal output approach:
    - One script per node analyzes all logs locally
    - Only failed tests reported in key=value format
    - Dramatic reduction in network traffic and execution time
    """
    # Build test metadata from groups
    test_metadata = []
    port_offset = 0

    if round_type == "intra_group":
        for group_id, group_nodes in test_groups.items():
            for server_node in group_nodes:
                for client_node in group_nodes:
                    if server_node != client_node:
                        for server_iface in expected_interfaces:
                            for client_iface in expected_interfaces:
                                port = port_start + port_offset
                                combo_name = (
                                    f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                                )
                                pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

                                test_metadata.append(
                                    {
                                        'pair_key': pair_key,
                                        'server_node': server_node,
                                        'client_node': client_node,
                                        'server_iface': server_iface,
                                        'client_iface': client_iface,
                                        'combo_name': combo_name,
                                        'port': port,
                                        'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                                        'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                                    }
                                )
                                port_offset += 1

    elif round_type == "inter_group":
        for test_name, test_config in test_groups.items():
            group1_nodes = test_config['group1']
            group2_nodes = test_config['group2']

            for server_node in group1_nodes:
                for client_node in group2_nodes:
                    for server_iface in expected_interfaces:
                        for client_iface in expected_interfaces:
                            port = port_start + port_offset
                            combo_name = (
                                f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                            )
                            pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

                            test_metadata.append(
                                {
                                    'pair_key': pair_key,
                                    'server_node': server_node,
                                    'client_node': client_node,
                                    'server_iface': server_iface,
                                    'client_iface': client_iface,
                                    'combo_name': combo_name,
                                    'port': port,
                                    'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                                    'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                                }
                            )
                            port_offset += 1

    # Use ScriptLet for efficient result collection
    return collect_test_results_with_scriptlet(phdl, test_metadata, expected_interfaces, gid_index)


def _run_ibv_rc_pingpong_batch(phdl, pairs, base_port, timeout, expected_interfaces=None, gid_index="3"):
    """
    Run cross-interface ibv_rc_pingpong tests in parallel batches.

    Tests all server_interface → client_interface combinations to catch network routing issues.
    This is critical because RCCL uses cross-interface communication patterns.

    Args:
        phdl: Parallel SSH handle
        pairs: List of (server_node, client_node) tuples
        base_port: Base port number for this batch
        timeout: Timeout in seconds
        expected_interfaces: List of RDMA interfaces to test
        gid_index: GID index to use for tests

    Returns:
        dict: Results for all cross-interface combinations
    """
    if not pairs or not expected_interfaces:
        return {}

    log.info(
        f"Running cross-interface connectivity tests: {len(expected_interfaces)}×{len(expected_interfaces)} = {len(expected_interfaces) ** 2} interface combinations"
    )

    all_results = {}
    command_store = {}

    # Generate all cross-interface test combinations
    test_combinations = []
    for server_iface in expected_interfaces:
        for client_iface in expected_interfaces:
            for pair_idx, (server_node, client_node) in enumerate(pairs):
                combo_name = f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                port = (
                    base_port
                    + (len(expected_interfaces) * expected_interfaces.index(server_iface))
                    + expected_interfaces.index(client_iface) * 10
                    + pair_idx
                )

                test_combinations.append(
                    {
                        'server_node': server_node,
                        'client_node': client_node,
                        'server_interface': server_iface,
                        'client_interface': client_iface,
                        'combo_name': combo_name,
                        'port': port,
                        'pair_key': f"{server_node} ↔ {client_node} ({combo_name})",
                    }
                )

    # Execute tests in parallel batches to avoid resource exhaustion
    # Dynamic batch sizing based on cluster size and interface count
    num_nodes = len(
        set([test['server_node'] for test in test_combinations] + [test['client_node'] for test in test_combinations])
    )
    num_interfaces = len(expected_interfaces)

    # Calculate optimal batch size:
    # - Small clusters (≤4 nodes): Run all tests in parallel
    # - Medium clusters (5-8 nodes): Batch by interface count
    # - Large clusters (>8 nodes): Conservative batching
    total_tests = len(test_combinations)

    if num_nodes <= 4:
        batch_size = total_tests  # Run ALL tests in parallel
        log.info(f"Small cluster ({num_nodes} nodes): Running all {total_tests} tests in parallel")
    elif num_nodes <= 8:
        batch_size = min(num_interfaces * num_interfaces, 64)  # One full interface matrix
        log.info(f"Medium cluster ({num_nodes} nodes): Batching {batch_size} tests per batch")
    else:
        batch_size = 32  # Conservative for large clusters
        log.info(f"Large cluster ({num_nodes} nodes): Conservative batching of {batch_size} tests per batch")

    for batch_start in range(0, total_tests, batch_size):
        batch_end = min(batch_start + batch_size, total_tests)
        batch = test_combinations[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_tests + batch_size - 1) // batch_size

        log.info(f"Executing batch {batch_num}/{total_batches} with {len(batch)} tests")

        # Build command lists for this batch
        host_server_cmds = {}
        host_client_cmds = {}

        for test in batch:
            server_node = test['server_node']
            client_node = test['client_node']
            server_iface = test['server_interface']
            client_iface = test['client_interface']
            port = test['port']
            pair_key = test['pair_key']

            # Generate commands
            server_cmd_display = f"ibv_rc_pingpong -d {server_iface} -g {gid_index} -p {port}"
            client_cmd_display = f"ibv_rc_pingpong -d {client_iface} -g {gid_index} -p {port} {server_node}"
            server_cmd_exec = f"timeout 120 ibv_rc_pingpong -d {server_iface} -g {gid_index} -p {port} > {temp_dir}/ibv_rc_server_{server_iface}_{port}.log 2>&1 &"
            client_cmd_exec = f"timeout 30 ibv_rc_pingpong -d {client_iface} -g {gid_index} -p {port} {server_node} > {temp_dir}/ibv_rc_client_{client_iface}_{port}.log 2>&1 &"

            # Store commands for HTML report
            command_store[pair_key] = {
                'server_cmd': server_cmd_display,
                'client_cmd': client_cmd_display,
                'gid_index': gid_index,
                'port': port,
            }

            # Group commands by host
            if server_node not in host_server_cmds:
                host_server_cmds[server_node] = []
            host_server_cmds[server_node].append(server_cmd_exec)

            if client_node not in host_client_cmds:
                host_client_cmds[client_node] = []
            host_client_cmds[client_node].append(client_cmd_exec)

        # Execute servers first
        server_cmd_list = []
        for host in phdl.host_list:
            if host in host_server_cmds:
                # Use space separator for background processes (commands end with &)
                combined_cmd = " ".join(host_server_cmds[host])
                server_cmd_list.append(combined_cmd)
            else:
                server_cmd_list.append("echo 'No server role'")

        phdl.exec_cmd_list(server_cmd_list)
        time.sleep(3)  # Wait for servers to start

        # Execute clients
        client_cmd_list = []
        for host in phdl.host_list:
            if host in host_client_cmds:
                combined_cmd = "; ".join(host_client_cmds[host])
                client_cmd_list.append(combined_cmd)
            else:
                client_cmd_list.append("echo 'No client role'")

        phdl.exec_cmd_list(client_cmd_list)
        time.sleep(timeout + 2)  # Wait for clients to complete

        # Collect results for this batch
        for test in batch:
            server_node = test['server_node']
            client_node = test['client_node']
            server_iface = test['server_interface']
            client_iface = test['client_interface']
            port = test['port']
            pair_key = test['pair_key']
            combo_name = test['combo_name']

            # Read log files
            client_log_cmd = (
                f"cat {temp_dir}/ibv_rc_client_{client_iface}_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"
            )
            server_log_cmd = (
                f"cat {temp_dir}/ibv_rc_server_{server_iface}_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"
            )

            client_results = phdl.exec(client_log_cmd)
            client_output = client_results.get(client_node, 'LOG_NOT_FOUND')

            server_results = phdl.exec(server_log_cmd)
            server_output = server_results.get(server_node, 'LOG_NOT_FOUND')

            # Analyze results
            success = _analyze_ibv_rc_pingpong_output(client_output, server_output)

            # Store results with stored commands
            stored_commands = command_store.get(pair_key, {})
            all_results[pair_key] = {
                'status': 'PASS' if success else 'FAIL',
                'server_node': server_node,
                'client_node': client_node,
                'interface': combo_name,
                'port': port,
                'client_output': client_output,
                'server_output': server_output,
                'error_details': [] if success else _extract_ibv_rc_pingpong_errors(client_output, server_output),
                'server_cmd': stored_commands.get('server_cmd', 'N/A'),
                'client_cmd': stored_commands.get('client_cmd', 'N/A'),
                'gid_index': stored_commands.get('gid_index', gid_index),
            }

            if success:
                log.info(f"✅ {pair_key}: Connection successful")
            else:
                log.warning(f"❌ {pair_key}: Connection failed")

        # Clean up log files for this batch
        cleanup_ports = [test['port'] for test in batch]
        cleanup_cmd = f"rm -f {temp_dir}/ibv_rc_*_{{{','.join(map(str, cleanup_ports))}}}.log 2>/dev/null || true"
        phdl.exec(cleanup_cmd)

        # Brief pause between batches
        if batch_end < total_tests:
            time.sleep(2)

    log.info(f"Completed {total_tests} cross-interface connectivity tests")
    return all_results


def _analyze_ibv_rc_pingpong_output(client_output, server_output):
    """
    Analyze ibv_rc_pingpong output to determine if the test was successful.

    Args:
        client_output: Output from ibv_rc_pingpong client
        server_output: Output from ibv_rc_pingpong server

    Returns:
        bool: True if test was successful
    """
    # Check for success indicators
    success_patterns = [
        r"\d+ bytes in .* seconds",  # Performance results indicate success
        r"local address:.*GID.*remote address:.*GID",  # Connection established
    ]

    # Check for error patterns
    error_patterns = [
        r"Failed to modify QP.*to RTR",
        r"Failed to modify QP.*to RTS",
        r"Couldn.*t connect to",
        r"Unable to Connect",
        r"Failed status transport retry counter exceeded",
        r"parse WC failed",
    ]

    # Check client output
    client_success = any(re.search(pattern, client_output, re.IGNORECASE) for pattern in success_patterns)
    client_error = any(re.search(pattern, client_output, re.IGNORECASE) for pattern in error_patterns)

    # Check server output
    server_success = any(re.search(pattern, server_output, re.IGNORECASE) for pattern in success_patterns)
    server_error = any(re.search(pattern, server_output, re.IGNORECASE) for pattern in error_patterns)

    # Success if either client or server shows success and no critical errors
    if (client_success or server_success) and not (client_error or server_error):
        return True

    # Also check for specific success case where we have GID information even without performance data
    gid_pattern = r"local address:.*GID.*remote address:.*GID"
    if (re.search(gid_pattern, client_output) or re.search(gid_pattern, server_output)) and not (
        client_error or server_error
    ):
        return True

    return False


def _extract_ibv_rc_pingpong_errors(client_output, server_output):
    """
    Extract detailed error messages from ibv_rc_pingpong output including GID and address information.

    Args:
        client_output: Output from ibv_rc_pingpong client
        server_output: Output from ibv_rc_pingpong server

    Returns:
        list: List of detailed error messages with GID/address information
    """
    errors = []

    # Extract GID and address information
    def extract_connection_details(output, role):
        details = {}

        # Extract local address and GID (multiple patterns)
        local_patterns = [
            r'local address:\s+LID\s+0x\w+\s+QPN\s+0x\w+\s+PSN\s+0x\w+.*?GID\s+([^\s\n]+)',
            r'local address:.*?GID\s+([^\s\n,]+)',
            r'GID:\s+([^\s\n,]+)',
        ]
        for pattern in local_patterns:
            local_match = re.search(pattern, output)
            if local_match:
                details['local_gid'] = local_match.group(1)
                break

        # Extract remote address and GID (multiple patterns)
        remote_patterns = [
            r'remote address:\s+LID\s+0x\w+\s+QPN\s+0x\w+\s+PSN\s+0x\w+.*?GID\s+([^\s\n]+)',
            r'remote address:.*?GID\s+([^\s\n,]+)',
        ]
        for pattern in remote_patterns:
            remote_match = re.search(pattern, output)
            if remote_match:
                details['remote_gid'] = remote_match.group(1)
                break

        # Extract device information
        device_match = re.search(r'Device\s*:\s*(\w+)', output)
        if device_match:
            details['device'] = device_match.group(1)

        # Extract GID index
        gid_index_match = re.search(r'GID index\s*:\s*(\d+)', output)
        if gid_index_match:
            details['gid_index'] = gid_index_match.group(1)

        return details

    # Get connection details from both client and server
    client_details = extract_connection_details(client_output, 'client')
    server_details = extract_connection_details(server_output, 'server')

    # Combine details for comprehensive view
    connection_info = {}
    if client_details.get('local_gid') and client_details.get('remote_gid'):
        connection_info['client_local_gid'] = client_details['local_gid']
        connection_info['client_remote_gid'] = client_details['remote_gid']
    if server_details.get('local_gid') and server_details.get('remote_gid'):
        connection_info['server_local_gid'] = server_details['local_gid']
        connection_info['server_remote_gid'] = server_details['remote_gid']
    if client_details.get('device'):
        connection_info['device'] = client_details['device']
    if client_details.get('gid_index'):
        connection_info['gid_index'] = client_details['gid_index']

    # Extract error patterns with enhanced details
    error_patterns = [
        r"Failed to modify QP.*to RTR",
        r"Failed to modify QP.*to RTS",
        r"Couldn.*t connect to.*",
        r"Unable to Connect.*",
        r"Failed status transport retry counter exceeded.*",
        r"parse WC failed.*",
    ]

    # Extract errors from client output
    for pattern in error_patterns:
        matches = re.findall(pattern, client_output, re.IGNORECASE)
        for match in matches:
            error_msg = f"Client: {match}"
            if connection_info:
                details_parts = []
                if connection_info.get('client_local_gid'):
                    details_parts.append(f"Local GID: {connection_info['client_local_gid']}")
                if connection_info.get('client_remote_gid'):
                    details_parts.append(f"Remote GID: {connection_info['client_remote_gid']}")
                if connection_info.get('device'):
                    details_parts.append(f"Device: {connection_info['device']}")
                if connection_info.get('gid_index'):
                    details_parts.append(f"GID Index: {connection_info['gid_index']}")

                if details_parts:
                    error_msg += f" | {' | '.join(details_parts)}"
            errors.append(error_msg)

    # Extract errors from server output
    for pattern in error_patterns:
        matches = re.findall(pattern, server_output, re.IGNORECASE)
        for match in matches:
            error_msg = f"Server: {match}"
            if connection_info:
                details_parts = []
                if connection_info.get('server_local_gid'):
                    details_parts.append(f"Local GID: {connection_info['server_local_gid']}")
                if connection_info.get('server_remote_gid'):
                    details_parts.append(f"Remote GID: {connection_info['server_remote_gid']}")
                if connection_info.get('device'):
                    details_parts.append(f"Device: {connection_info['device']}")
                if connection_info.get('gid_index'):
                    details_parts.append(f"GID Index: {connection_info['gid_index']}")

                if details_parts:
                    error_msg += f" | {' | '.join(details_parts)}"
            errors.append(error_msg)

    # If no specific errors found but we have connection details, show them
    if not errors and connection_info:
        error_msg = "Connection failed - no specific error detected"
        details_parts = []
        for key, value in connection_info.items():
            if key.startswith('client_'):
                details_parts.append(f"{key.replace('client_', '').replace('_', ' ').title()}: {value}")
        if details_parts:
            error_msg += f" | {' | '.join(details_parts)}"
        errors.append(error_msg)
    elif not errors:
        errors.append("Unknown error - ibv_rc_pingpong failed to establish connection")

    return errors


def generate_preflight_summary(
    gid_results,
    connectivity_results,
    rocm_results,
    interface_results,
    reachability_results=None,
    ssh_connectivity_results=None,
):
    """
    Generate a comprehensive summary of all preflight check results.

    Args:
        gid_results: Results from GID consistency check
        connectivity_results: Results from RDMA connectivity check
        rocm_results: Results from ROCm version check
        interface_results: Results from interface name check
        reachability_results: Results from SSH reachability check
        ssh_connectivity_results: Results from SSH full mesh connectivity check

    Returns:
        dict: Comprehensive summary of all checks
    """
    summary = {
        'overall_status': 'PASS',
        'checks': {
            'ssh_reachability': _summarize_reachability_results(reachability_results),
            'gid_consistency': _summarize_gid_results(gid_results),
            'rdma_connectivity': _summarize_connectivity_results(connectivity_results),
            'rocm_versions': _summarize_rocm_results(rocm_results),
            'interface_names': _summarize_interface_results(interface_results),
        },
        'recommendations': [],
    }

    # Add SSH connectivity results if available
    if ssh_connectivity_results:
        summary['checks']['ssh_connectivity'] = _summarize_ssh_connectivity_results(ssh_connectivity_results)

    # Determine overall status (skipped tests don't affect overall status)
    for check_name, check_summary in summary['checks'].items():
        if check_summary['status'] == 'FAIL':
            summary['overall_status'] = 'FAIL'

    # Generate recommendations
    if summary['checks']['gid_consistency']['status'] == 'FAIL':
        summary['recommendations'].append("Fix GID configuration on RDMA interfaces before running performance tests")

    if summary['checks']['rdma_connectivity']['status'] == 'FAIL':
        summary['recommendations'].append("Address RDMA connectivity issues between node pairs")
    elif summary['checks']['rdma_connectivity']['status'] == 'SKIPPED':
        summary['recommendations'].append("Consider running RDMA connectivity tests for comprehensive validation")

    if summary['checks']['rocm_versions']['status'] == 'FAIL':
        summary['recommendations'].append("Ensure consistent ROCm versions across all cluster nodes")

    if summary['checks']['interface_names']['status'] == 'FAIL':
        summary['recommendations'].append("Standardize RDMA interface naming across cluster nodes")

    if summary['overall_status'] == 'PASS':
        summary['recommendations'].append("All preflight checks passed - cluster is ready for performance testing")

    return summary


def _summarize_gid_results(gid_results):
    """Summarize GID consistency check results."""
    total_interfaces = 0
    ok_interfaces = 0
    failed_nodes = []

    for node, result in gid_results.items():
        for interface, interface_result in result['interfaces'].items():
            total_interfaces += 1
            if interface_result.get('status') == 'OK':
                ok_interfaces += 1

        if result['status'] == 'FAIL':
            failed_nodes.append(node)

    return {
        'status': 'PASS' if not failed_nodes else 'FAIL',
        'total_interfaces': total_interfaces,
        'ok_interfaces': ok_interfaces,
        'failed_nodes': failed_nodes,
        'summary': f"{ok_interfaces}/{total_interfaces} interfaces have valid GID",
    }


def _summarize_connectivity_results(connectivity_results):
    """Summarize RDMA connectivity check results."""
    # Handle skipped tests (either by configuration or due to failures)
    if connectivity_results.get('skipped', False) or connectivity_results.get('status') == 'SKIPPED':
        return {
            'status': 'SKIPPED',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'mode': connectivity_results.get('mode', 'unknown'),
            'summary': connectivity_results.get('message', 'RDMA connectivity test skipped'),
        }

    # Handle normal test results
    if 'failed_pairs' not in connectivity_results:
        # Fallback for malformed results
        return {
            'status': 'ERROR',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'mode': connectivity_results.get('mode', 'unknown'),
            'summary': 'RDMA connectivity test failed to complete properly',
        }

    return {
        'status': 'PASS' if connectivity_results['failed_pairs'] == 0 else 'FAIL',
        'total_pairs': connectivity_results['total_pairs'],
        'successful_pairs': connectivity_results['successful_pairs'],
        'failed_pairs': connectivity_results['failed_pairs'],
        'mode': connectivity_results['mode'],
        'summary': f"{connectivity_results['successful_pairs']}/{connectivity_results['total_pairs']} pairs connected successfully",
    }


def _summarize_rocm_results(rocm_results):
    """Summarize ROCm version check results."""
    total_nodes = len(rocm_results)
    consistent_nodes = sum(1 for result in rocm_results.values() if result['status'] == 'PASS')
    failed_nodes = [node for node, result in rocm_results.items() if result['status'] == 'FAIL']

    return {
        'status': 'PASS' if consistent_nodes == total_nodes else 'FAIL',
        'total_nodes': total_nodes,
        'consistent_nodes': consistent_nodes,
        'failed_nodes': failed_nodes,
        'summary': f"{consistent_nodes}/{total_nodes} nodes have consistent ROCm version",
    }


def _summarize_interface_results(interface_results):
    """Summarize interface name check results."""
    total_nodes = len(interface_results)
    compliant_nodes = sum(1 for result in interface_results.values() if result['status'] == 'PASS')
    failed_nodes = [node for node, result in interface_results.items() if result['status'] == 'FAIL']

    return {
        'status': 'PASS' if compliant_nodes == total_nodes else 'FAIL',
        'total_nodes': total_nodes,
        'compliant_nodes': compliant_nodes,
        'failed_nodes': failed_nodes,
        'summary': f"{compliant_nodes}/{total_nodes} nodes have compliant interface names",
    }


def _summarize_reachability_results(reachability_results):
    """Summarize SSH reachability check results."""
    if not reachability_results:
        return {
            'status': 'UNKNOWN',
            'total_nodes': 0,
            'reachable_nodes': 0,
            'unreachable_nodes': [],
            'summary': 'SSH reachability test not performed',
        }

    total_nodes = reachability_results.get('total_nodes', 0)
    reachable_nodes = reachability_results.get('reachable_nodes', 0)
    unreachable_nodes = reachability_results.get('unreachable_nodes', [])

    status = 'PASS' if len(unreachable_nodes) == 0 else 'WARNING'

    return {
        'status': status,
        'total_nodes': total_nodes,
        'reachable_nodes': reachable_nodes,
        'unreachable_nodes': unreachable_nodes,
        'summary': f"{reachable_nodes}/{total_nodes} nodes reachable",
    }


def _summarize_ssh_connectivity_results(ssh_connectivity_results):
    """Summarize SSH full mesh connectivity check results."""
    if not ssh_connectivity_results:
        return {
            'status': 'UNKNOWN',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'summary': 'SSH connectivity test not performed',
        }

    if ssh_connectivity_results.get('skipped', False):
        return {
            'status': 'SKIPPED',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'summary': 'SSH connectivity test skipped',
        }

    total_pairs = ssh_connectivity_results.get('total_pairs', 0)
    successful_pairs = ssh_connectivity_results.get('successful_pairs', 0)
    failed_pairs = ssh_connectivity_results.get('failed_pairs', 0)

    # Determine status based on results
    if 'error' in ssh_connectivity_results:
        status = 'ERROR'
        summary = f"SSH connectivity test failed: {ssh_connectivity_results['error']}"
    elif failed_pairs == 0:
        status = 'PASS'
        summary = f"{successful_pairs}/{total_pairs} pairs connected successfully"
    else:
        status = 'FAIL'
        success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0
        summary = f"{successful_pairs}/{total_pairs} pairs connected successfully ({success_rate:.1f}%)"

    return {
        'status': status,
        'total_pairs': total_pairs,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'summary': summary,
    }


def generate_html_report(results, config_dict, output_path=None):
    """
    Generate comprehensive HTML report for preflight check results.

    Args:
        results: Complete preflight results dictionary
        config_dict: Configuration used for the tests
        output_path: Optional custom output path for the report

    Returns:
        str: Path to generated HTML report
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config_dict.get('report_output_dir', '/tmp/preflight_reports')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"preflight_report_{timestamp}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate HTML content
    html_content = _generate_html_content(results, config_dict)

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

    log.info(f"HTML report generated: {output_path}")
    return str(output_path)


def _generate_html_content(results, config_dict):
    """Generate the complete HTML content for the preflight report."""

    summary = results.get('summary', {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Cluster Preflight Check Report</title>
    <style>
        {_get_html_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GPU Cluster Preflight Check Report</h1>
            <div class="report-meta">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Overall Status:</strong> <span class="status-{summary.get('overall_status', 'UNKNOWN').lower()}">{summary.get('overall_status', 'UNKNOWN')}</span></p>
            </div>
        </header>

        {_generate_executive_summary_html(summary)}
        {_generate_gid_consistency_html(results.get('gid_consistency', {}))}
        {_generate_connectivity_html(results.get('rdma_connectivity', {}))}
        {_generate_ssh_connectivity_html(results.get('ssh_connectivity', {}))}
        {_generate_rocm_versions_html(results.get('rocm_versions', {}))}
        {_generate_interface_names_html(results.get('interface_names', {}))}
        {_generate_configuration_html(config_dict)}
        {_generate_recommendations_html(summary.get('recommendations', []))}
    </div>
</body>
</html>
"""
    return html


def _get_html_styles():
    """Return CSS styles for the HTML report."""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        header {
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 {
            color: #333;
            margin: 0;
        }
        h2 {
            color: #007acc;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        h3 {
            color: #555;
        }
        .report-meta {
            margin-top: 10px;
            color: #666;
        }
        .status-pass {
            color: #28a745;
            font-weight: bold;
        }
        .status-fail {
            color: #dc3545;
            font-weight: bold;
        }
        .status-skipped {
            color: #6c757d;
            font-weight: bold;
        }
        .error-summary {
            color: #dc3545;
            font-weight: bold;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            padding: 15px 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        .summary-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }
        .summary-table tr:hover {
            background-color: #f8f9fa;
        }
        .check-name {
            font-weight: 600;
            color: #495057;
        }
        .status-cell {
            text-align: center;
            width: 120px;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            white-space: nowrap;
        }
        .status-badge.status-pass {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-badge.status-fail {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status-badge.status-skipped {
            background-color: #e2e3e5;
            color: #6c757d;
            border: 1px solid #d6d8db;
        }
        .results-cell {
            text-align: center;
            font-weight: 600;
            color: #495057;
            width: 100px;
        }
        .details-cell {
            color: #6c757d;
            font-size: 0.95em;
        }
        .summary-row-pass {
            border-left: 4px solid #28a745;
        }
        .summary-row-fail {
            border-left: 4px solid #dc3545;
        }
        .summary-row-skipped {
            border-left: 4px solid #6c757d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .gid-cell {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            word-break: break-all;
            max-width: 200px;
        }
        .gid-cell small {
            color: #666;
            font-weight: normal;
        }
        code {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            padding: 2px 6px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.85em;
            color: #495057;
            word-break: break-all;
            display: block;
            margin: 2px 0;
        }
        .connectivity-matrix {
            display: grid;
            gap: 2px;
            margin: 20px 0;
        }
        .matrix-cell {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
            font-size: 12px;
        }
        .matrix-cell.header {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .matrix-cell.pass {
            background-color: #d4edda;
            color: #155724;
        }
        .matrix-cell.fail {
            background-color: #f8d7da;
            color: #721c24;
        }
        .matrix-cell.not-tested {
            background-color: #e2e3e5;
            color: #6c757d;
        }
        .error-list {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }
        .error-list ul {
            margin: 0;
            padding-left: 20px;
        }
        .error-list li {
            color: #721c24;
        }
        .recommendations {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 20px;
            margin: 20px 0;
        }
        .recommendations h3 {
            color: #0c5460;
            margin-top: 0;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .config-section {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }
        .config-section pre {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
    """


def _generate_executive_summary_html(summary):
    """Generate summary section with table layout."""
    if not summary or 'checks' not in summary:
        return "<section><h2>Summary</h2><p>No summary data available.</p></section>"

    html = """
    <section>
        <h2>Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Results</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
    """

    for check_name, check_summary in summary['checks'].items():
        status = check_summary['status']
        if status == 'PASS':
            status_class = 'pass'
            status_icon = '✅'
        elif status == 'FAIL':
            status_class = 'fail'
            status_icon = '❌'
        else:  # SKIPPED
            status_class = 'skipped'
            status_icon = '⏭️'

        display_name = check_name.replace('_', ' ').title()

        # Extract key metrics from summary for the Results column
        summary_text = check_summary['summary']
        results_text = summary_text

        # Try to extract numbers for cleaner display
        if '/' in summary_text:
            # Extract fraction like "102/102" or "48/51"
            import re

            fraction_match = re.search(r'(\d+/\d+)', summary_text)
            if fraction_match:
                results_text = fraction_match.group(1)

        html += f"""
                <tr class="summary-row-{status_class}">
                    <td class="check-name">{display_name}</td>
                    <td class="status-cell">
                        <span class="status-badge status-{status_class}">
                            {status_icon} {status}
                        </span>
                    </td>
                    <td class="results-cell">{results_text}</td>
                    <td class="details-cell">{summary_text}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </section>
    """
    return html


def _generate_gid_consistency_html(gid_results):
    """Generate GID inconsistencies section - only show failed nodes."""
    if not gid_results:
        return ""

    # Filter to only failed nodes
    failed_nodes = {node: result for node, result in gid_results.items() if result['status'] == 'FAIL'}

    if not failed_nodes:
        return ""  # No failures, no section needed

    html = """
    <section>
        <h2>GID Inconsistencies</h2>
        <p class="error-summary">The following nodes have GID consistency issues:</p>
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Failed Interfaces</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
    """

    for node, result in failed_nodes.items():
        # Build details for failed interfaces only
        failed_interfaces = []
        issues = []

        for iface_name, iface_result in result.get('interfaces', {}).items():
            if iface_result.get('status') != 'OK':
                failed_interfaces.append(iface_name)
                status = iface_result.get('status', 'UNKNOWN')
                error = iface_result.get('error', '')
                issues.append(f"{iface_name}: {status} ({error})" if error else f"{iface_name}: {status}")

        # Add general errors
        if result.get('errors'):
            issues.extend(result['errors'])

        failed_ifaces_str = ', '.join(failed_interfaces) if failed_interfaces else 'Multiple'
        issues_str = '; '.join(issues)

        html += f"""
            <tr>
                <td>{node}</td>
                <td>{failed_ifaces_str}</td>
                <td>{issues_str}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </section>
    """
    return html


def _generate_connectivity_html(connectivity_results):
    """Generate RDMA connection failures section - only show failed pairs."""
    if not connectivity_results:
        return ""

    if connectivity_results.get('skipped', False):
        return ""  # Skip section entirely if test was skipped

    # Check if there are any failures
    failed_pairs = connectivity_results.get('failed_pairs', 0)
    if failed_pairs == 0:
        return ""  # No failures, no section needed

    html = f"""
    <section>
        <h2>RDMA Connection Failures</h2>
        <p class="error-summary">Found {failed_pairs} failed connection(s) out of {connectivity_results.get('total_pairs', 0)} total pairs tested:</p>
        <table>
            <thead>
                <tr>
                    <th>Failed Node Pair</th>
                    <th>Interface</th>
                    <th>Local GID</th>
                    <th>Remote GID</th>
                    <th>Server Command</th>
                    <th>Client Command</th>
                    <th>Error Details</th>
                </tr>
            </thead>
            <tbody>
    """

    # Only show failed pairs
    if connectivity_results.get('pair_results'):
        # Only show failed pairs
        for pair_key, pair_result in connectivity_results['pair_results'].items():
            if pair_result['status'] == 'FAIL':
                # Parse detailed error information
                error_details = pair_result.get('error_details', [])
                local_gid = 'N/A'
                remote_gid = 'N/A'
                gid_index = 'N/A'
                error_msg = 'Connection failed'

                if error_details:
                    # Extract GID and connection details from error messages
                    for error in error_details:
                        if 'Local GID:' in error:
                            local_match = re.search(r'Local GID:\s*([^\s|]+)', error)
                            if local_match:
                                local_gid = local_match.group(1)
                        if 'Remote GID:' in error:
                            remote_match = re.search(r'Remote GID:\s*([^\s|]+)', error)
                            if remote_match:
                                remote_gid = remote_match.group(1)
                        if 'GID Index:' in error:
                            gid_match = re.search(r'GID Index:\s*([^\s|]+)', error)
                            if gid_match:
                                gid_index = gid_match.group(1)

                        # Extract the main error message (before the first |)
                        if '|' in error:
                            error_msg = error.split('|')[0].strip()
                        else:
                            error_msg = error

                # If we still don't have GID info, try to extract from client/server output
                if local_gid == 'N/A' or remote_gid == 'N/A':
                    client_output = pair_result.get('client_output', '')
                    server_output = pair_result.get('server_output', '')

                    # Try to extract from client output first
                    if local_gid == 'N/A':
                        local_match = re.search(r'local address:.*?GID\s+([^\s\n]+)', client_output)
                        if local_match:
                            local_gid = local_match.group(1)

                    if remote_gid == 'N/A':
                        remote_match = re.search(r'remote address:.*?GID\s+([^\s\n]+)', client_output)
                        if remote_match:
                            remote_gid = remote_match.group(1)

                    # Also try to extract from server output as fallback
                    if local_gid == 'N/A':
                        local_match = re.search(r'local address:.*?GID\s+([^\s\n]+)', server_output)
                        if local_match:
                            local_gid = local_match.group(1)

                    if remote_gid == 'N/A':
                        remote_match = re.search(r'remote address:.*?GID\s+([^\s\n]+)', server_output)
                        if remote_match:
                            remote_gid = remote_match.group(1)

                    if gid_index == 'N/A':
                        # Try client output first, then server output
                        gid_idx_match = re.search(r'GID index\s*:\s*(\d+)', client_output)
                        if gid_idx_match:
                            gid_index = gid_idx_match.group(1)
                        else:
                            gid_idx_match = re.search(r'GID index\s*:\s*(\d+)', server_output)
                            if gid_idx_match:
                                gid_index = gid_idx_match.group(1)

                # Extract interface from pair_key if present
                if '(' in pair_key:
                    base_pair = pair_key.split(' (')[0]
                    interface = pair_key.split('(')[1].rstrip(')')
                else:
                    base_pair = pair_key
                    interface = pair_result.get('interface', 'default')

                # Get the actual commands used (stored during execution)
                server_cmd = pair_result.get('server_cmd', 'N/A')
                client_cmd = pair_result.get('client_cmd', 'N/A')
                server_node = pair_result.get('server_node', 'unknown')
                client_node = pair_result.get('client_node', 'unknown')

                # Convert GIDs to more readable format if they're IPv4-mapped
                def format_gid(gid):
                    if gid.startswith('::ffff:') and len(gid.split('.')) == 4:
                        # Extract IP from IPv4-mapped IPv6
                        ip_part = gid.replace('::ffff:', '')
                        return f"{gid}<br><small>({ip_part})</small>"
                    return gid

                formatted_local_gid = format_gid(local_gid) if local_gid != 'N/A' else 'N/A'
                formatted_remote_gid = format_gid(remote_gid) if remote_gid != 'N/A' else 'N/A'

                html += f"""
                    <tr>
                        <td>{base_pair}</td>
                        <td>{interface}</td>
                        <td class="gid-cell">{formatted_local_gid}</td>
                        <td class="gid-cell">{formatted_remote_gid}</td>
                        <td><code>ssh {server_node} "{server_cmd}"</code></td>
                        <td><code>ssh {client_node} "{client_cmd}"</code></td>
                        <td>{error_msg}</td>
                    </tr>
                """

    html += """
            </tbody>
        </table>
    </section>
    """
    return html


def _generate_ssh_connectivity_html(ssh_results):
    """Generate SSH connection failures section - only show failed pairs."""
    if not ssh_results:
        return ""

    if ssh_results.get('skipped', False):
        return ""  # Skip section entirely if test was skipped

    # Check if there are any failures
    failed_pairs = ssh_results.get('failed_pairs', 0)
    if failed_pairs == 0:
        return ""  # No failures, no section needed

    html = f"""
    <section>
        <h2>SSH Connection Failures</h2>
        <p class="error-summary">Found {failed_pairs} failed SSH connection(s) out of {ssh_results.get('total_pairs', 0)} total pairs tested:</p>
        <table>
            <thead>
                <tr>
                    <th>Source Node</th>
                    <th>Target Node</th>
                    <th>Error Details</th>
                </tr>
            </thead>
            <tbody>
    """

    # Only show failed pairs
    if ssh_results.get('pair_results'):
        # Sort failed pairs for consistent display
        failed_pairs_list = []
        for pair_key, status in ssh_results['pair_results'].items():
            if status.startswith('FAILED'):
                # Parse pair key (format: "source → target")
                if ' → ' in pair_key:
                    source_node, target_node = pair_key.split(' → ', 1)
                    error_msg = status.replace('FAILED - ', '') if 'FAILED - ' in status else 'SSH connection failed'
                    failed_pairs_list.append((source_node.strip(), target_node.strip(), error_msg))

        # Sort by source node, then target node
        failed_pairs_list.sort(key=lambda x: (x[0], x[1]))

        # Generate table rows
        for source_node, target_node, error_msg in failed_pairs_list:
            html += f"""
                <tr>
                    <td>{source_node}</td>
                    <td>{target_node}</td>
                    <td class="error-details">{error_msg}</td>
                </tr>
            """

    html += """
            </tbody>
        </table>
    </section>
    """

    return html


def _generate_rocm_versions_html(rocm_results):
    """Generate ROCm version inconsistencies section - only show failed nodes."""
    if not rocm_results:
        return ""

    # Filter to only failed nodes
    failed_nodes = {node: result for node, result in rocm_results.items() if result['status'] == 'FAIL'}

    if not failed_nodes:
        return ""  # No failures, no section needed

    html = """
    <section>
        <h2>ROCm Version Inconsistencies</h2>
        <p class="error-summary">The following nodes have ROCm version mismatches:</p>
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Detected Version</th>
                    <th>Expected Version</th>
                    <th>Issue</th>
                </tr>
            </thead>
            <tbody>
    """

    for node, result in failed_nodes.items():
        errors = ', '.join(result.get('errors', [])) if result.get('errors') else 'Version mismatch'

        html += f"""
            <tr>
                <td>{node}</td>
                <td>{result.get('detected_version', 'Unknown')}</td>
                <td>{result.get('expected_version', 'Unknown')}</td>
                <td>{errors}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </section>
    """
    return html


def _generate_interface_names_html(interface_results):
    """Generate RDMA interface inconsistencies section - only show failed nodes."""
    if not interface_results:
        return ""

    # Filter to only failed nodes
    failed_nodes = {node: result for node, result in interface_results.items() if result['status'] == 'FAIL'}

    if not failed_nodes:
        return ""  # No failures, no section needed

    html = """
    <section>
        <h2>RDMA Interface Inconsistencies</h2>
        <p class="error-summary">The following nodes have RDMA interface issues:</p>
        <table>
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Missing</th>
                    <th>Inactive</th>
                    <th>Down</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
    """

    for node, result in failed_nodes.items():
        missing_interfaces = ', '.join(result.get('missing_interfaces', [])) or 'None'
        inactive_interfaces = ', '.join(result.get('inactive_interfaces', [])) or 'None'
        down_interfaces = ', '.join(result.get('down_interfaces', [])) or 'None'
        errors = ', '.join(result.get('errors', [])) if result.get('errors') else 'Interface issues detected'

        html += f"""
            <tr>
                <td>{node}</td>
                <td>{missing_interfaces}</td>
                <td>{inactive_interfaces}</td>
                <td>{down_interfaces}</td>
                <td>{errors}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </section>
    """
    return html


def test_ssh_full_mesh_connectivity(phdl, node_list, timeout=5, config_dict=None):
    """
    Test SSH connectivity between all cluster nodes (full mesh).

    This validates that every node can SSH to every other node, which is
    critical for MPI job launches and distributed computing frameworks.

    Args:
        phdl: Parallel SSH handle
        node_list: List of cluster nodes
        timeout: SSH connection timeout in seconds
        config_dict: Optional config dictionary for ScriptLet debug mode

    Returns:
        dict: Results with total/successful/failed pairs and detailed failure info
    """
    log.info(f"Testing SSH full mesh connectivity (timeout: {timeout}s)")

    if len(node_list) < 2:
        log.warning("Need at least 2 nodes for SSH mesh testing")
        return {
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'pair_results': {},
            'node_status': {},
            'skipped': True,
        }

    # Get ScriptLet debug setting
    scriptlet_debug = config_dict.get('scriptlet_debug', False) if config_dict else False
    temp_dir = "/tmp/preflight"

    results = {
        'total_pairs': 0,
        'successful_pairs': 0,
        'failed_pairs': 0,
        'pair_results': {},
        'node_status': {},
        'timeout': timeout,
    }

    # Calculate total SSH pairs (each node to every other node)
    total_pairs = len(node_list) * (len(node_list) - 1)
    results['total_pairs'] = total_pairs

    log.info(f"Testing SSH connectivity: {len(node_list)} nodes, {total_pairs} total pairs")

    try:
        from .scriptlet import ScriptLet

        log.info(f"Creating ScriptLet with debug={scriptlet_debug}, temp_dir={temp_dir}")
        with ScriptLet(phdl, debug=scriptlet_debug, temp_dir=temp_dir) as scriptlet:
            # Phase 1: Generate SSH test scripts for each source node
            log.info("Phase 1: Generating SSH test scripts for each node")

            for source_node in node_list:
                # Create list of target nodes (all nodes except self)
                target_nodes = [node for node in node_list if node != source_node]

                # Generate SSH test script for this source node
                script_content = _generate_ssh_test_script(source_node, target_nodes, timeout)
                script_id = f"ssh_test_{source_node}"

                log.info(f"Creating SSH script '{script_id}' for {source_node} → {len(target_nodes)} targets")
                log.debug(f"Script content preview: {script_content[:200]}...")
                scriptlet.create_script(script_id, script_content)
                log.info(f"Successfully created SSH test script for {source_node}")

            # Phase 2: Copy SSH test scripts to nodes
            log.info(f"Phase 2: Copying SSH test scripts to {len(node_list)} nodes")

            # Create script mapping for copying and execution
            script_mapping = {}
            for source_node in node_list:
                script_id = f"ssh_test_{source_node}"
                script_mapping[source_node] = script_id

            # Copy scripts to their respective nodes
            copy_results = scriptlet.copy_script_list(script_mapping)
            log.info(f"Script copy results: {copy_results}")

            # Check if any copies failed
            failed_copies = [node for node, result in copy_results.items() if "FAILED" in result]
            if failed_copies:
                log.error(f"Failed to copy scripts to nodes: {failed_copies}")
                for node in failed_copies[:3]:  # Log first 3 failures
                    log.error(f"  {node}: {copy_results[node]}")

            # Phase 3: Execute SSH test scripts in parallel
            log.info(f"Phase 3: Executing SSH tests on {len(node_list)} nodes in parallel")

            # Calculate reasonable timeout for script execution
            script_timeout = (timeout + 2) * (len(node_list) - 1) + 30  # Buffer for script overhead

            execution_results = scriptlet.run_parallel_group(script_mapping, timeout=script_timeout)

            # Phase 4: Collect and analyze results
            log.info("Phase 4: Collecting SSH test results")

            pair_results = {}
            node_status = {}
            successful_pairs = 0
            failed_pairs = 0

            for source_node in node_list:
                node_status[source_node] = {
                    'total_targets': len(node_list) - 1,
                    'successful_targets': 0,
                    'failed_targets': 0,
                    'execution_status': execution_results.get(source_node, 'UNKNOWN'),
                }

                # Read results from this source node
                try:
                    result_content = execution_results.get(source_node, '')

                    # Parse SSH test results (new compact format: only failures reported)
                    # All pairs are successful unless explicitly reported as failed
                    reported_failures = set()

                    for line in result_content.strip().split('\n'):
                        if '=' in line and 'FAILED:' in line:
                            # Format: "source→target=FAILED:error_message"
                            pair_part, failure_part = line.split('=', 1)
                            if '→' in pair_part and failure_part.startswith('FAILED:'):
                                error_msg = failure_part[7:]  # Remove "FAILED:" prefix
                                pair_key = pair_part.replace('→', ' → ')  # Convert to standard format
                                pair_results[pair_key] = f"FAILED - {error_msg}"
                                reported_failures.add(pair_key)
                                failed_pairs += 1
                                node_status[source_node]['failed_targets'] += 1

                    # All other pairs (not reported as failures) are successful
                    for target_node in node_list:
                        if target_node != source_node:
                            pair_key = f"{source_node} → {target_node}"
                            if pair_key not in reported_failures:
                                pair_results[pair_key] = "SUCCESS"
                                successful_pairs += 1
                                node_status[source_node]['successful_targets'] += 1

                except Exception as e:
                    log.error(f"Failed to read SSH test results from {source_node}: {e}")
                    # Mark all targets as failed for this source node
                    for target_node in node_list:
                        if target_node != source_node:
                            pair_key = f"{source_node} → {target_node}"
                            pair_results[pair_key] = f"SCRIPT_ERROR: {str(e)}"
                            failed_pairs += 1
                            node_status[source_node]['failed_targets'] += 1

            # Update results
            results.update(
                {
                    'successful_pairs': successful_pairs,
                    'failed_pairs': failed_pairs,
                    'pair_results': pair_results,
                    'node_status': node_status,
                }
            )

            # Log summary
            success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0
            log.info(f"SSH mesh connectivity: {successful_pairs}/{total_pairs} pairs successful ({success_rate:.1f}%)")

            if failed_pairs > 0:
                log.warning(f"SSH connectivity issues detected: {failed_pairs} failed connections")

                # Log some example failures for debugging
                failure_examples = [(k, v) for k, v in pair_results.items() if 'SUCCESS' not in v]
                for pair, error in failure_examples[:5]:  # Show first 5 failures
                    log.warning(f"  {pair}: {error}")

                if len(failure_examples) > 5:
                    log.warning(f"  ... and {len(failure_examples) - 5} more failures")

            return results

    except Exception as e:
        log.error(f"SSH full mesh connectivity test failed: {e}")
        return {
            'total_pairs': total_pairs,
            'successful_pairs': 0,
            'failed_pairs': total_pairs,
            'pair_results': {},
            'node_status': {},
            'error': str(e),
        }


def _generate_ssh_test_script(source_node, target_nodes, timeout):
    """
    Generate optimized SSH connectivity test script for a source node.

    This script only reports failures in compact key=value format to minimize
    output size and avoid overwhelming the parallel SSH result collection.

    Args:
        source_node: Node that will run this script
        target_nodes: List of nodes to test SSH connectivity to
        timeout: SSH connection timeout in seconds

    Returns:
        str: Shell script content for SSH testing (reports failures only)
    """
    script_lines = [
        "#!/bin/bash",
        "# Optimized SSH Full Mesh Connectivity Test",
        f"# Source: {source_node}",
        f"# Targets: {len(target_nodes)} nodes",
        "# Only reports failures in key=value format",
        "",
        "# SSH connection options for automated testing",
        f"SSH_OPTS='-o ConnectTimeout={timeout} -o BatchMode=yes -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR'",
        "",
        "# Test SSH connectivity to each target node (report failures only)",
    ]

    for target_node in target_nodes:
        script_lines.extend(
            [
                f"# Test connection to {target_node}",
                f"if ! ssh $SSH_OPTS {target_node} 'exit 0' >/dev/null 2>&1; then",
                "    # Connection failed - capture specific error",
                f"    error=$(ssh $SSH_OPTS {target_node} 'exit 0' 2>&1 | head -1)",
                "    if [ -z \"$error\" ]; then",
                "        error=\"SSH connection failed\"",
                "    fi",
                f"    echo \"{source_node}→{target_node}=FAILED:$error\"",
                "fi",
                "",
            ]
        )

    script_lines.append("# End of SSH connectivity test (failures reported above)")

    return '\n'.join(script_lines)


def _generate_configuration_html(config_dict):
    """Generate configuration section."""
    html = """
    <section>
        <h2>Test Configuration</h2>
        <div class="config-section">
            <pre>
    """

    # Filter out comment fields for cleaner display
    clean_config = {k: v for k, v in config_dict.items() if not k.startswith('_comment')}
    html += json.dumps(clean_config, indent=2)

    html += """
            </pre>
        </div>
    </section>
    """
    return html


def _generate_recommendations_html(recommendations):
    """Generate recommendations section."""
    if not recommendations:
        return ""

    html = """
    <section>
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
    """

    for recommendation in recommendations:
        html += f"<li>{recommendation}</li>"

    html += """
            </ul>
        </div>
    </section>
    """
    return html
