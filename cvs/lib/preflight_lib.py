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
    try:
        rdma_dict = linux_utils.get_rdma_nic_dict(phdl)
    except Exception as e:
        log.warning(f"Failed to get RDMA interfaces via rdma link, falling back to /sys/class/infiniband: {e}")
        # Fallback to direct /sys/class/infiniband listing
        cmd = """
        if [ -d /sys/class/infiniband ]; then
            ls /sys/class/infiniband/ 2>/dev/null || echo 'NO_INTERFACES'
        else
            echo 'NO_INFINIBAND_DIR'
        fi
        """
        out_dict = phdl.exec(cmd)
        rdma_dict = {}
        for node, output in out_dict.items():
            if 'NO_INTERFACES' in output or 'NO_INFINIBAND_DIR' in output:
                rdma_dict[node] = {}
            else:
                # Convert to format similar to get_rdma_nic_dict
                interfaces = [iface.strip() for iface in output.strip().split('\n') if iface.strip()]
                rdma_dict[node] = {iface: {'eth_device': f'{iface}'} for iface in interfaces}

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


def generate_full_mesh_batches(node_list):
    """
    Generate optimal batches for full mesh testing with no overlapping nodes.

    Args:
        node_list: List of node names

    Returns:
        list: List of batches, each containing non-overlapping pairs
    """
    n = len(node_list)
    if n < 2:
        return []

    batches = []

    # Use round-robin tournament algorithm for perfect matching
    for round_num in range(n - 1):
        batch = []

        # Generate perfect matching for this round
        for i in range(n // 2):
            node1_idx = i
            node2_idx = (round_num - i) % (n - 1)
            if node2_idx >= i:
                node2_idx += 1

            batch.append((node_list[node1_idx], node_list[node2_idx]))

        batches.append(batch)

    return batches


def check_rdma_connectivity(
    phdl, node_list, mode="basic", port_range="9000-9999", timeout=10, expected_interfaces=None, gid_index="3"
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
    }

    # Initialize node status
    for node in node_list:
        results['node_status'][node] = {'server_tests': 0, 'client_tests': 0, 'successful_tests': 0, 'failed_tests': 0}

    if mode == "full_mesh":
        # Use batched approach for full mesh
        batches = generate_full_mesh_batches(node_list)

        for batch_num, batch_pairs in enumerate(batches):
            log.info(f"Running batch {batch_num + 1}/{len(batches)} with {len(batch_pairs)} pairs")
            batch_results = _run_ibv_rc_pingpong_batch(
                phdl, batch_pairs, port_start + batch_num * 100, timeout, expected_interfaces
            )

            # Merge batch results
            for pair_key, pair_result in batch_results.items():
                results['pair_results'][pair_key] = pair_result

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

            # Calculate total tests for this batch (pairs × interfaces)
            num_interfaces = len(expected_interfaces) if expected_interfaces else 1
            results['total_pairs'] += len(batch_pairs) * num_interfaces

            # Brief pause between batches
            if batch_num < len(batches) - 1:
                time.sleep(5)

    else:
        # Single batch for basic or sample mode
        pairs = generate_node_pairs(node_list, mode)
        batch_results = _run_ibv_rc_pingpong_batch(phdl, pairs, port_start, timeout, expected_interfaces, gid_index)

        # Calculate total tests (pairs × interfaces)
        num_interfaces = len(expected_interfaces) if expected_interfaces else 1
        results['total_pairs'] = len(pairs) * num_interfaces
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


def _run_ibv_rc_pingpong_batch(phdl, pairs, base_port, timeout, expected_interfaces=None, gid_index="3"):
    """
    Run a batch of ibv_rc_pingpong tests in parallel across all specified interfaces.
    Uses direct IB verbs for more accurate RCCL-like connectivity testing.

    Args:
        phdl: Parallel SSH handle
        pairs: List of (server_node, client_node) tuples
        base_port: Base port number for this batch
        timeout: Timeout in seconds
        expected_interfaces: List of RDMA interfaces to test (if None, uses first available)
        gid_index: GID index to use for connections (default "3" matches RCCL)

    Returns:
        dict: Results for this batch (includes results for each interface)
    """
    if not pairs:
        return {}

    # If no interfaces specified, use default behavior
    if not expected_interfaces:
        expected_interfaces = ["default"]

    all_results = {}

    # Test each interface separately
    for iface_idx, interface in enumerate(expected_interfaces):
        log.info(f"Testing interface {interface} with ibv_rc_pingpong ({iface_idx + 1}/{len(expected_interfaces)})")

        interface_results = {}

        # Build command lists in host_list order for this interface
        server_cmd_list = []
        client_cmd_list = []

        # Create mapping of what each host should do for this interface
        host_server_cmds = {}
        host_client_cmds = {}

        for i, (server_node, client_node) in enumerate(pairs):
            # Use unique port for each interface and pair combination
            port = base_port + (iface_idx * 100) + i

            # Server command for this pair and interface
            if interface != "default":
                server_cmd = f"timeout {timeout + 10} ibv_rc_pingpong -d {interface} -g {gid_index} -p {port} > /tmp/ibv_rc_server_{interface}_{port}.log 2>&1 &"
            else:
                server_cmd = f"timeout {timeout + 10} ibv_rc_pingpong -g {gid_index} -p {port} > /tmp/ibv_rc_server_default_{port}.log 2>&1 &"

            if server_node not in host_server_cmds:
                host_server_cmds[server_node] = []
            host_server_cmds[server_node].append(server_cmd)

            # Client command for this pair and interface (will be run after servers start)
            if interface != "default":
                client_cmd = f"timeout {timeout} ibv_rc_pingpong -d {interface} -g {gid_index} -p {port} {server_node} > /tmp/ibv_rc_client_{interface}_{port}.log 2>&1"
            else:
                client_cmd = f"timeout {timeout} ibv_rc_pingpong -g {gid_index} -p {port} {server_node} > /tmp/ibv_rc_client_default_{port}.log 2>&1"

            if client_node not in host_client_cmds:
                host_client_cmds[client_node] = []
            host_client_cmds[client_node].append((client_cmd, port, server_node))

        # Build server command list in host_list order
        for host in phdl.host_list:
            if host in host_server_cmds:
                # Combine all server commands for this host
                combined_cmd = "; ".join(host_server_cmds[host])
                server_cmd_list.append(combined_cmd)
            else:
                server_cmd_list.append("echo 'No server role for this host'")

        # Execute server commands
        log.info(f"Starting ibv_rc_pingpong servers for interface {interface}")
        phdl.exec_cmd_list(server_cmd_list)

        # Wait for servers to start
        time.sleep(5)

        # Build and execute client commands
        client_cmd_list = []
        for host in phdl.host_list:
            if host in host_client_cmds:
                # Run client commands sequentially for this host
                cmds = [cmd_info[0] for cmd_info in host_client_cmds[host]]
                combined_cmd = "; ".join(cmds)
                client_cmd_list.append(combined_cmd)
            else:
                client_cmd_list.append("echo 'No client role for this host'")

        log.info(f"Starting ibv_rc_pingpong clients for interface {interface}")
        phdl.exec_cmd_list(client_cmd_list)

        # Wait for clients to complete
        time.sleep(timeout + 5)

        # Collect results for each pair
        for i, (server_node, client_node) in enumerate(pairs):
            port = base_port + (iface_idx * 100) + i
            pair_key = f"{server_node} ↔ {client_node} ({interface})"

            # Check client results
            if interface != "default":
                client_log_cmd = f"cat /tmp/ibv_rc_client_{interface}_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"
                server_log_cmd = f"cat /tmp/ibv_rc_server_{interface}_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"
            else:
                client_log_cmd = f"cat /tmp/ibv_rc_client_default_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"
                server_log_cmd = f"cat /tmp/ibv_rc_server_default_{port}.log 2>/dev/null || echo 'LOG_NOT_FOUND'"

            client_results = phdl.exec(client_log_cmd)
            client_output = client_results.get(client_node, 'LOG_NOT_FOUND')

            # Check server results
            server_results = phdl.exec(server_log_cmd)
            server_output = server_results.get(server_node, 'LOG_NOT_FOUND')

            # Analyze results
            success = _analyze_ibv_rc_pingpong_output(client_output, server_output)

            interface_results[pair_key] = {
                'status': 'PASS' if success else 'FAIL',
                'server_node': server_node,
                'client_node': client_node,
                'interface': interface,
                'port': port,
                'client_output': client_output,
                'server_output': server_output,
                'error_details': [] if success else _extract_ibv_rc_pingpong_errors(client_output, server_output),
            }

        # Add interface results to overall results
        all_results.update(interface_results)

    # Cleanup log files
    cleanup_cmd = "rm -f /tmp/ibv_rc_*.log"
    phdl.exec(cleanup_cmd)

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


def generate_preflight_summary(gid_results, connectivity_results, rocm_results, interface_results):
    """
    Generate a comprehensive summary of all preflight check results.

    Args:
        gid_results: Results from GID consistency check
        connectivity_results: Results from RDMA connectivity check
        rocm_results: Results from ROCm version check
        interface_results: Results from interface name check

    Returns:
        dict: Comprehensive summary of all checks
    """
    summary = {
        'overall_status': 'PASS',
        'checks': {
            'gid_consistency': _summarize_gid_results(gid_results),
            'rdma_connectivity': _summarize_connectivity_results(connectivity_results),
            'rocm_versions': _summarize_rocm_results(rocm_results),
            'interface_names': _summarize_interface_results(interface_results),
        },
        'recommendations': [],
    }

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
    if connectivity_results.get('skipped', False):
        return {
            'status': 'SKIPPED',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'mode': connectivity_results['mode'],
            'summary': 'RDMA connectivity test skipped by configuration',
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
                        <td>{error_msg}</td>
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
