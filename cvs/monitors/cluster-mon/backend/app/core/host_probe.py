"""
TCP socket-based host reachability probing.

Provides lightweight TCP connection testing to quickly determine which hosts
are reachable before attempting SSH connections.
"""

import socket
import time
import logging
import paramiko
import json
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def tcp_probe(host: str, port: int = 22, timeout: int = 5) -> Tuple[str, bool]:
    """
    Attempt a TCP connection to host:port to test reachability.

    This is much faster than SSH connection attempts (~5 seconds vs 60+ seconds
    for unreachable hosts) and allows quick discovery of which hosts are online.

    Args:
        host: Hostname or IP address to probe
        port: TCP port to connect to (default 22 for SSH)
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (host, is_reachable) where is_reachable is True if connection succeeded
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.close()
        return host, True
    except Exception:
        # Any exception (timeout, connection refused, etc.) means unreachable
        return host, False


def discover_reachable_hosts(
    hosts: List[str], port: int = 22, timeout: int = 5, max_workers: int = 100
) -> Tuple[List[str], List[str]]:
    """
    Probe multiple hosts in parallel to determine which are reachable.

    Uses ThreadPoolExecutor to probe many hosts concurrently. For 617 nodes,
    this completes in ~10 seconds with 100 workers (reduced for stability).

    Args:
        hosts: List of hostnames/IPs to probe
        port: TCP port to probe (default 22 for SSH)
        timeout: Per-host timeout in seconds (default 5)
        max_workers: Maximum number of concurrent probe threads (default 100)

    Returns:
        Tuple of (reachable_hosts, unreachable_hosts)
    """
    if not hosts:
        return [], []

    logger.info(f"Probing {len(hosts)} hosts for reachability (port {port}, timeout {timeout}s)...")
    probe_start = time.time()

    reachable = []
    unreachable = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all probe tasks
        futures = {executor.submit(tcp_probe, host, port, timeout): host for host in hosts}

        # Collect results as they complete
        for future in as_completed(futures):
            host, is_reachable = future.result()
            if is_reachable:
                reachable.append(host)
            else:
                unreachable.append(host)

    probe_duration = time.time() - probe_start
    logger.info(f"Probe completed in {probe_duration:.2f}s: {len(reachable)} reachable, {len(unreachable)} unreachable")

    return reachable, unreachable


def probe_from_bastion(
    jump_client: paramiko.SSHClient, hosts: List[str], port: int = 22, timeout: int = 5
) -> Tuple[str, List[str]]:
    """
    Probe cluster nodes from a bastion/jump host.

    Executes Python script on the jump host to perform TCP probes from there.
    This is necessary when cluster nodes are only accessible from the jump host.

    Args:
        jump_client: Connected paramiko.SSHClient to the jump host
        hosts: List of cluster node hostnames/IPs to probe
        port: TCP port to probe (default 22 for SSH)
        timeout: Per-host timeout in seconds

    Returns:
        Tuple of (reachable_hosts, unreachable_hosts)

    Raises:
        Exception: If jump host execution fails or returns invalid JSON
    """
    logger.info(f"Probing {len(hosts)} cluster nodes via bastion (port {port}, timeout {timeout}s)...")
    probe_start = time.time()

    # Build Python script to run on jump host
    # Uses same tcp_probe logic but outputs JSON
    probe_script = f"""
import socket
import json
import sys

hosts = {hosts}
port = {port}
timeout = {timeout}

reachable = []
unreachable = []

for host in hosts:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.close()
        reachable.append(host)
    except Exception:
        unreachable.append(host)

# Output JSON result
print(json.dumps({{"reachable": reachable, "unreachable": unreachable}}))
"""

    try:
        # Execute probe script on jump host
        # Use heredoc to avoid quoting issues
        stdin, stdout, stderr = jump_client.exec_command(
            f"python3 - <<'EOF'\n{probe_script}\nEOF",
            timeout=max(60, len(hosts) * timeout // 10),  # Generous timeout based on node count
        )

        # Read output
        result_output = stdout.read().decode('utf-8', errors='ignore').strip()
        error_output = stderr.read().decode('utf-8', errors='ignore').strip()

        if error_output:
            logger.warning(f"Probe script stderr: {error_output[:200]}")

        if not result_output:
            raise Exception(f"No output from probe script. stderr: {error_output}")

        # Parse JSON result
        result = json.loads(result_output)
        reachable = result.get("reachable", [])
        unreachable = result.get("unreachable", [])

        probe_duration = time.time() - probe_start
        logger.info(
            f"Probe via bastion completed in {probe_duration:.2f}s: "
            f"{len(reachable)} reachable, {len(unreachable)} unreachable"
        )

        return reachable, unreachable

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse probe results as JSON: {e}")
        logger.error(f"Output was: {result_output[:500]}")
        raise Exception(f"Probe script returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Failed to probe from bastion: {e}")
        raise
