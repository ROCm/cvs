"""
Shared utility for parallel SSH command collection via the Go gpu-collector binary.
All nodes run simultaneously; all commands run in parallel per node.
Falls back gracefully when the binary is unavailable or when using JumpHostPssh.
"""

import json
import logging
import os
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_GO_BINARY = os.environ.get("GPU_COLLECTOR_BIN", "/usr/local/bin/gpu-collector")


def collect_parallel(
    ssh_manager,
    commands: Dict[str, str],
    timeout: int = 60,
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Run multiple shell commands across all nodes using the Go binary.

    Args:
        ssh_manager: Pssh instance with .reachable_hosts, .user, .pkey, .password
        commands:    {name: shell_command} — all commands run in parallel per node
        timeout:     per-command timeout in seconds (global timeout = timeout + 20s)

    Returns:
        {name: {host: output_str}} — same format as ssh_manager.exec() per command,
        or None if the binary is unavailable or errored (caller should fall back).
    """
    from app.core.jump_host_pssh import JumpHostPssh

    if not os.path.isfile(_GO_BINARY):
        logger.debug(f"Go binary not found at {_GO_BINARY}, using parallel-ssh fallback")
        return None

    if isinstance(ssh_manager, JumpHostPssh):
        logger.debug("Jump host mode detected, using parallel-ssh fallback")
        return None

    if not ssh_manager.reachable_hosts:
        # No hosts — return empty result dicts, no fallback needed
        return {name: {} for name in commands}

    inp = {
        "hosts": ssh_manager.reachable_hosts,
        "ssh_user": ssh_manager.user,
        "ssh_key_path": ssh_manager.pkey or "",
        "ssh_password": ssh_manager.password or "",
        "ssh_port": 22,
        "commands": commands,
        "per_host_timeout_s": timeout,
        "global_timeout_s": timeout + 20,
    }

    try:
        proc = subprocess.run(
            [_GO_BINARY],
            input=json.dumps(inp).encode(),
            capture_output=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Go binary timed out after {timeout + 30}s")
        return None
    except Exception as e:
        logger.error(f"Go binary execution failed: {e}")
        return None

    if proc.returncode != 0:
        logger.error(f"Go binary exited {proc.returncode}: {proc.stderr.decode()[:300]}")
        return None

    try:
        go_out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        logger.error(f"Go binary output parse error: {e}")
        return None

    raw_results = go_out.get("results", {})
    duration_ms = go_out.get("collection_duration_ms", 0)
    unreachable = go_out.get("unreachable", [])

    logger.info(f"Go binary: {len(raw_results)} nodes in {duration_ms}ms, {len(unreachable)} unreachable")

    # Convert {host: {cmd_name: {status, raw}}} → {cmd_name: {host: output_str}}
    by_cmd: Dict[str, Dict[str, str]] = {name: {} for name in commands}
    for host, host_results in raw_results.items():
        for name, result in host_results.items():
            if name in by_cmd:
                if result.get("status") == "ok":
                    by_cmd[name][host] = result.get("raw", "")
                else:
                    by_cmd[name][host] = f"ERROR: {result.get('error', 'unknown error')}"

    return by_cmd
