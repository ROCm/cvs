"""
System logs API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


def validate_grep_command(grep_cmd: str) -> tuple[bool, str]:
    """
    Validate that grep command is safe (only grep/egrep with safe flags).

    Args:
        grep_cmd: User-provided grep command (may include pipes)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not grep_cmd or not grep_cmd.strip():
        return False, "Empty command"

    # Dangerous characters that should never appear
    dangerous_chars = [';', '&', '$', '`', '(', ')', '{', '}', '<', '>', '\n', '\r']
    for char in dangerous_chars:
        if char in grep_cmd:
            return False, f"Invalid character '{char}' in command"

    # Dangerous keywords
    dangerous_keywords = ['bash', 'sh', 'exec', 'eval', 'sudo', 'rm', 'mv', 'cp', 'dd', 'cat', 'tee', 'chmod', 'chown']
    cmd_lower = grep_cmd.lower()
    for keyword in dangerous_keywords:
        if keyword in cmd_lower:
            return False, f"Command contains forbidden keyword: {keyword}"

    # Split by pipe and validate each segment
    segments = grep_cmd.split('|')

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Each segment must start with grep or egrep
        if not (segment.startswith('grep ') or segment.startswith('egrep ')):
            return False, f"Invalid segment (must start with 'grep' or 'egrep'): {segment}"

        # Extract command name and check allowed flags
        parts = segment.split()
        # cmd_name = parts[0]  # 'grep' or 'egrep'

        # Check flags (everything between command and pattern)
        allowed_flags = ['-i', '-v', '-E', '-A', '-B', '-C', '-w', '-x', '-o', '-n', '-c', '-m']
        for part in parts[1:]:
            # If it starts with -, it's a flag or flag argument
            if part.startswith('-'):
                # Check if it's a known flag or combined flags like -iE
                if part not in allowed_flags:
                    # Check if it's combined flags
                    if not all(f'-{c}' in allowed_flags for c in part[1:] if c.isalpha()):
                        # Might be flag with argument like -A 5
                        flag_base = part.split('=')[0] if '=' in part else part
                        if flag_base not in allowed_flags and flag_base.rstrip('0123456789') not in allowed_flags:
                            return False, f"Invalid flag: {part}"

    return True, ""


@router.get("/dmesg")
async def get_dmesg_errors() -> Dict[str, Any]:
    """
    Get dmesg error logs from all cluster nodes.

    Collects: :emerg, :alert, :crit, :err level messages
    """
    from app.main import app_state

    logger.info("API: /api/logs/dmesg endpoint called")

    if not app_state.ssh_manager:
        logger.error("SSH manager not initialized")
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        from app.collectors.logs_collector import LogsCollector

        collector = LogsCollector()
        logs_data = await collector.collect_all_logs(app_state.ssh_manager)

        # Count nodes with actual data
        amd_with_data = sum(1 for v in logs_data.get("amd_logs", {}).values() if isinstance(v, str) and v.strip())
        dmesg_with_data = sum(1 for v in logs_data.get("dmesg_errors", {}).values() if isinstance(v, str) and v.strip())
        userspace_with_data = sum(
            1 for v in logs_data.get("userspace_errors", {}).values() if isinstance(v, str) and v.strip()
        )

        logger.info(
            f"API: Returning logs - {amd_with_data} nodes with AMD logs, {dmesg_with_data} with dmesg errors, {userspace_with_data} with userspace errors"
        )

        return logs_data

    except Exception as e:
        logger.error(f"API: Failed to collect logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to collect logs: {str(e)}")


@router.get("/search")
async def search_dmesg_logs(
    grep_command: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="grep/egrep command (e.g., \"grep -i 'error' | grep -v 'vital'\")",
    ),
) -> Dict[str, Any]:
    """
    Search dmesg logs across all cluster nodes using custom grep command.

    Allows powerful grep/egrep pipe commands with strict validation for security.

    Args:
        grep_command: grep/egrep command with pipes (e.g., "grep -i 'error' | grep -v 'vital'")

    Returns:
        {
            "timestamp": "...",
            "grep_command": "validated_command",
            "results": {
                "node1": "first 5 matching lines",
                "node2": "first 5 matching lines",
                ...
            }
        }
    """
    from app.main import app_state

    logger.info(f"API: /api/logs/search endpoint called with grep_command: {grep_command}")

    if not app_state.ssh_manager:
        logger.error("SSH manager not initialized")
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    # Validate grep command for security
    is_valid, error_msg = validate_grep_command(grep_command)
    if not is_valid:
        logger.warning(f"Invalid grep command rejected: {grep_command} - Reason: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid grep command: {error_msg}")

    logger.info(f"Grep command validated successfully: {grep_command}")

    try:
        # Build safe command: sudo dmesg -T | <validated_grep_command> | head -5
        # Add head -5 to limit output per node
        # IMPORTANT: Use single quotes for outer bash -c, escape single quotes in grep_command
        # Bash single quote escaping: replace ' with '\'' (end quote, escaped quote, start quote)
        escaped_grep_cmd = grep_command.replace("'", "'\\''")
        cmd = f"bash -c 'sudo dmesg -T 2>/dev/null | {escaped_grep_cmd} | head -5 || echo \"\"'"

        logger.info(f"Executing search on {len(app_state.ssh_manager.get_reachable_hosts())} nodes")
        logger.info(f"Command: {cmd[:200]}...")

        # Execute with 60 second timeout
        results = await app_state.ssh_manager.exec_async(cmd, timeout=60)

        # Filter out empty results and errors
        search_results = {}
        nodes_with_results = 0

        for node, output in results.items():
            if output and not output.startswith("ERROR") and not output.startswith("ABORT") and output.strip():
                search_results[node] = output.strip()
                nodes_with_results += 1

        logger.info(f"Search complete: {nodes_with_results} nodes have matching logs")

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "grep_command": grep_command,
            "results": search_results,
            "total_nodes_searched": len(results),
            "nodes_with_results": nodes_with_results,
        }

    except Exception as e:
        logger.error(f"API: Failed to search logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")
