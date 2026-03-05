"""
System logs collector for dmesg errors.
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LogsCollector:
    """Collects system error logs from dmesg."""

    async def collect_dmesg_errors(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect critical system errors from dmesg.

        Runs: sudo dmesg --decode -T -l emerg,alert,crit,err

        Returns:
            {
                "node1": "error log output...",
                "node2": "error log output...",
                ...
            }
        """
        logger.info("Collecting dmesg error logs from all nodes")

        cmd = "sudo dmesg --decode -T -l emerg,alert,crit,err 2>/dev/null || echo ''"
        output = await ssh_manager.exec_async(cmd)

        logs = {}
        for node, log_output in output.items():
            if log_output.startswith("ERROR") or log_output.startswith("ABORT"):
                logs[node] = {"error": "Failed to collect logs"}
            else:
                # Store the log output (may be empty if no errors)
                logs[node] = log_output.strip()

        logger.info(f"Dmesg error logs collected from {len(logs)} nodes")
        return logs

    async def collect_userspace_errors(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect userspace errors including OOM, segfaults, crashes, and ML framework errors.

        Captures:
        - OOM kills, segfaults, general protection faults
        - Call traces, stack traces, kernel bugs
        - Hardware errors, MCE
        - ML framework errors: PyTorch, TensorFlow, Megatron, JAX, VLLM, SGLang, Triton

        Returns:
            {
                "node1": "userspace error log output...",
                "node2": "userspace error log output...",
                ...
            }
        """
        logger.info("Collecting userspace error logs (OOM, segfaults, crashes, ML frameworks) from all nodes")

        # Comprehensive error pattern including ML frameworks
        # Use -l to filter levels first, then egrep for userspace patterns
        cmd = """sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | egrep -i 'oom|out of memory|killed process|segfault|general protection|call trace|bug:|hardware error|mce|stack trace|pytorch|torch|tensorflow|megatron|jax|vllm|sglang|triton.*error|triton.*exception|triton.*failed' 2>/dev/null || echo ''"""
        output = await ssh_manager.exec_async(cmd)

        logs = {}
        for node, log_output in output.items():
            if log_output.startswith("ERROR") or log_output.startswith("ABORT"):
                logs[node] = {"error": "Failed to collect logs"}
            else:
                # Store the log output (may be empty if no errors)
                logs[node] = log_output.strip()

        logger.info(f"Userspace error logs collected from {len(logs)} nodes")
        return logs

    async def collect_all_logs(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all system logs.
        """
        logger.info("Collecting all system logs")

        dmesg_logs = await self.collect_dmesg_errors(ssh_manager)
        userspace_logs = await self.collect_userspace_errors(ssh_manager)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dmesg_errors": dmesg_logs,
            "userspace_errors": userspace_logs,
        }
