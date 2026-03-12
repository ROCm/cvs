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

        cmd = "bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err 2>/dev/null || echo \"\"'"
        logger.info(f"Running command: {cmd}")
        output = await ssh_manager.exec_async(cmd, timeout=60)

        logger.info(f"Received output from {len(output)} nodes")

        logs = {}
        nodes_with_errors = 0
        nodes_empty = 0
        nodes_failed = 0

        for node, log_output in output.items():
            if log_output.startswith("ERROR") or log_output.startswith("ABORT"):
                logs[node] = {"error": "Failed to collect logs"}
                nodes_failed += 1
                logger.warning(f"Node {node}: Failed to collect logs - {log_output[:100]}")
            else:
                # Store the log output (may be empty if no errors)
                stripped_output = log_output.strip()
                logs[node] = stripped_output
                if stripped_output:
                    nodes_with_errors += 1
                    logger.info(f"Node {node}: Found {len(stripped_output)} chars of dmesg errors")
                else:
                    nodes_empty += 1

        logger.info(
            f"Dmesg error logs collected from {len(logs)} nodes: {nodes_with_errors} with errors, {nodes_empty} clean, {nodes_failed} failed"
        )
        return logs

    async def collect_amd_logs(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect AMD-specific hardware and driver error logs from dmesg.

        Captures emerg, alert, crit, err levels filtered for AMD hardware components:
        - PCIe, XGMI (interconnect)
        - amdgpu (GPU driver)
        - epyc, cpu (CPU)
        - ionic, bnxt, mlnx, mellanox (NICs)
        - Link (connectivity)

        Returns:
            {
                "node1": "AMD-specific error log output...",
                "node2": "AMD-specific error log output...",
                ...
            }
        """
        logger.info("Collecting AMD-specific hardware/driver error logs from all nodes")

        # Filter for AMD-specific components (case-insensitive)
        # cmd = """bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | grep -iE "PCIe|XGMI|amdgpu|epyc|cpu|ionic|bnxt|mlnx|mellanox|Link" 2>/dev/null || echo ""'"""
        cmd = """bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | grep -iE "PCIe|XGMI|amdgpu|epyc|cpu|ionic|bnxt|mlnx|mellanox|Link|error|fail" 2>/dev/null | grep -iv "vital buffer"  2>/dev/null || echo ""'"""
        logger.info(f"Running command: {cmd[:150]}...")
        output = await ssh_manager.exec_async(cmd, timeout=60)

        logger.info(f"Received output from {len(output)} nodes")

        logs = {}
        nodes_with_errors = 0
        nodes_empty = 0
        nodes_failed = 0

        for node, log_output in output.items():
            if log_output.startswith("ERROR") or log_output.startswith("ABORT"):
                logs[node] = {"error": "Failed to collect AMD logs"}
                nodes_failed += 1
                logger.warning(f"Node {node}: Failed to collect AMD logs - {log_output[:100]}")
            else:
                # Store the log output (may be empty if no AMD errors)
                stripped_output = log_output.strip()
                logs[node] = stripped_output
                if stripped_output:
                    nodes_with_errors += 1
                    logger.info(f"Node {node}: Found {len(stripped_output)} chars of AMD-specific errors")
                else:
                    nodes_empty += 1

        logger.info(
            f"AMD logs collected from {len(logs)} nodes: {nodes_with_errors} with errors, {nodes_empty} clean, {nodes_failed} failed"
        )
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
        cmd = """bash -c 'sudo dmesg --decode -T -l emerg,alert,crit,err,warn 2>/dev/null | egrep -i "oom|out of memory|killed process|segfault|general protection|call trace|bug:|hardware error|mce|stack trace|pytorch|torch|tensorflow|megatron|jax|vllm|sglang|triton.*error|triton.*exception|triton.*failed" 2>/dev/null || echo ""'"""
        logger.info(f"Running command: {cmd[:150]}...")
        output = await ssh_manager.exec_async(cmd, timeout=60)

        logger.info(f"Received output from {len(output)} nodes")

        logs = {}
        nodes_with_errors = 0
        nodes_empty = 0
        nodes_failed = 0

        for node, log_output in output.items():
            if log_output.startswith("ERROR") or log_output.startswith("ABORT"):
                logs[node] = {"error": "Failed to collect logs"}
                nodes_failed += 1
                logger.warning(f"Node {node}: Failed to collect userspace logs - {log_output[:100]}")
            else:
                # Store the log output (may be empty if no errors)
                stripped_output = log_output.strip()
                logs[node] = stripped_output
                if stripped_output:
                    nodes_with_errors += 1
                    logger.info(f"Node {node}: Found {len(stripped_output)} chars of userspace errors")
                else:
                    nodes_empty += 1

        logger.info(
            f"Userspace error logs collected from {len(logs)} nodes: {nodes_with_errors} with errors, {nodes_empty} clean, {nodes_failed} failed"
        )
        return logs

    async def collect_all_logs(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all system logs including AMD-specific hardware/driver logs.
        """
        logger.info("Collecting all system logs")

        amd_logs = await self.collect_amd_logs(ssh_manager)
        dmesg_logs = await self.collect_dmesg_errors(ssh_manager)
        userspace_logs = await self.collect_userspace_errors(ssh_manager)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "amd_logs": amd_logs,
            "dmesg_errors": dmesg_logs,
            "userspace_errors": userspace_logs,
        }
