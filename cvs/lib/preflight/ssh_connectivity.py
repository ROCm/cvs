"""
SSH Connectivity Testing Module

This module provides functionality for testing SSH connectivity across cluster nodes.
"""

from cvs.lib import globals
from cvs.lib.preflight.base import PreflightCheck

log = globals.log


class SshConnectivityCheck(PreflightCheck):
    """Check SSH connectivity across cluster nodes."""

    def __init__(self, phdl, node_list, timeout=5, config_dict=None):
        """
        Initialize SSH connectivity check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            node_list: List of cluster nodes to test
            timeout: SSH connection timeout in seconds
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.node_list = node_list
        self.timeout = timeout

    def run(self):
        """
        Execute SSH connectivity test across all cluster nodes.

        Returns:
            dict: Results with SSH connectivity status
        """
        return self._run_full_mesh_ssh()

    def _generate_ssh_test_script(self, source_node, target_nodes):
        """
        Generate optimized SSH connectivity test script for a source node.

        Reports failures in compact key=value format only.
        """
        script_lines = [
            "#!/bin/bash",
            "# Optimized SSH Full Mesh Connectivity Test",
            f"# Source: {source_node}",
            f"# Targets: {len(target_nodes)} nodes",
            "# Only reports failures in key=value format",
            "",
            "# SSH connection options for automated testing",
            f"SSH_OPTS='-o ConnectTimeout={self.timeout} -o BatchMode=yes -o StrictHostKeyChecking=no -o PasswordAuthentication=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR'",
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

        return "\n".join(script_lines)

    def _run_full_mesh_ssh(self):
        """Test SSH connectivity between all cluster nodes (full mesh)."""
        log.info(f"Testing SSH full mesh connectivity (timeout: {self.timeout}s)")

        if len(self.node_list) < 2:
            log.warning("Need at least 2 nodes for SSH mesh testing")
            return {
                "total_pairs": 0,
                "successful_pairs": 0,
                "failed_pairs": 0,
                "pair_results": {},
                "node_status": {},
                "skipped": True,
            }

        scriptlet_debug = self.config_dict.get("scriptlet_debug", False) if self.config_dict else False
        temp_dir = "/tmp/preflight"

        results = {
            "total_pairs": 0,
            "successful_pairs": 0,
            "failed_pairs": 0,
            "pair_results": {},
            "node_status": {},
            "timeout": self.timeout,
        }

        total_pairs = len(self.node_list) * (len(self.node_list) - 1)
        results["total_pairs"] = total_pairs

        log.info(f"Testing SSH connectivity: {len(self.node_list)} nodes, {total_pairs} total pairs")

        try:
            from cvs.lib.scriptlet import ScriptLet

            log.info(f"Creating ScriptLet with debug={scriptlet_debug}, temp_dir={temp_dir}")
            with ScriptLet(self.phdl, debug=scriptlet_debug, temp_dir=temp_dir) as scriptlet:
                log.info("Phase 1: Generating SSH test scripts for each node")

                for source_node in self.node_list:
                    target_nodes = [node for node in self.node_list if node != source_node]
                    script_content = self._generate_ssh_test_script(source_node, target_nodes)
                    script_id = f"ssh_test_{source_node}"

                    log.info(f"Creating SSH script '{script_id}' for {source_node} → {len(target_nodes)} targets")
                    log.debug(f"Script content preview: {script_content[:200]}...")
                    scriptlet.create_script(script_id, script_content)
                    log.info(f"Successfully created SSH test script for {source_node}")

                log.info(f"Phase 2: Copying SSH test scripts to {len(self.node_list)} nodes")

                script_mapping = {source_node: f"ssh_test_{source_node}" for source_node in self.node_list}

                copy_results = scriptlet.copy_script_list(script_mapping)
                log.info(f"Script copy results: {copy_results}")

                failed_copies = [node for node, result in copy_results.items() if "FAILED" in result]
                if failed_copies:
                    log.error(f"Failed to copy scripts to nodes: {failed_copies}")
                    for node in failed_copies[:3]:
                        log.error(f"  {node}: {copy_results[node]}")

                log.info(f"Phase 3: Executing SSH tests on {len(self.node_list)} nodes in parallel")

                script_timeout = (self.timeout + 2) * (len(self.node_list) - 1) + 30

                execution_results = scriptlet.run_parallel_group(script_mapping, timeout=script_timeout)

                log.info("Phase 4: Collecting SSH test results")

                pair_results = {}
                node_status = {}
                successful_pairs = 0
                failed_pairs = 0

                for source_node in self.node_list:
                    node_status[source_node] = {
                        "total_targets": len(self.node_list) - 1,
                        "successful_targets": 0,
                        "failed_targets": 0,
                        "execution_status": execution_results.get(source_node, "UNKNOWN"),
                    }

                    try:
                        result_content = execution_results.get(source_node, "")
                        reported_failures = set()

                        for line in result_content.strip().split("\n"):
                            if "=" in line and "FAILED:" in line:
                                pair_part, failure_part = line.split("=", 1)
                                if "→" in pair_part and failure_part.startswith("FAILED:"):
                                    error_msg = failure_part[7:]
                                    pair_key = pair_part.replace("→", " → ")
                                    pair_results[pair_key] = f"FAILED - {error_msg}"
                                    reported_failures.add(pair_key)
                                    failed_pairs += 1
                                    node_status[source_node]["failed_targets"] += 1

                        for target_node in self.node_list:
                            if target_node != source_node:
                                pair_key = f"{source_node} → {target_node}"
                                if pair_key not in reported_failures:
                                    pair_results[pair_key] = "SUCCESS"
                                    successful_pairs += 1
                                    node_status[source_node]["successful_targets"] += 1

                    except Exception as e:
                        log.error(f"Failed to read SSH test results from {source_node}: {e}")
                        for target_node in self.node_list:
                            if target_node != source_node:
                                pair_key = f"{source_node} → {target_node}"
                                pair_results[pair_key] = f"SCRIPT_ERROR: {str(e)}"
                                failed_pairs += 1
                                node_status[source_node]["failed_targets"] += 1

                results.update(
                    {
                        "successful_pairs": successful_pairs,
                        "failed_pairs": failed_pairs,
                        "pair_results": pair_results,
                        "node_status": node_status,
                    }
                )

                success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0
                log.info(
                    f"SSH mesh connectivity: {successful_pairs}/{total_pairs} pairs successful ({success_rate:.1f}%)"
                )

                if failed_pairs > 0:
                    log.warning(f"SSH connectivity issues detected: {failed_pairs} failed connections")

                    failure_examples = [(k, v) for k, v in pair_results.items() if "SUCCESS" not in v]
                    for pair, error in failure_examples[:5]:
                        log.warning(f"  {pair}: {error}")

                    if len(failure_examples) > 5:
                        log.warning(f"  ... and {len(failure_examples) - 5} more failures")

                return results

        except Exception as e:
            log.error(f"SSH full mesh connectivity test failed: {e}")
            return {
                "total_pairs": total_pairs,
                "successful_pairs": 0,
                "failed_pairs": total_pairs,
                "pair_results": {},
                "node_status": {},
                "error": str(e),
            }
