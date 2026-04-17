"""
ScriptLet: Efficient script distribution and execution for cluster operations.

This module provides the ScriptLet class for managing script lifecycle across
cluster nodes, optimizing SSH operations by batching script distribution and
execution.
"""

import os
import tempfile
import logging

log = logging.getLogger(__name__)


class ScriptLet:
    """
    A utility class for efficient script distribution and execution across cluster nodes.

    Key features:
    - Batch script copying to minimize SSH overhead
    - Parallel script execution across nodes
    - Centralized script lifecycle management
    - Optional debug mode for script preservation
    - Minimal output collection for performance

    Responsibilities:
    1. Generate and manage script files locally
    2. Copy scripts to remote nodes efficiently
    3. Execute scripts and collect results in batch
    4. Handle cleanup of temporary files
    5. Support debugging through script preservation
    """

    def __init__(
        self,
        phdl,
        debug=False,
        cleanup_on_exit=True,
        temp_dir="/tmp/preflight",
        cleanup_on_init=False,
        preserve_temp_dir_on_exit=False,
    ):
        """
        Initialize ScriptLet with a parallel SSH handle.

        Args:
            phdl: Parallel SSH handle with copy_file_list (or copy_script_list) and exec_cmd_list methods
            debug: If True, preserve scripts for debugging (don't auto-cleanup)
            cleanup_on_exit: Whether to cleanup temp files on destruction
            temp_dir: Remote directory for storing scripts and logs
            cleanup_on_init: Whether to cleanup temp directory on initialization
            preserve_temp_dir_on_exit: If True, exit cleanup removes only ScriptLet ``.sh`` files, not
                ``rm -rf`` of ``temp_dir`` (keeps co-located logs and other artifacts in that directory).
        """
        self.phdl = phdl
        self.debug = debug
        self.cleanup_on_exit = cleanup_on_exit and not debug
        self.temp_dir = temp_dir
        self.preserve_temp_dir_on_exit = bool(preserve_temp_dir_on_exit)
        self.local_scripts = {}  # {script_id: local_path}
        self.remote_scripts = {}  # {script_id: remote_path}

        # Clean up temp directory for fresh test environment if requested
        # Directory will be recreated when needed by _ensure_temp_directory()
        if cleanup_on_init:
            self._cleanup_temp_directory(force=True)

        log.info(f"ScriptLet initialized (debug={debug}, cleanup_on_exit={self.cleanup_on_exit}, temp_dir={temp_dir})")

    def create_script(self, script_id, script_content, remote_path=None):
        """
        Create a local script file with given content.

        Args:
            script_id: Unique identifier for this script
            script_content: Shell script content
            remote_path: Target path on remote nodes (auto-generated if None)

        Returns:
            str: Local path of created script file

        Raises:
            ValueError: If script_id already exists
        """
        if script_id in self.local_scripts:
            raise ValueError(f"Script ID '{script_id}' already exists")

        if remote_path is None:
            remote_path = f"{self.temp_dir}/scriptlet_{script_id}.sh"

        # Create temporary local script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, prefix=f'scriptlet_{script_id}_') as f:
            f.write(script_content)
            local_path = f.name

        # Make script executable
        os.chmod(local_path, 0o755)

        self.local_scripts[script_id] = local_path
        self.remote_scripts[script_id] = remote_path

        log.debug(f"Created script '{script_id}': {local_path} -> {remote_path}")
        return local_path

    def copy_script(self, script_id, target_nodes=None):
        """
        Copy a single script to specified nodes or all reachable nodes.

        Args:
            script_id: ID of script to copy
            target_nodes: List of nodes to copy to (all reachable nodes if None)

        Returns:
            Dict: {node: "SUCCESS/FAILED - details"}

        Raises:
            ValueError: If script_id doesn't exist
        """
        if script_id not in self.local_scripts:
            raise ValueError(f"Script '{script_id}' not found. Create it first with create_script()")

        local_path = self.local_scripts[script_id]
        remote_path = self.remote_scripts[script_id]

        if target_nodes is None:
            target_nodes = self.phdl.reachable_hosts

        node_path_map = {node: (local_path, remote_path) for node in target_nodes}

        log.info(f"Copying script '{script_id}' to {len(target_nodes)} nodes")
        results = self.phdl.copy_file_list(node_path_map)

        # Log copy results
        successful = [r for r in results.values() if "SUCCESS" in r]
        failed = [r for r in results.values() if "FAILED" in r]
        log.info(f"Script copy completed: {len(successful)} successful, {len(failed)} failed")

        if failed and log.isEnabledFor(logging.WARNING):
            for failure in failed[:3]:  # Log first 3 failures
                log.warning(f"Copy failed: {failure}")

        return results

    def copy_script_list(self, script_mapping):
        """
        Copy different scripts to different nodes using phdl native parallel SCP (libssh2).

        Args:
            script_mapping: {node: script_id} mapping

        Returns:
            Dict: {node: "node: SUCCESS/FAILED - details"}

        Raises:
            ValueError: If any script_id doesn't exist
        """
        missing_scripts = [sid for sid in script_mapping.values() if sid not in self.local_scripts]
        if missing_scripts:
            raise ValueError(f"Scripts not found: {missing_scripts}")

        node_path_map = {
            node: (self.local_scripts[sid], self.remote_scripts[sid]) for node, sid in script_mapping.items()
        }

        self._ensure_temp_directory(list(script_mapping.keys()))

        log.info(f"Copying {len(set(script_mapping.values()))} different scripts to {len(script_mapping)} nodes")

        results = self.phdl.copy_file_list(node_path_map)

        successful = [r for r in results.values() if "SUCCESS" in r]
        failed = [r for r in results.values() if "FAILED" in r]
        log.info(f"Script copy completed: {len(successful)} successful, {len(failed)} failed")

        if failed:
            for failure in list(failed)[:3]:
                log.warning(f"Copy failed: {failure}")

        return results

    def run_parallel_group(self, script_mapping, timeout=30, cleanup_after_run=None):
        """
        Execute different scripts on different nodes in parallel using a single phdl.exec_cmd_list call.

        This is the efficient way to run scripts when each node needs to execute a different script,
        avoiding the "Script not targeted for this node" messages and enabling true parallelization.

        Args:
            script_mapping: {node: script_id} mapping
            timeout: Command timeout in seconds
            cleanup_after_run: Whether to cleanup scripts after execution

        Returns:
            Dict: {node: execution_output}

        Raises:
            ValueError: If any script_id doesn't exist
        """
        # Validate all script IDs exist
        missing_scripts = [sid for sid in script_mapping.values() if sid not in self.local_scripts]
        if missing_scripts:
            raise ValueError(f"Scripts not found: {missing_scripts}")

        # Build command list in the same order as phdl.reachable_hosts to avoid KeyError
        cmd_list = []
        executed_nodes = []

        for node in self.phdl.reachable_hosts:
            if node in script_mapping:
                script_id = script_mapping[node]
                remote_script_path = self.remote_scripts[script_id]
                cmd_list.append(f"chmod +x {remote_script_path} && {remote_script_path}")
            else:
                # Node is reachable but no script for it - use dummy command
                cmd_list.append("echo 'No script for this node'")
            executed_nodes.append(node)

        log.info(
            f"Executing {len([c for c in cmd_list if 'chmod' in c])} scripts across {len(executed_nodes)} reachable nodes"
        )

        # Execute all scripts in parallel using single phdl call
        raw_results = self.phdl.exec_cmd_list(cmd_list, timeout=timeout)

        # Convert results back to node-keyed dictionary, only include nodes that had actual scripts
        results = {}
        for i, node in enumerate(executed_nodes):
            if node in script_mapping and node in raw_results:
                results[node] = raw_results[node]

        # Handle cleanup if requested
        cleanup_now = cleanup_after_run if cleanup_after_run is not None else (not self.debug)
        if cleanup_now:
            for script_id in set(script_mapping.values()):
                self.cleanup(script_id)

        return results

    def run(self, script_id, target_nodes=None, timeout=30, cleanup_after_run=None):
        """
        Execute a script on specified nodes and return results.

        Args:
            script_id: ID of script to execute
            target_nodes: List of nodes to run on (all reachable nodes if None)
            timeout: Execution timeout in seconds
            cleanup_after_run: Whether to cleanup script after execution
                             (defaults to not self.debug)

        Returns:
            Dict: {node: script_output}

        Raises:
            ValueError: If script_id doesn't exist
        """
        if script_id not in self.remote_scripts:
            raise ValueError(f"Script '{script_id}' not found or not copied to remote nodes")

        remote_path = self.remote_scripts[script_id]

        if target_nodes is None:
            target_nodes = self.phdl.reachable_hosts

        if cleanup_after_run is None:
            cleanup_after_run = not self.debug

        # Build command list for execution
        cmd_list = []
        for node in self.phdl.host_list:
            if node in target_nodes:
                if cleanup_after_run:
                    # Execute and cleanup in one command
                    cmd = f"chmod +x {remote_path} && {remote_path} && rm -f {remote_path} 2>/dev/null || true"
                else:
                    # Execute only (preserve for debugging)
                    cmd = f"chmod +x {remote_path} && {remote_path}"
            else:
                cmd = "echo 'Script not targeted for this node'"
            cmd_list.append(cmd)

        log.info(f"Executing script '{script_id}' on {len(target_nodes)} nodes (timeout={timeout}s)")

        # Execute and return results
        results = self.phdl.exec_cmd_list(cmd_list, timeout=timeout)

        # Filter results to only target nodes
        filtered_results = {node: results.get(node, 'NO_OUTPUT') for node in target_nodes}

        # Log execution summary
        successful_runs = len([r for r in filtered_results.values() if r != 'NO_OUTPUT'])
        log.info(f"Script execution completed: {successful_runs}/{len(target_nodes)} nodes responded")

        if cleanup_after_run:
            log.debug(f"Remote cleanup completed for script '{script_id}'")
        elif self.debug:
            log.debug(f"Script '{script_id}' preserved on remote nodes for debugging: {remote_path}")

        return filtered_results

    def cleanup(self, script_id=None):
        """
        Clean up local and remote script files.

        Args:
            script_id: Specific script to cleanup (all scripts if None)
        """
        if self.debug:
            log.debug("Cleanup skipped due to debug mode")
            return

        if script_id:
            # Cleanup specific script
            self._cleanup_single_script(script_id)
        else:
            # Cleanup all scripts and optionally the entire temp directory
            script_ids = list(self.local_scripts.keys())
            for sid in script_ids:
                self._cleanup_single_script(sid)
            if not self.preserve_temp_dir_on_exit:
                self._cleanup_temp_directory()

        log.debug(f"Cleanup completed for {'all scripts' if script_id is None else f'script {script_id}'}")

    def _cleanup_single_script(self, script_id):
        """Clean up a single script's local and remote files."""
        # Cleanup local file
        if script_id in self.local_scripts:
            try:
                os.unlink(self.local_scripts[script_id])
            except OSError as e:
                log.warning(f"Failed to cleanup local script '{script_id}': {e}")
            del self.local_scripts[script_id]

        # Cleanup remote files
        if script_id in self.remote_scripts:
            remote_path = self.remote_scripts[script_id]
            cleanup_cmd = f"rm -f {remote_path} 2>/dev/null || true"
            try:
                self.phdl.exec(cleanup_cmd)
            except Exception as e:
                log.warning(f"Failed to cleanup remote script '{script_id}': {e}")
            del self.remote_scripts[script_id]

    def _cleanup_temp_directory(self, force=False):
        """Clean up the entire temp directory on remote nodes."""
        if self.debug and not force:
            log.debug(f"Debug mode enabled - preserving temp directory '{self.temp_dir}' for inspection")
            return

        cleanup_cmd = f"rm -rf {self.temp_dir} 2>/dev/null || true"
        try:
            self.phdl.exec(cleanup_cmd)
            log.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            log.warning(f"Failed to cleanup temp directory '{self.temp_dir}': {e}")

    def _ensure_temp_directory(self, target_nodes):
        """Ensure temp directory exists on target nodes."""
        if not target_nodes:
            return

        mkdir_cmd = f"mkdir -p {self.temp_dir}"
        try:
            # Create directory on all target nodes using simple exec
            self.phdl.exec(mkdir_cmd, timeout=10)
            log.debug(f"Ensured temp directory exists: {self.temp_dir}")
        except Exception as e:
            log.warning(f"Failed to create temp directory '{self.temp_dir}': {e}")

    def list_scripts(self):
        """
        List all managed scripts and their paths.

        Returns:
            Dict: {script_id: {'local': local_path, 'remote': remote_path}}
        """
        scripts_info = {}
        for script_id in self.local_scripts:
            scripts_info[script_id] = {
                'local': self.local_scripts[script_id],
                'remote': self.remote_scripts.get(script_id, 'Not set'),
            }
        return scripts_info

    def __del__(self):
        """Cleanup on destruction if enabled."""
        if self.cleanup_on_exit:
            try:
                self.cleanup()
            except Exception:
                # Don't raise exceptions in destructor
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if not self.debug:
            self.cleanup()
