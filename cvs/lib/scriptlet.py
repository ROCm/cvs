"""
ScriptLet: Efficient script distribution and execution for cluster operations.

This module provides the ScriptLet class for managing script lifecycle across
cluster nodes, optimizing SSH operations by batching script distribution and
execution.
"""

import os
import tempfile
import logging
import shlex

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
        scriptlet_workspace="/tmp/preflight",
        cleanup_on_init=False,
        preserve_workspace_on_exit=False,
    ):
        """
        Initialize ScriptLet with a parallel SSH handle.

        Args:
            phdl: Parallel SSH handle with upload_file_list (or copy_script_list) and exec_cmd_list methods
            debug: If True, preserve scripts for debugging (don't auto-cleanup)
            cleanup_on_exit: Whether to cleanup temp files on destruction
            scriptlet_workspace: Remote directory for storing scripts and logs
            cleanup_on_init: Whether to cleanup workspace on initialization
            preserve_workspace_on_exit: If True, exit cleanup removes only ScriptLet ``.sh`` files, not
                ``rm -rf`` of ``scriptlet_workspace`` (keeps co-located logs and other artifacts in that directory).
        """
        self.phdl = phdl
        self.debug = debug
        self.cleanup_on_exit = cleanup_on_exit and not debug
        self.workspace_dir = scriptlet_workspace
        self.cleanup_on_init = bool(cleanup_on_init)
        self.preserve_workspace_on_exit = bool(preserve_workspace_on_exit)
        self.local_scripts = {}  # {script_id: local_path}
        self.remote_scripts = {}  # {script_id: {"host": host, "path": remote_path}}

        # Clean up temp directory for fresh test environment if requested
        # Directory will be recreated when needed by _ensure_workspace_directory()
        if cleanup_on_init:
            self._cleanup_workspace_directory(force=True)

        log.info(
            f"ScriptLet initialized (debug={debug}, cleanup_on_exit={self.cleanup_on_exit}, workspace={self.workspace_dir})"
        )

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
            remote_path = f"{self.workspace_dir}/scriptlet_{script_id}.sh"

        # Create temporary local script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, prefix=f'scriptlet_{script_id}_') as f:
            f.write(script_content)
            local_path = f.name

        # Make script executable
        os.chmod(local_path, 0o755)

        self.local_scripts[script_id] = local_path
        self.remote_scripts[script_id] = {"host": None, "path": remote_path}  # host will be set when copied

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
        remote_path = self.remote_scripts[script_id]["path"]

        if target_nodes is None:
            target_nodes = self.phdl.reachable_hosts

        node_path_map = {node: (local_path, remote_path) for node in target_nodes}

        log.info(f"Copying script '{script_id}' to {len(target_nodes)} nodes")
        results = self.phdl.upload_file_list(node_path_map)

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

        # Update remote_scripts with host information
        for host, script_id in script_mapping.items():
            if script_id in self.remote_scripts:
                self.remote_scripts[script_id]["host"] = host

        node_path_map = {
            node: (self.local_scripts[sid], self.remote_scripts[sid]["path"]) for node, sid in script_mapping.items()
        }

        # When cleanup_on_init=True, the workspace was just force-cleaned and must be recreated.
        # For cleanup_on_init=False flows (e.g., subsequent phases sharing the same workspace),
        # skip the extra mkdir round-trip to reduce overhead.
        if self.cleanup_on_init:
            self._ensure_workspace_directory(list(script_mapping.keys()))

        log.info(f"Copying {len(set(script_mapping.values()))} different scripts to {len(script_mapping)} nodes")

        results = self.phdl.upload_file_list(node_path_map)

        successful = [r for r in results.values() if "SUCCESS" in r]
        failed = [r for r in results.values() if "FAILED" in r]
        log.info(f"Script copy completed: {len(successful)} successful, {len(failed)} failed")

        if failed:
            for failure in list(failed)[:3]:
                log.warning(f"Copy failed: {failure}")

        return results

    def run_parallel_group(self, script_mapping, timeout=30, cleanup_after_run=False):
        """
        Execute different scripts on different nodes in parallel using a single phdl.exec_cmd_list call.

        This is the efficient way to run scripts when each node needs to execute a different script,
        avoiding the "Script not targeted for this node" messages and enabling true parallelization.

        Args:
            script_mapping: {node: script_id} mapping
            timeout: Command timeout in seconds
            cleanup_after_run: Whether to cleanup scripts after execution (default: False,
                               let __exit__ handle cleanup for better efficiency)

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
                remote_script_path = self.remote_scripts[script_id]["path"]
                quoted_remote_script_path = shlex.quote(remote_script_path)
                cmd_list.append(f"chmod +x {quoted_remote_script_path} && {quoted_remote_script_path}")
            else:
                # Node is reachable but no script for it - use dummy command
                cmd_list.append("echo 'No script for this node'")
            executed_nodes.append(node)

        log.info(
            f"Executing {len([c for c in cmd_list if 'chmod' in c])} scripts across {len(executed_nodes)} reachable nodes"
        )

        # Execute all scripts in parallel using single phdl call
        raw_results = self.phdl.exec_cmd_list(cmd_list, timeout=timeout, print_console=False)

        # Convert results back to node-keyed dictionary, only include nodes that had actual scripts
        results = {}
        for i, node in enumerate(executed_nodes):
            if node in script_mapping and node in raw_results:
                results[node] = raw_results[node]

        # Handle cleanup if requested
        if cleanup_after_run and not self.debug:
            self.cleanup_script_list(script_mapping)

        return results

    def cleanup_script_list(self, script_mapping):
        """
        Clean up different scripts from different nodes in batch mode for efficiency.

        Args:
            script_mapping: {node: script_id} mapping (same format as copy_script_list)
        """
        if self.debug:
            log.debug("Cleanup skipped due to debug mode")
            return

        if not script_mapping:
            return

        # Clean up local files (only once per unique script)
        unique_scripts = set(script_mapping.values())
        for script_id in unique_scripts:
            if script_id in self.local_scripts:
                try:
                    os.unlink(self.local_scripts[script_id])
                    del self.local_scripts[script_id]
                except OSError as e:
                    log.warning(f"Failed to cleanup local script '{script_id}': {e}")

        # Build parallel cleanup commands for remote files
        cleanup_commands = []
        for node in self.phdl.reachable_hosts:
            if node in script_mapping:
                script_id = script_mapping[node]
                if script_id in self.remote_scripts:
                    remote_path = self.remote_scripts[script_id]["path"]
                    cleanup_commands.append(f"rm -f {shlex.quote(remote_path)} 2>/dev/null || true")
                else:
                    cleanup_commands.append("true")  # No-op for nodes without scripts
            else:
                cleanup_commands.append("true")  # No-op for nodes not in mapping

        # Execute all cleanup commands in parallel
        if cleanup_commands:
            log.info(f"Cleaning up {len(unique_scripts)} scripts from {len(script_mapping)} nodes")
            try:
                self.phdl.exec_cmd_list(cleanup_commands, print_console=False)
            except Exception as e:
                log.warning(f"Failed to cleanup remote scripts: {e}")

        # Remove scripts from remote tracking
        for script_id in unique_scripts:
            if script_id in self.remote_scripts:
                del self.remote_scripts[script_id]

    def cleanup(self):
        """
        Clean up local and remote script files.
        """
        if self.debug:
            log.debug("Cleanup skipped due to debug mode")
            return

        # Cleanup all scripts using batch method for efficiency
        if self.remote_scripts or self.local_scripts:
            # Use batch cleanup for remote scripts if any exist
            if self.remote_scripts:
                # Build script mapping from remote_scripts structure
                script_mapping = {}
                for script_id, script_info in self.remote_scripts.items():
                    if script_info["host"] is not None:
                        script_mapping[script_info["host"]] = script_id

                if script_mapping:
                    log.debug(
                        f"Using batch cleanup for {len(script_mapping)} scripts across {len(script_mapping)} hosts"
                    )
                    self.cleanup_script_list(script_mapping)

            # Clean any remaining local scripts that weren't cleaned by batch
            for script_id in list(self.local_scripts.keys()):
                if script_id in self.local_scripts:
                    try:
                        os.unlink(self.local_scripts[script_id])
                    except OSError as e:
                        log.warning(f"Failed to cleanup local script '{script_id}': {e}")
                    del self.local_scripts[script_id]

        if not self.preserve_workspace_on_exit:
            self._cleanup_workspace_directory()

        log.debug("Cleanup completed for all scripts")

    def _cleanup_workspace_directory(self, force=False):
        """Clean up the entire workspace directory on remote nodes."""
        if self.debug and not force:
            log.debug(f"Debug mode enabled - preserving workspace directory '{self.workspace_dir}' for inspection")
            return

        cleanup_cmd = f"rm -rf {shlex.quote(self.workspace_dir)} 2>/dev/null || true"
        try:
            self.phdl.exec(cleanup_cmd, print_console=False)
            log.info(f"Cleaned up workspace directory: {self.workspace_dir}")
        except Exception as e:
            log.warning(f"Failed to cleanup workspace directory '{self.workspace_dir}': {e}")

    def _ensure_workspace_directory(self, target_nodes):
        """Ensure workspace directory exists on target nodes."""
        if not target_nodes:
            return

        mkdir_cmd = f"mkdir -p {shlex.quote(self.workspace_dir)}"
        try:
            # Create directory on all target nodes using simple exec
            self.phdl.exec(mkdir_cmd, timeout=10, print_console=False)
            log.debug(f"Ensured workspace directory exists: {self.workspace_dir}")
        except Exception as e:
            log.warning(f"Failed to create workspace directory '{self.workspace_dir}': {e}")

    def list_scripts(self):
        """
        List all managed scripts and their paths.

        Returns:
            Dict: {script_id: {'local': local_path, 'remote': remote_path, 'host': host}}
        """
        scripts_info = {}

        for script_id in self.local_scripts:
            remote_info = self.remote_scripts.get(script_id, {"path": "Not set", "host": "Not assigned"})
            scripts_info[script_id] = {
                'local': self.local_scripts[script_id],
                'remote': remote_info["path"],
                'host': remote_info["host"],
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
