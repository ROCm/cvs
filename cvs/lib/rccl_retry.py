"""
RCCL Test Retry and Cleanup Decorator.
Handles hung processes from fabric congestion and transient failures.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import logging

from .retry import RetryIfEnabled

log = logging.getLogger(__name__)


class RcclRetryIfEnabled(RetryIfEnabled):
    """
    RCCL-specific retry decorator with cleanup logic.

    Extends RetryIfEnabled to add RCCL-specific cleanup functionality
    for handling hung processes from fabric congestion in CSP environments.

    Usage:
        @RcclRetryIfEnabled()
        def rccl_cluster_test_default(..., retry_config=None):
            # Function implementation
    """

    def __init__(self):
        """Initialize the decorator with RCCL test binary names"""
        super().__init__()
        self.rccl_test_binaries = [
            'all_reduce_perf',
            'all_gather_perf',
            'reduce_scatter_perf',
            'broadcast_perf',
            'reduce_perf',
            'alltoall_perf',
            'alltoallv_perf',
            'gather_perf',
            'scatter_perf',
            'sendrecv_perf',
            'hypercube_perf'
        ]

    def _cleanup_after_failure(self, kwargs, retry_config):
        """
        Perform RCCL cleanup after failure when retry is enabled.

        Args:
            kwargs: Function kwargs
            retry_config: Retry configuration
        """
        # Extract required parameters
        phdl = kwargs.get('phdl')
        rccl_tests_dir = kwargs.get('rccl_tests_dir')
        user_name = kwargs.get('user_name')

        if all([phdl, rccl_tests_dir, user_name]):
            log.info("Performing cleanup after test failure...")
            self._cleanup_hung_processes(phdl, rccl_tests_dir, user_name)

    def _cleanup_hung_processes(self, phdl, rccl_tests_dir, user_name=None):
        """
        Kill any hung RCCL test processes and clean up temporary files.

        This is critical for CSP environments where fabric congestion can cause
        RCCL tests to hang indefinitely, blocking subsequent tests.

        Args:
            phdl: Parallel SSH handle for executing commands on all nodes
            rccl_tests_dir: Path to RCCL tests installation
            user_name: Optional user name to filter processes by ownership
        """
        log.info("Starting RCCL cleanup - checking for hung processes and temporary files...")

        cleanup_commands = []
        found_processes = []

        # Add user flag for process filtering if user specified
        user_flag = f"-u {user_name}" if user_name else ""

        # First, check what processes exist before killing them
        log.info("Checking for existing RCCL processes before cleanup...")

        # Check for RCCL test processes
        check_cmd = f"ps aux {user_flag} | grep -E '(all_reduce|all_gather|reduce_scatter|broadcast|reduce|alltoall|gather|scatter|sendrecv|hypercube)_perf' | grep -v grep | wc -l"
        try:
            result = phdl.exec(check_cmd, stop_on_errors=False)
            for node, output in result.items():
                count = output.strip()
                if count and count != "0":
                    found_processes.append(f"{count} RCCL test processes on {node}")
                    log.info(f"Found {count} RCCL test processes on {node}")
        except Exception as e:
            log.warning(f"Could not check RCCL processes: {e}")

        # Check for MPI processes
        mpi_check_cmd = f"ps aux {user_flag} | grep -E 'mpirun.*rccl|orted' | grep -v grep | wc -l"
        try:
            result = phdl.exec(mpi_check_cmd, stop_on_errors=False)
            for node, output in result.items():
                count = output.strip()
                if count and count != "0":
                    found_processes.append(f"{count} MPI processes on {node}")
                    log.info(f"Found {count} MPI processes on {node}")
        except Exception as e:
            log.warning(f"Could not check MPI processes: {e}")

        if not found_processes:
            log.info("No hung RCCL or MPI processes found - cleanup may not be necessary")
        else:
            log.warning(f"Found processes requiring cleanup: {', '.join(found_processes)}")

        # Kill RCCL test processes
        log.info("Killing RCCL test processes...")
        for binary in self.rccl_test_binaries:
            cmd = f"pkill -9 {user_flag} -f '{rccl_tests_dir}/{binary}' 2>/dev/null || true"
            cleanup_commands.append(cmd)
            log.debug(f"Will kill: {binary} processes")

        # Kill MPI processes related to RCCL
        log.info("Killing MPI processes...")
        cleanup_commands.extend([
            f"pkill -9 {user_flag} -f 'mpirun.*rccl' 2>/dev/null || true",
            f"pkill -9 {user_flag} -f 'orted' 2>/dev/null || true",  # OpenMPI daemons
        ])

        # Execute cleanup on all nodes
        killed_processes = []
        for cmd in cleanup_commands:
            try:
                result = phdl.exec(cmd, stop_on_errors=False)
                # Check if the command actually killed something by looking at exit codes
                for node, output in result.items():
                    # pkill returns 0 if it killed processes, 1 if no processes found
                    # But we use || true so it always succeeds
                    killed_processes.append(f"Executed: {cmd} on {node}")
                log.debug(f"Executed cleanup: {cmd}")
            except Exception as e:
                log.warning(f"Cleanup command failed (non-fatal): {cmd}, error: {e}")

        # Clean up temporary files
        log.info("Cleaning up temporary files...")
        cleanup_commands.extend([
            "rm -f /tmp/rccl_*.json 2>/dev/null || true",
            "rm -f /tmp/rccl_hosts_file.txt 2>/dev/null || true",
            "rm -f /tmp/ompi.* 2>/dev/null || true",  # OpenMPI session files
        ])

        # Execute file cleanup
        for cmd in cleanup_commands[-3:]:  # Last 3 commands are file cleanup
            try:
                phdl.exec(cmd, stop_on_errors=False)
                log.debug(f"Executed file cleanup: {cmd}")
            except Exception as e:
                log.warning(f"File cleanup command failed (non-fatal): {cmd}, error: {e}")

        # Summary
        if found_processes:
            log.info(f"RCCL cleanup completed - removed: {', '.join(found_processes)}")
        else:
            log.info("RCCL cleanup completed - no processes were found to clean up")

        log.info("RCCL cleanup process finished")


# Create a singleton instance for use as decorator
rccl_retry_if_enabled = RcclRetryIfEnabled()