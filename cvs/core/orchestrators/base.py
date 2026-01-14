'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from abc import ABC, abstractmethod


class Orchestrator(ABC):
    """
    Abstract base class for orchestrator backends.

    Orchestrators handle distributed test execution across cluster nodes.
    Different backends (PSSH, Slurm, K8s) implement this interface to provide
    pluggable execution strategies while maintaining separation from host management.
    """

    def __init__(self, log, config, stop_on_errors=False):
        """
        Initialize orchestrator with configuration.

        Args:
            log: Logger instance
            config: Configuration dictionary containing orchestrator settings
            stop_on_errors: Whether to stop execution on first error
        """
        self.log = log
        self.config = config
        self.stop_on_errors = stop_on_errors

    @abstractmethod
    def exec(self, cmd, hosts=None, timeout=None):
        """
        Execute command across hosts using the orchestrator's execution strategy.

        This is the primary execution method that handles container vs baremetal
        execution transparently based on orchestrator configuration.

        Args:
            cmd: Command to execute
            hosts: Target hosts (if None, uses all hosts)
            timeout: Command timeout in seconds

        Returns:
            Dictionary mapping hosts to execution results
        """
        pass

    @abstractmethod
    def exec_on_head(self, cmd, timeout=None):
        """
        Execute command on head node only.

        Args:
            cmd: Command to execute
            timeout: Command timeout in seconds

        """
        pass

    @abstractmethod
    def setup_env(self, hosts, env_script=None):
        """
        Set up execution environment on target hosts.

        Args:
            hosts: List of host addresses
            env_script: Optional script to source for environment setup

        Returns:
            True if setup succeeded, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self, hosts):
        """
        Clean up resources after test execution.

        Args:
            hosts: List of host addresses

        Returns:
            True if cleanup succeeded, False otherwise
        """
        pass

    def distribute_using_mpi(self, rank_cmd, mpi_hosts, ranks_per_host, env_vars, mpi_install_dir, mpi_extra_args=None):
        """
        Distribute MPI job across hosts using the provided arguments.

        This method provides MPI-based distribution for orchestrators that support it.
        Default implementation raises NotImplementedError.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments

        Returns:
            Execution results

        Raises:
            NotImplementedError: If MPI distribution is not supported
        """
        raise NotImplementedError(f"MPI distribution not supported by {self.__class__.__name__}")
