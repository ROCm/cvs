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

    The ABC also exposes a small surface for backend-blind test suites that must
    work identically on baremetal and container backends without inspecting
    orchestrator_type at source-time:

      - privileged_prefix() returns the prefix needed to run a command with
        elevated privileges in the suite's execution context. Baremetal needs
        "sudo "; container runs as root and needs "".
      - prepare() / dispose() are the lifecycle hooks for backend-specific
        setup and teardown. Suites call prepare() before any orch.exec and
        register dispose() as a finalizer.
      - host_all / host_head are the host-level Pssh handles (NOT routed
        through the container runtime), used for kernel/network/firewall
        commands that must hit the physical host even in container mode.
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

    def privileged_prefix(self):
        """
        Return the command prefix needed to run a privileged command in the
        suite's execution context.

        Default: "sudo " (correct for baremetal; the SSH user is unprivileged).
        Subclasses where the execution context already runs as root (e.g.,
        ContainerOrchestrator) override to "".

        Returns:
            str: prefix to prepend to commands that need root, e.g. "sudo ".
        """
        return "sudo "

    def prepare(self):
        """
        Backend-specific setup performed once before the first orch.exec call.

        Default: no-op. Override in subclasses that need to set up containers,
        sshd, etc.

        Implementations MUST roll back any partial setup on failure (e.g.,
        ContainerOrchestrator stops half-launched containers if setup_sshd
        fails) so that the caller's addfinalizer(dispose) hook is not
        responsible for cleaning up half-initialized state.

        Returns:
            bool: True on success, False on failure (suite should treat as fatal).
        """
        return True

    def dispose(self):
        """
        Backend-specific teardown. Counterpart to prepare().

        Default: no-op. Subclasses override to release containers, run
        cleanup() on the host-level handles, etc. Must be idempotent and safe
        to call even if prepare() was never invoked or failed mid-way.

        Returns:
            bool: True on success, False on failure (do not raise from
                  finalizers; log and return False instead).
        """
        return True

    @property
    @abstractmethod
    def host_all(self):
        """
        Pssh handle that targets all cluster nodes at the host (NOT container)
        namespace. Used for commands that must run on the physical host even
        in container mode: lsmod, dmesg, service ufw, rdma link, ibv_devinfo,
        cat /opt/rocm/.info/version, etc.

        Returns:
            Pssh: parallel SSH handle to all hosts.
        """
        pass

    @property
    @abstractmethod
    def host_head(self):
        """
        Pssh handle that targets only the head node at the host namespace.
        Counterpart to host_all for single-node host-level operations.

        Returns:
            Pssh: parallel SSH handle to the head host only.
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
