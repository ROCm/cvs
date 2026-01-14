'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.base import Orchestrator
from cvs.lib.parallel_ssh_lib import Pssh


class BaremetalOrchestrator(Orchestrator):
    """
    Baremetal orchestrator implementation using Parallel-SSH.

    Executes commands directly on host systems via SSH without containers.
    Provides the foundation for containerized orchestrators.

    Integrates with OrchestratorConfig for standardized configuration.
    """

    def __init__(self, log, config, stop_on_errors=False):
        """
        Initialize baremetal orchestrator from OrchestratorConfig.

        Args:
            log: Logger instance
            config: OrchestratorConfig instance
            stop_on_errors: Whether to stop execution on first error
        """
        super().__init__(log, config, stop_on_errors)

        # Set orchestrator type for runtime identification
        self.orchestrator_type = "baremetal"

        # SSH port for MPI communication (overridable by subclasses)
        self.ssh_port = 22

        # Extract hosts from OrchestratorConfig node_dict
        self.hosts = (
            list(config.node_dict.keys())
            if isinstance(config.node_dict, dict)
            else [node.get('mgmt_ip') for node in config.node_dict]
        )

        self.head_node = self.hosts[0]  # First node is head

        self.user = config.get('username')
        self.pkey = config.get('priv_key_file')
        self.password = config.get('password')  # Optional

        # Initialize TWO ParallelSSH handles like original CVS pattern:
        # head - Single head node (for mpirun, result collection, etc.)
        # all - Parallel across all nodes (for setup, cleanup, verification)
        self.head = Pssh(
            log,
            [self.head_node],
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            host_key_check=False,
            stop_on_errors=self.stop_on_errors,
        )
        self.all = Pssh(
            log,
            self.hosts,
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            host_key_check=False,
            stop_on_errors=self.stop_on_errors,
        )

    def exec(self, cmd, hosts=None, timeout=None):
        """
        Execute command across hosts via SSH (baremetal execution).

        Args:
            cmd: Command to execute
            hosts: Target hosts (if None, uses all hosts)
            timeout: Command timeout

        Returns:
            Dictionary mapping hosts to execution results
        """
        if hosts is None:
            hosts = self.hosts

        # Use appropriate handle based on target hosts
        if set(hosts) == set(self.hosts):
            return self.all.exec(cmd, timeout=timeout)
        else:
            # For arbitrary subset (including head node), create temporary handle
            pssh = Pssh(
                self.log,
                hosts,
                user=self.user,
                password=self.password,
                pkey=self.pkey,
                host_key_check=False,
                stop_on_errors=self.stop_on_errors,
            )
            return pssh.exec(cmd, timeout=timeout)

    def exec_on_head(self, cmd, timeout=None):
        """
        Execute command on head node only via SSH.

        Args:
            cmd: Command to execute
            timeout: Command timeout

        Returns:
            Dictionary mapping head node to execution result
        """
        return self.head.exec(cmd, timeout=timeout)

    def setup_env(self, hosts, env_script=None):
        """Set up environment on hosts."""
        if not env_script:
            self.log.info("No environment script specified, skipping setup")
            return True

        self.log.info(f"Setting up environment on {len(hosts)} hosts")

        # Use appropriate handle
        if set(hosts) == set(self.hosts):
            result = self.all.exec(f"bash {env_script}", timeout=60, detailed=True)
        elif len(hosts) == 1 and hosts[0] == self.head_node:
            result = self.exec_on_head(f"bash {env_script}", timeout=60, detailed=True)
        else:
            pssh = Pssh(
                self.log,
                hosts,
                user=self.user,
                password=self.password,
                pkey=self.pkey,
                host_key_check=False,
                stop_on_errors=self.stop_on_errors,
            )
            result = pssh.exec(f"bash {env_script}", timeout=60, detailed=True)

        # Check if all hosts succeeded
        success = all(output['exit_code'] == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output['exit_code'] != 0]
            self.log.error(f"Environment setup failed on hosts: {failed}")

        return success

    def cleanup(self, hosts):
        """Clean up resources after test execution."""
        self.log.info(f"Cleaning up on {len(hosts)} hosts")
        # Basic cleanup - no specific resources to clean for SSH-only orchestrator
        return True

    def build_mpi_cmd(
        self,
        rank_cmd,
        mpi_hosts,
        ranks_per_host,
        env_vars,
        mpi_install_dir,
        mpi_extra_args=None,
        no_of_global_ranks=None,
    ):
        """
        Build MPI command string for distributed execution.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Full MPI command string
        """
        # Create MPI hostfile
        host_file_params = ''
        for host in mpi_hosts:
            host_file_params += f'{host} slots={ranks_per_host}\n'

        # Create hostfile on head node
        cmd = 'sudo rm -f /tmp/mpi_hosts.txt'
        self.exec_on_head(cmd)

        cmd = f'bash -c \'echo "{host_file_params}" > /tmp/mpi_hosts.txt\''
        self.exec_on_head(cmd)

        # Build MPI runner arguments
        if no_of_global_ranks is None:
            no_of_global_ranks = len(mpi_hosts) * ranks_per_host
        mpi_runner_args = [
            '--np',
            str(no_of_global_ranks),
            '--allow-run-as-root',
            '--hostfile',
            '/tmp/mpi_hosts.txt',
        ]

        if mpi_extra_args:
            mpi_runner_args.extend(mpi_extra_args)

        full_mpi_cmd = self.get_mpi_command(rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)

        return full_mpi_cmd

    def distribute_using_mpi(
        self,
        rank_cmd,
        mpi_hosts,
        ranks_per_host,
        env_vars,
        mpi_install_dir,
        mpi_extra_args=None,
        no_of_global_ranks=None,
    ):
        """
        Distribute MPI job across hosts using the provided arguments.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_hosts: List of host IPs/names for MPI hostfile
            ranks_per_host: Number of MPI ranks per host (uniform across hosts)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory
            mpi_extra_args: List of additional mpirun arguments
            no_of_global_ranks: Total number of MPI ranks (optional, defaults to len(mpi_hosts) * ranks_per_host)

        Returns:
            Execution results from head node
        """
        full_mpi_cmd = self.build_mpi_cmd(
            rank_cmd, mpi_hosts, ranks_per_host, env_vars, mpi_install_dir, mpi_extra_args, no_of_global_ranks
        )

        self.log.info("Launching MPI job")
        self.log.debug(f"MPI command: {full_mpi_cmd}")

        # Execute on head node
        result = self.exec_on_head(full_mpi_cmd, timeout=500)  # Default timeout

        return result

    def get_mpi_command(self, rank_cmd, mpi_runner_args, env_vars, mpi_install_dir):
        """
        Get the full MPI command string without executing it.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_runner_args: List of mpirun arguments
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory

        Returns:
            Full MPI command string
        """
        # Build environment variable arguments for mpirun
        env_args = []
        for key, value in env_vars.items():
            env_args.append(f'-x {key}={value}')

        # Build MPI runner arguments string
        mpi_runner_args_str = ' '.join(mpi_runner_args)

        # Add SSH options to auto-resolve host key issues in test environments
        ssh_options = f'--mca plm_rsh_agent ssh --mca plm_rsh_args "-p {self.ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"'

        # Construct full MPI command
        full_mpi_cmd = f'{mpi_install_dir}/mpirun {ssh_options} {mpi_runner_args_str} {" ".join(env_args)} {rank_cmd}'

        return full_mpi_cmd
