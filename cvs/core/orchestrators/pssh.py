'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.base import Orchestrator
from cvs.lib.parallel_ssh_lib import Pssh


class PsshOrchestrator(Orchestrator):
    """
    Parallel-SSH based orchestrator implementation.

    Supports containerized execution with two modes:
    1. Run MPI on Host: mpirun on host, SSH to launch docker run on nodes
    2. Direct container execution: Run test commands in containers on each node

    Integrates with CVS's cluster_dict structure from INPUT/ configs.
    """

    def __init__(self, log, cluster_dict, stop_on_errors=False):
        """
        Initialize PSSH orchestrator from CVS cluster_dict.

        Args:
            log: Logger instance
            cluster_dict: CVS cluster configuration with keys:
                - node_dict: Dictionary of nodes
                - head_node_dict: Optional head node specification with mgmt_ip
                - username: SSH user
                - priv_key_file: Private key path
                - orchestrator: Orchestrator type (optional)
                - container: Container config (optional)
            stop_on_errors: Whether to stop execution on first error
        """
        super().__init__(log, cluster_dict, stop_on_errors)

        # Extract hosts from CVS node_dict structure
        self.hosts = list(cluster_dict['node_dict'].keys())

        # Determine head node - use head_node_dict if available, otherwise first node
        if 'head_node_dict' in cluster_dict and cluster_dict['head_node_dict'].get('mgmt_ip'):
            self.head_node = cluster_dict['head_node_dict']['mgmt_ip']
        else:
            self.head_node = self.hosts[0]  # First node is head

        self.user = cluster_dict['username']
        self.pkey = cluster_dict['priv_key_file']
        self.password = cluster_dict.get('password')  # Optional

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
            result = self.all.exec(f"bash {env_script}", timeout=60)
        elif len(hosts) == 1 and hosts[0] == self.head_node:
            result = self.head.exec(f"bash {env_script}", timeout=60)
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
            result = pssh.exec(f"bash {env_script}", timeout=60)

        # Check if all hosts succeeded
        success = all(output.get('exit_code') == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output.get('exit_code') != 0]
            self.log.error(f"Environment setup failed on hosts: {failed}")

        return success

    def cleanup(self, hosts):
        """Clean up resources after test execution."""
        self.log.info(f"Cleaning up on {len(hosts)} hosts")
        # Basic cleanup - no specific resources to clean for SSH-only orchestrator
        return True

        return True

    def distribute_using_mpi(self, rank_cmd, mpi_runner_args, env_vars, mpi_install_dir):
        """
        Distribute MPI job across hosts using the provided arguments.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_runner_args: List of mpirun arguments (including --np, --hostfile, etc.)
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory

        Returns:
            Execution results from head node
        """
        full_mpi_cmd = self.get_mpi_command(rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)

        self.log.info("Launching MPI job")
        self.log.debug(f"MPI command: {full_mpi_cmd}")

        # Execute on head node
        result = self.head.exec(full_mpi_cmd, timeout=500)  # Default timeout

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

        # Construct full MPI command
        full_mpi_cmd = f'{mpi_install_dir}/mpirun {mpi_runner_args_str} {" ".join(env_args)} {rank_cmd}'

        return full_mpi_cmd
