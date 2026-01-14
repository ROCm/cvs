'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrator import Orchestrator
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.core.container_runner import ContainerRunner


class PsshOrchestrator(Orchestrator):
    """
    Parallel-SSH based orchestrator implementation.

    Supports containerized execution with two modes:
    1. Run MPI on Host: mpirun on host, SSH to launch docker run on nodes
    2. Direct container execution: Run test commands in containers on each node

    Integrates with CVS's cluster_dict structure from INPUT/ configs.
    """

    def __init__(self, log, cluster_dict):
        """
        Initialize PSSH orchestrator from CVS cluster_dict.

        Args:
            log: Logger instance
            cluster_dict: CVS cluster configuration with keys:
                - node_dict: Dictionary of nodes
                - username: SSH user
                - priv_key_file: Private key path
                - orchestrator: Orchestrator type (optional)
                - container: Container config (optional)
        """
        super().__init__(log, cluster_dict)

        # Extract hosts from CVS node_dict structure
        self.hosts = list(cluster_dict['node_dict'].keys())
        self.user = cluster_dict['username']
        self.pkey = cluster_dict['priv_key_file']
        self.password = cluster_dict.get('password')  # Optional

        # Initialize ParallelSSH
        self.pssh = Pssh(
            log,
            self.hosts,
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            host_key_check=False,
            stop_on_errors=False,
        )

        # Container support
        self.container_config = cluster_dict.get('container', {})
        self.container_enabled = self.container_config.get('enabled', False)

        if self.container_enabled:
            runtime = self.container_config.get('runtime', 'docker')
            self.container_runner = ContainerRunner(log, runtime=runtime)
            self.log.info(f"Container support enabled with runtime: {runtime}")
        else:
            self.container_runner = None
            self.log.info("Container support disabled")

    def run_test(
        self,
        cmd,
        hosts,
        env=None,
        timeout=None,
        container_config=None,
    ):
        """
        Execute test command across hosts.

        Args:
            cmd: Command to execute
            hosts: List of target hosts
            env: Environment variables
            timeout: Command timeout
            container_config: Test-specific container config (image, volumes, etc.)

        Returns:
            Dictionary mapping hosts to execution results
        """
        if self.container_enabled and container_config:
            return self._run_containerized(cmd, hosts, env, timeout, container_config)
        else:
            return self._run_native(cmd, hosts, env, timeout)

    def _run_native(self, cmd, hosts, env=None, timeout=None):
        """Run command natively via SSH (no containers)."""
        self.log.info(f"Running command on {len(hosts)} hosts")

        # Build command with environment
        if env:
            env_str = ' '.join([f"{k}='{v}'" for k, v in env.items()])
            full_cmd = f"{env_str} {cmd}"
        else:
            full_cmd = cmd

        return self.pssh.exec(full_cmd, timeout=timeout)

    def _run_containerized(
        self,
        cmd,
        hosts,
        env=None,
        timeout=None,
        container_config=None,
    ):
        """
        Run command in containers.

        Supports "Run MPI on Host" mode: mpirun on local host, SSH to nodes to launch containers.
        """
        image = container_config.get('image')
        volumes = container_config.get('volumes', [])
        gpu_passthrough = container_config.get('gpu_passthrough', True)
        container_env = container_config.get('env', {})

        if not image:
            raise ValueError("Container image must be specified in container_config")

        # Merge environment variables
        merged_env = {**(env or {}), **container_env}

        # Build container run command
        # Use host networking for MPI communication
        container_cmd = self.container_runner.build_run_command(
            image=image,
            cmd=cmd,
            env=merged_env,
            volumes=volumes,
            gpu_passthrough=gpu_passthrough,
            network_mode='host',
            user="$(id -u):$(id -g)",  # Match host user for SSH key access
        )

        self.log.info(f"Running containerized command on {len(hosts)} hosts")
        self.log.debug(f"Container command: {container_cmd}")

        return self.pssh.exec(container_cmd, timeout=timeout)

    def run_mpi_on_host(
        self,
        mpi_cmd,
        hosts,
        container_config=None,
        ranks_per_host=1,
        timeout=None,
    ):
        """
        Run MPI on host, launching containers on remote nodes.

        This is the "Run MPI on Host" mode from DESIGN.md.

        Args:
            mpi_cmd: MPI command to run (e.g., './all_reduce_perf -b 16M -e 8G')
            hosts: List of target hosts
            container_config: Container configuration
            ranks_per_host: MPI ranks per host
            timeout: Timeout in seconds

        Returns:
            Execution results
        """
        if not container_config:
            raise ValueError("container_config required for containerized MPI")

        image = container_config.get('image')
        volumes = container_config.get('volumes', [])
        gpu_passthrough = container_config.get('gpu_passthrough', True)
        container_env = container_config.get('env', {})

        # Build container run command
        container_run_cmd = self.container_runner.build_run_command(
            image=image,
            cmd=mpi_cmd,
            env=container_env,
            volumes=volumes,
            gpu_passthrough=gpu_passthrough,
            network_mode='host',
        )

        # Build MPI hostlist
        total_ranks = len(hosts) * ranks_per_host
        host_spec = ','.join([f"{host}:{ranks_per_host}" for host in hosts])

        # Construct mpirun command that launches containers
        full_mpi_cmd = f"mpirun -np {total_ranks} -H {host_spec} {container_run_cmd}"

        self.log.info(f"Launching MPI job: {total_ranks} ranks across {len(hosts)} hosts")
        self.log.debug(f"MPI command: {full_mpi_cmd}")

        # Execute on head node (first host)
        head_host = hosts[0]
        result = self.pssh.exec_on_hosts([head_host], full_mpi_cmd, timeout=timeout)

        return result

    def setup_env(self, hosts, env_script=None):
        """Set up environment on hosts."""
        if not env_script:
            self.log.info("No environment script specified, skipping setup")
            return True

        self.log.info(f"Setting up environment on {len(hosts)} hosts")
        result = self.pssh.exec(f"bash {env_script}", timeout=60)

        # Check if all hosts succeeded
        success = all(output.get('exit_code') == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output.get('exit_code') != 0]
            self.log.error(f"Environment setup failed on hosts: {failed}")

        return success

    def cleanup(self, hosts):
        """Clean up resources after test execution."""
        self.log.info(f"Cleaning up on {len(hosts)} hosts")

        if self.container_enabled and self.container_runner:
            # Clean up containers via SSH on each host
            cleanup_cmd = "docker container prune -f 2>/dev/null || true"
            result = self.pssh.exec(cleanup_cmd, timeout=30, print_console=False)

            success = all(output.get('exit_code') == 0 for output in result.values())
            if not success:
                self.log.warning("Container cleanup had issues on some hosts")

            return success

        return True

    def get_reachable_hosts(self):
        """Get list of currently reachable hosts."""
        return self.pssh.reachable_hosts
