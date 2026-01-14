'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.pssh import PsshOrchestrator
from cvs.core.runners.container import ContainerRunner
import time


class DockerOrchestrator(PsshOrchestrator):
    """
    Docker-based orchestrator that extends PsshOrchestrator for containerized execution.

    Uses SSH transport (via PsshOrchestrator) but executes commands in Docker containers
    on cluster nodes. Supports long-running containers for efficient command execution.
    """

    def __init__(self, log, cluster_dict, stop_on_errors=False):
        """
        Initialize Docker orchestrator.

        Args:
            log: Logger instance
            cluster_dict: CVS cluster configuration
            stop_on_errors: Whether to stop execution on first error
        """
        super().__init__(log, cluster_dict, stop_on_errors)

        # Container-specific initialization
        self.container_config = cluster_dict.get('container', {})
        if not self.container_config:
            raise ValueError("DockerOrchestrator requires 'container' config in cluster_dict")

        self.container_enabled = True
        self.container_id = None  # Track running container ID

        runtime = self.container_config.get('runtime', 'docker')
        self.container_runner = ContainerRunner(log, runtime=runtime)
        self.log.info(f"DockerOrchestrator initialized with runtime: {runtime}")

    def start_container(self, container_config=None):
        """
        Start long-running containers on all nodes for efficient command execution.

        Instead of starting/stopping container for each command (slow), we start
        a single long-running container per node and use 'docker exec' for commands.

        Args:
            container_config: Container configuration (image, volumes, env, etc.)

        Returns:
            bool: True if containers started successfully on all nodes
        """
        if not container_config:
            container_config = self.container_config

        if not container_config or not container_config.get('image'):
            self.log.warning("No container config or image specified, skipping container start")
            return False

        image = container_config['image']
        volumes = container_config.get('volumes', [])
        gpu_passthrough = container_config.get('gpu_passthrough', True)
        container_env = container_config.get('env', {})

        # Generate unique container name
        self.container_id = f"cvs_test_{int(time.time())}"

        # Build docker run command for long-running container
        # Use 'sleep infinity' to keep container alive
        vol_args = ' '.join([f'-v {v}' for v in volumes])
        env_args = ' '.join([f'-e {k}={v}' for k, v in container_env.items()])
        gpu_args = '--gpus all' if gpu_passthrough else ''

        cmd = f"docker run -d --name {self.container_id} {gpu_args} --network host {vol_args} {env_args} {image} sleep infinity"

        self.log.info(f"Starting long-running containers on {len(self.hosts)} nodes: {self.container_id}")
        self.log.debug(f"Container start command: {cmd}")

        result = self.all.exec(cmd, timeout=60)

        # Check if all hosts started successfully
        success = all(output.get('exit_code') == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output.get('exit_code') != 0]
            self.log.error(f"Container startup failed on hosts: {failed}")
            # Clean up partial starts
            self.stop_container()
            self.container_id = None
            return False

        self.log.info(f"Containers started successfully: {self.container_id}")
        return True

    def exec(self, cmd, hosts=None, timeout=None):
        """
        Execute command in running containers.

        Args:
            cmd: Command to execute inside container
            hosts: Target hosts (if None, uses all hosts)
            timeout: Command timeout

        Returns:
            Dictionary mapping hosts to execution results

        Raises:
            RuntimeError: If no containers are currently running
        """
        if not self.container_id:
            raise RuntimeError("No containers running. Call start_container() first.")

        # Use docker exec to run command in existing container
        exec_cmd = f"docker exec {self.container_id} {cmd}"
        return self.all.exec(exec_cmd, timeout=timeout)

    def exec_on_head(self, cmd, timeout=None):
        """
        Execute command in container on head node only.

        Args:
            cmd: Command to execute inside container
            timeout: Command timeout

        Returns:
            Dictionary mapping head node to execution result

        Raises:
            RuntimeError: If no containers are currently running
        """
        if not self.container_id:
            raise RuntimeError("No containers running. Call start_container() first.")

        # Use docker exec to run command in container on head node only
        exec_cmd = f"docker exec {self.container_id} {cmd}"
        return self.head.exec(exec_cmd, timeout=timeout)

    def stop_container(self):
        """
        Stop and remove long-running containers on all nodes.

        Returns:
            bool: True if containers stopped successfully
        """
        if not self.container_id:
            self.log.info("No container to stop")
            return True

        self.log.info(f"Stopping containers: {self.container_id}")

        # Force remove container (stops if running)
        cmd = f"docker rm -f {self.container_id} 2>/dev/null || true"
        result = self.all.exec(cmd, timeout=30, print_console=False)

        success = all(output.get('exit_code') == 0 for output in result.values())
        if not success:
            self.log.warning("Container stop had issues on some hosts")

        self.container_id = None
        return success

    def distribute_using_mpi(self, rank_cmd, mpi_runner_args, env_vars, mpi_install_dir):
        """
        Distribute MPI job across hosts using containers.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_runner_args: List of mpirun arguments
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory

        Returns:
            Execution results
        """
        # For containerized execution, wrap the rank_cmd with docker exec
        if self.container_id:
            # If long-running container is available, use docker exec
            wrapped_rank_cmd = f"docker exec {self.container_id} {rank_cmd}"
        else:
            # Fallback to docker run (less efficient)
            container_config = self.container_config
            image = container_config.get('image', 'ubuntu:latest')
            wrapped_rank_cmd = f"docker run --rm {image} {rank_cmd}"

        # Merge env_vars with container environment if needed
        # For MPI command, env_vars are passed to mpirun
        # Container env is handled in the wrapped command if needed

        # Call parent implementation with wrapped command
        return super().distribute_using_mpi(wrapped_rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)

    def get_mpi_command(self, rank_cmd, mpi_runner_args, env_vars, mpi_install_dir):
        """
        Get the full MPI command string for containerized execution.

        Args:
            rank_cmd: The command to execute on each MPI rank
            mpi_runner_args: List of mpirun arguments
            env_vars: Dict of environment variables to set
            mpi_install_dir: Path to MPI installation directory

        Returns:
            Full MPI command string
        """
        # Wrap the rank_cmd for container execution
        if self.container_id:
            wrapped_rank_cmd = f"docker exec {self.container_id} {rank_cmd}"
        else:
            container_config = self.container_config
            image = container_config.get('image', 'ubuntu:latest')
            wrapped_rank_cmd = f"docker run --rm {image} {rank_cmd}"

        return super().get_mpi_command(wrapped_rank_cmd, mpi_runner_args, env_vars, mpi_install_dir)
