'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import subprocess


class ContainerRunner:
    """
    Handles container operations (pull, run, cleanup) for Docker and Enroot.

    Provides a unified interface for container management across different
    runtimes, integrating with orchestrators for containerized execution.
    """

    def __init__(self, log, runtime='docker'):
        """
        Initialize container runner.

        Args:
            log: Logger instance
            runtime: Container runtime ('docker' or 'enroot')
        """
        self.log = log
        self.runtime = runtime.lower()

        if self.runtime not in ['docker', 'enroot']:
            raise ValueError(f"Unsupported runtime: {runtime}. Use 'docker' or 'enroot'")

    def build_run_command(
        self,
        image,
        cmd,
        env=None,
        volumes=None,
        gpu_passthrough=True,
        network_mode='host',
        user=None,
    ):
        """
        Build container run command for the specified runtime.

        Args:
            image: Container image name/path
            cmd: Command to run inside container
            env: Environment variables
            volumes: Volume mounts (format: 'host:container' or 'host:container:ro')
            gpu_passthrough: Enable GPU passthrough
            network_mode: Network mode ('host', 'bridge', etc.)
            user: Run as specific user (UID:GID or username)

        Returns:
            Complete container run command string
        """
        if self.runtime == 'docker':
            return self._build_docker_command(image, cmd, env, volumes, gpu_passthrough, network_mode, user)
        elif self.runtime == 'enroot':
            return self._build_enroot_command(image, cmd, env, volumes, gpu_passthrough, user)

    def _build_docker_command(
        self,
        image,
        cmd,
        env=None,
        volumes=None,
        gpu_passthrough=True,
        network_mode='host',
        user=None,
    ):
        """Build Docker run command."""
        cmd_parts = ['docker', 'run', '--rm']

        # Network mode (critical for MPI communication)
        cmd_parts.append(f'--network {network_mode}')

        # GPU passthrough
        if gpu_passthrough:
            cmd_parts.append('--gpus all')

        # User
        if user:
            cmd_parts.append(f'--user {user}')

        # Environment variables
        if env:
            for key, value in env.items():
                cmd_parts.append(f"-e {key}='{value}'")

        # Volume mounts
        if volumes:
            for vol in volumes:
                cmd_parts.append(f'-v {vol}')

        # Image and command
        cmd_parts.append(image)
        cmd_parts.append(cmd)

        return ' '.join(cmd_parts)

    def _build_enroot_command(
        self,
        image,
        cmd,
        env=None,
        volumes=None,
        gpu_passthrough=True,
        user=None,
    ):
        """Build Enroot run command (for Slurm integration)."""
        # Enroot is typically used via srun --container-image
        # This provides a fallback for direct enroot execution
        cmd_parts = ['enroot', 'start']

        # Environment variables
        if env:
            for key, value in env.items():
                cmd_parts.append(f"--env {key}='{value}'")

        # Volume mounts
        if volumes:
            for vol in volumes:
                cmd_parts.append(f'--mount {vol}')

        # Image and command
        cmd_parts.append(image)
        cmd_parts.append(cmd)

        return ' '.join(cmd_parts)

    def pull_image(self, image, output_path=None):
        """
        Pull container image.

        Args:
            image: Image name (e.g., 'rocm/rccl-tests:latest')
            output_path: For Enroot, path to save .sqsh file

        Returns:
            True if pull succeeded
        """
        try:
            if self.runtime == 'docker':
                self.log.info(f"Pulling Docker image: {image}")
                result = subprocess.run(['docker', 'pull', image], capture_output=True, text=True)
                if result.returncode == 0:
                    self.log.info(f"Successfully pulled {image}")
                    return True
                else:
                    self.log.error(f"Failed to pull {image}: {result.stderr}")
                    return False

            elif self.runtime == 'enroot':
                if not output_path:
                    raise ValueError("output_path required for Enroot image pull")

                self.log.info(f"Importing Enroot image: {image} to {output_path}")
                # Convert docker:// to enroot format
                enroot_image = f"docker://{image}" if not image.startswith('docker://') else image
                result = subprocess.run(
                    ['enroot', 'import', '-o', output_path, enroot_image], capture_output=True, text=True
                )
                if result.returncode == 0:
                    self.log.info(f"Successfully imported to {output_path}")
                    return True
                else:
                    self.log.error(f"Failed to import {image}: {result.stderr}")
                    return False

        except Exception as e:
            self.log.error(f"Exception pulling image {image}: {str(e)}")
            return False

    def cleanup_containers(self, container_filter=None):
        """
        Clean up stopped containers.

        Args:
            container_filter: Optional filter for containers to clean

        Returns:
            True if cleanup succeeded
        """
        try:
            if self.runtime == 'docker':
                self.log.info("Cleaning up Docker containers")
                # Remove stopped containers
                cmd = ['docker', 'container', 'prune', '-f']
                if container_filter:
                    cmd.extend(['--filter', container_filter])

                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            elif self.runtime == 'enroot':
                self.log.info("Enroot cleanup handled via Slurm")
                return True

        except Exception as e:
            self.log.error(f"Exception during cleanup: {str(e)}")
            return False
