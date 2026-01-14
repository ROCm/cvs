'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''


class DockerRuntime:
    """Docker container runtime implementation."""

    def __init__(self, log, orchestrator):
        self.log = log
        self.orchestrator = orchestrator  # Reference to orchestrator for SSH execution

    def check_image_exists(self, image_name):
        """Check if the Docker image exists on all nodes."""
        cmd = f"sudo docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' | grep -q '^{image_name}$'"
        result = self.orchestrator.all.exec(cmd, timeout=30, detailed=True)
        # If grep succeeds (exit 0), image exists; if not, not found
        return all(res.get('exit_code') == 0 for res in result.values())

    def setup_containers(
        self,
        container_config,
        container_name,
        volumes=None,
        devices=None,
        capabilities=None,
        security_opts=None,
        environment=None,
        groups=None,
        ulimits=None,
        device_expansion=None,
    ):
        """Set up long-running Docker containers on all nodes.

        Args:
            container_config: Container configuration dictionary
            container_name: Name to assign to the containers
            volumes: Optional list of volume mounts (overrides config)
            devices: Optional list of device passthroughs (overrides config)
            capabilities: Optional list of Linux capabilities (overrides config)
            security_opts: Optional list of security options (overrides config)
            environment: Optional dict of environment variables (overrides config)
            groups: Optional list of groups to add (overrides config)
            ulimits: Optional list of ulimits (overrides config)
            device_expansion: Optional shell expansion string for dynamic device discovery
        """
        if not container_config or not container_config.get('image'):
            self.log.warning("No container config or image specified, skipping container start")
            return False

        launch = container_config.get('launch', False)
        if not launch:
            self.log.info("Container launch disabled, assuming containers are already running")
            return True

        image = container_config['image']

        # Use provided parameters or fall back to config
        volumes = volumes if volumes is not None else container_config.get('volumes', [])
        gpu_passthrough = container_config.get('gpu_passthrough', True)
        container_env = environment if environment is not None else container_config.get('env', {})

        # Build docker run command with additional args
        runtime_config = container_config.get('runtime', {})
        runtime_args_config = runtime_config.get('args', {})

        # Override runtime args with provided parameters
        if devices is not None:
            runtime_args_config = dict(runtime_args_config)
            runtime_args_config['devices'] = devices
        if capabilities is not None:
            runtime_args_config = dict(runtime_args_config)
            runtime_args_config['cap_add'] = capabilities
        if security_opts is not None:
            runtime_args_config = dict(runtime_args_config)
            runtime_args_config['security_opt'] = security_opts
        if groups is not None:
            runtime_args_config = dict(runtime_args_config)
            runtime_args_config['group_add'] = groups
        if ulimits is not None:
            runtime_args_config = dict(runtime_args_config)
            runtime_args_config['ulimit'] = ulimits

        additional_args = self._build_runtime_args(runtime_args_config)

        # Basic args
        vol_args = ' '.join([f'-v {v}' for v in volumes])
        env_args = ' '.join([f'-e {k}={v}' for k, v in container_env.items()])
        gpu_args = '--gpus all' if gpu_passthrough else ''
        network_args = '--network host' if not runtime_args_config.get('network') else ''

        # Combine all arguments
        all_args = [gpu_args, network_args, vol_args, env_args] + additional_args
        if device_expansion:
            all_args.append(device_expansion)
        all_args_str = ' '.join(arg for arg in all_args if arg)

        # Load image from tar if specified
        if 'image_tar' in container_config:
            if not self.check_image_exists(container_config['image']):
                self.log.info(f"Loading image from tar: {container_config['image_tar']}")
                load_result = self.load_image(container_config['image_tar'], timeout=300)
                failed_load = [host for host, res in load_result.items() if res.get('exit_code') != 0]
                if failed_load:
                    self.log.error(f"Failed to load image tar on hosts: {failed_load}")
                    return False
            else:
                self.log.info(f"Image {container_config['image']} already exists, skipping tar load")

        cmd = f"sudo docker run -d --name {container_name} {all_args_str} {image} sleep infinity"

        self.log.info(f"Starting long-running containers on {len(self.orchestrator.hosts)} nodes: {container_name}")
        self.log.debug(f"Container start command: {cmd}")

        # Remove any existing container with the same name
        remove_cmd = f"sudo docker rm -f {container_name} || true"
        self.orchestrator.all.exec(remove_cmd, timeout=30, print_console=False)

        result = self.orchestrator.all.exec(cmd, timeout=60, detailed=True)

        # Check if all hosts started successfully
        success = all(output['exit_code'] == 0 for output in result.values())

        if not success:
            failed = [host for host, output in result.items() if output['exit_code'] != 0]
            self.log.error(f"Container startup failed on hosts: {failed}")
            # Clean up partial starts
            self.teardown_containers(container_name)
            return False

        self.log.info(f"Containers started successfully: {container_name}")
        return True

    def teardown_containers(self, container_name):
        """Stop and remove Docker containers on all nodes."""
        if not container_name:
            self.log.info("No container to stop")
            return True

        self.log.info(f"Stopping containers: {container_name}")

        # Force remove container (stops if running)
        cmd = f"sudo docker rm -f {container_name} 2>/dev/null || true"
        result = self.orchestrator.all.exec(cmd, timeout=30, print_console=False, detailed=True)

        success = all(output['exit_code'] == 0 for output in result.values())
        if not success:
            self.log.warning("Container stop had issues on some hosts")

        return success

    def exec(self, container_name, cmd, hosts=None, timeout=None, detailed=False):
        """Execute command in running Docker containers."""
        # Use docker exec to run command in existing container
        exec_cmd = f"sudo docker exec {container_name} {cmd}"
        if hosts:
            # Execute on specific hosts
            results = {}
            for host in hosts:
                results.update(self.orchestrator.all.exec_on_host(exec_cmd, host, timeout=timeout, detailed=detailed))
            return results
        else:
            return self.orchestrator.all.exec(exec_cmd, timeout=timeout, detailed=detailed)

    def exec_on_head(self, container_name, cmd, timeout=None):
        """Execute command directly on head node (container)."""
        exec_cmd = f"sudo docker exec {container_name} {cmd}"
        return self.orchestrator.head.exec(exec_cmd, timeout=timeout)

    @staticmethod
    def _build_runtime_args(runtime_args_config):
        """Build container runtime arguments from configuration."""
        args = []

        if not runtime_args_config:
            return args

        # Volumes
        volumes = runtime_args_config.get('volumes', [])
        for vol in volumes:
            args.extend(['-v', vol])

        # Devices
        devices = runtime_args_config.get('devices', [])
        for dev in devices:
            args.extend(['--device', dev])

        # Environment variables
        env_vars = runtime_args_config.get('env', {})
        for key, value in env_vars:
            args.extend(['-e', f'{key}={value}'])

        # Capabilities
        cap_add = runtime_args_config.get('cap_add', [])
        for cap in cap_add:
            args.extend(['--cap-add', cap])

        # Security options
        security_opt = runtime_args_config.get('security_opt', [])
        for opt in security_opt:
            args.extend(['--security-opt', opt])

        # Group add
        group_add = runtime_args_config.get('group_add', [])
        for group in group_add:
            args.extend(['--group-add', group])

        # Network
        network = runtime_args_config.get('network')
        if network:
            args.extend(['--network', network])

        # IPC
        ipc = runtime_args_config.get('ipc')
        if ipc:
            args.extend(['--ipc', ipc])

        # Ulimit
        ulimit = runtime_args_config.get('ulimit', [])
        for ul in ulimit:
            args.extend(['--ulimit', ul])

        # Privileged
        if runtime_args_config.get('privileged', False):
            args.append('--privileged')

        return args

    def load_image(self, tar_path, timeout=None):
        """Load container image from tar file on all hosts."""
        cmd = f"sudo docker load < {tar_path}"
        timeout = timeout or 600  # Default 10 minutes

        # Load on all hosts for image distribution
        result = self.orchestrator.all.exec(cmd, timeout=timeout, detailed=True)
        return result
