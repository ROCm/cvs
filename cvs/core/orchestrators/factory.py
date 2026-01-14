'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


class OrchestratorConfig:
    """
    Configuration for orchestrator creation.

    Contains only the configuration keys required by orchestrators.

    Expected keys in cluster_config.json or <testsuite>_config.json:
    - orchestrator: Type of orchestrator ('baremetal', 'container', etc.) [optional, default: 'baremetal']
    - node_dict: List/dict of cluster nodes [required, no default]
    - username: SSH username for node access [required, no default]
    - priv_key_file: Path to SSH private key file [required, no default]
    - password: Optional SSH password (if not using key-based auth) [default: None]
    - head_node_dict: Optional head node configuration with 'mgmt_ip' key [default: {}]
    - container: Optional container configuration for docker orchestrator [default: {}]
          Example container configuration:
          ```json
          "container": {
            "enabled": true,
            "launch": false,
            "runtime": {
              "name": "docker",
              "args": {
                "volumes": ["/host/path:/container/path:ro"],
                "devices": ["/dev/kfd", "/dev/dri"],
                "env": {"GPUS": "8", "MULTINODE": "true"},
                "cap_add": ["SYS_PTRACE"],
                "security_opt": ["apparmor=unconfined", "seccomp=unconfined"],
                "group_add": ["video"],
                "network": "host",
                "ipc": "host",
                "ulimit": ["memlock=-1"],
                "privileged": true
              }
            },
            "image": "rocm/cvs:latest",
            "name": "myuser_rocm_cvs_latest"
          }
          ```
          launch: Containers are already running, test suite should not start/stop them [default: false]
    """

    def __init__(self, **kwargs):
        """
        Initialize orchestrator configuration.

        Args:
            **kwargs: Required orchestrator configuration keys
        """
        self.orchestrator = kwargs['orchestrator']
        self.node_dict = kwargs['node_dict']
        self.username = kwargs['username']
        self.priv_key_file = kwargs['priv_key_file']
        self.password = kwargs.get('password')
        self.head_node_dict = kwargs.get('head_node_dict', {})
        self.container = kwargs.get('container', {})

    def get(self, key, default=None):
        """Get configuration value with default."""
        return getattr(self, key, default)

    @classmethod
    def from_configs(cls, cluster_config, testsuite_config=None):
        """
        Create config from multiple configuration sources.

        Merges cluster_config.json and <testsuite>_config.json configurations, with testsuite_config
        taking precedence for overlapping keys.

        Args:
            cluster_config: Cluster configuration (dict or path to cluster_config.json)
                           Required keys: orchestrator, node_dict, username, priv_key_file
                           Optional keys: container,
                           head_node_dict, password (defaults provided for missing optional keys)
                           Container structure: {enabled: bool, launch: bool, runtime: {name: str, args: dict}, image: str, name: str, ...}
            testsuite_config: Test suite specific configuration (dict or path to <testsuite>_config.json)
                            Can override any keys from cluster_config

        Returns:
            OrchestratorConfig instance with extracted orchestrator keys
        """
        # Handle file paths or dicts
        if isinstance(cluster_config, str):
            import json  # Lazy import - only loaded when reading config files

            with open(cluster_config) as f:
                cluster_config = json.load(f)

        if isinstance(testsuite_config, str):
            import json  # Lazy import - only loaded when reading config files

            with open(testsuite_config) as f:
                testsuite_config = json.load(f)

        # Merge configurations
        merged_config = cluster_config.copy()
        if testsuite_config:
            merged_config.update(testsuite_config)

        # Extract only required keys for orchestrators
        required_config = {
            'orchestrator': merged_config.get('orchestrator', 'baremetal'),
            'node_dict': merged_config.get('node_dict'),
            'username': merged_config.get('username'),
            'priv_key_file': merged_config.get('priv_key_file'),
            'password': merged_config.get('password'),
            'head_node_dict': merged_config.get('head_node_dict', {}),
            'container': merged_config.get('container', {}),
        }

        # Validate required keys
        if required_config['node_dict'] is None:
            raise ValueError("node_dict must be provided in configuration")
        if required_config['username'] is None:
            raise ValueError("username must be provided in configuration")
        if required_config['priv_key_file'] is None:
            raise ValueError("priv_key_file must be provided in configuration")

        return cls(**required_config)


class OrchestratorFactory:
    """
    Factory for creating orchestrator instances based on configuration.

    Provides pluggable architecture for adding new backends without
    modifying existing test code.
    """

    @staticmethod
    def create_orchestrator(log, config, stop_on_errors=False):
        """
        Create orchestrator instance from configuration.

        Args:
            log: Logger instance
            config: OrchestratorConfig instance
            stop_on_errors: Whether to stop execution on first error

        Returns:
            Orchestrator instance

        Raises:
            ValueError: If orchestrator type is unsupported
            TypeError: If config is not OrchestratorConfig instance
        """
        if not isinstance(config, OrchestratorConfig):
            raise TypeError("config must be OrchestratorConfig instance")

        orchestrator_type = config.orchestrator.lower()

        if orchestrator_type == 'baremetal':
            return BaremetalOrchestrator(log, config, stop_on_errors)
        elif orchestrator_type == 'container':
            return ContainerOrchestrator(log, config, stop_on_errors)
        else:
            raise ValueError(f"Unsupported orchestrator type: {orchestrator_type}")

    @staticmethod
    def get_supported_backends():
        """
        Get list of supported orchestrator backends.

        Returns:
            List of backend names
        """
        return ['baremetal', 'container']
