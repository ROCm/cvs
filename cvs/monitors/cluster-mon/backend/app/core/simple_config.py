"""
Simplified configuration loader - reads directly from YAML.
Avoids Pydantic BaseSettings nested model issues.
"""

import yaml
from pathlib import Path
from typing import List, Optional


class SimpleConfig:
    """Simple configuration loader from YAML file."""

    def __init__(self, yaml_path: str = None):
        # Auto-detect config path for both dev and Docker
        if yaml_path is None:
            # Try Docker path first
            docker_path = Path("/app/config/cluster.yaml")
            if docker_path.exists():
                yaml_path = str(docker_path)
            else:
                # Fallback to development path
                yaml_path = "../config/cluster.yaml"

        self.yaml_path = Path(yaml_path).resolve()
        self.config_data = {}
        self.load()

    def load(self):
        """Load configuration from YAML file."""
        if self.yaml_path.exists():
            with open(self.yaml_path) as f:
                data = yaml.safe_load(f)
                self.config_data = data.get("cluster", {})
        else:
            print(f"Warning: Config file not found at {self.yaml_path}")
            self.config_data = {}

    def get_nodes_file(self) -> str:
        """Get nodes file path."""
        # Try Docker path first
        docker_nodes = Path("/app/config/nodes.txt")
        if docker_nodes.exists():
            return str(docker_nodes)
        return self.config_data.get("nodes_file", "../config/nodes.txt")

    def load_nodes_from_file(self) -> List[str]:
        """Load node IPs from nodes file."""
        # Try multiple possible paths
        import os

        possible_paths = [
            Path("/app/config/nodes.txt"),  # Docker path (first priority)
            Path("../config/nodes.txt"),  # Development path
            Path(os.path.join(os.getenv("CLUSTER_MONITOR_HOME", "."), "config/nodes.txt")),
            Path(self.get_nodes_file()),
        ]

        for nodes_file in possible_paths:
            nodes_file = nodes_file.resolve()
            if nodes_file.exists():
                with open(nodes_file) as f:
                    nodes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                if nodes:
                    return nodes

        return []

    # SSH Configuration
    @property
    def ssh_username(self) -> str:
        import os

        default_user = os.getenv("USER", "root")
        return self.config_data.get("ssh", {}).get("username", default_user)

    @property
    def ssh_password(self) -> Optional[str]:
        # SECURITY: Password is stored in memory only (app_state), never in YAML
        try:
            from app.main import app_state

            return app_state.ssh_password
        except:
            return None

    @property
    def ssh_key_file(self) -> str:
        return self.config_data.get("ssh", {}).get("key_file", "~/.ssh/id_rsa")

    @property
    def ssh_timeout(self) -> int:
        return self.config_data.get("ssh", {}).get("timeout", 30)

    # Jump Host Configuration
    @property
    def jump_host_enabled(self) -> bool:
        return self.config_data.get("ssh", {}).get("jump_host", {}).get("enabled", False)

    @property
    def jump_host(self) -> Optional[str]:
        if self.jump_host_enabled:
            return self.config_data.get("ssh", {}).get("jump_host", {}).get("host")
        return None

    @property
    def jump_host_username(self) -> str:
        import os

        default_user = os.getenv("USER", "root")
        return self.config_data.get("ssh", {}).get("jump_host", {}).get("username", default_user)

    @property
    def jump_host_password(self) -> Optional[str]:
        # SECURITY: Password is stored in memory only (app_state), never in YAML
        # However, for testing/development, we also check YAML
        try:
            from app.main import app_state

            if app_state.jump_host_password:
                return app_state.jump_host_password
        except:
            pass

        # Fallback: Read from YAML (for testing/development only)
        # Production should never have passwords in YAML
        return self.config_data.get("ssh", {}).get("jump_host", {}).get("password")

    @property
    def jump_host_key_file(self) -> str:
        """Local keyfile to SSH to jump host."""
        return self.config_data.get("ssh", {}).get("jump_host", {}).get("key_file", "~/.ssh/id_rsa")

    @property
    def node_username_via_jumphost(self) -> str:
        """Username for cluster nodes when using jump host."""
        import os

        default_user = os.getenv("USER", "root")
        # First check for node_username_via_jumphost at ssh level, then check jump_host.node_username
        return self.config_data.get("ssh", {}).get("node_username_via_jumphost") or self.config_data.get("ssh", {}).get(
            "jump_host", {}
        ).get("node_username", default_user)

    @property
    def node_key_file_on_jumphost(self) -> str:
        """Path to private key ON JUMP HOST for accessing cluster nodes."""
        # First check for node_key_file_on_jumphost at ssh level, then check jump_host.node_key_file
        return self.config_data.get("ssh", {}).get("node_key_file_on_jumphost") or self.config_data.get("ssh", {}).get(
            "jump_host", {}
        ).get("node_key_file", "~/.ssh/id_rsa")

    # Polling Configuration
    @property
    def polling_interval(self) -> int:
        return self.config_data.get("polling", {}).get("interval", 60)

    @property
    def polling_batch_size(self) -> int:
        return self.config_data.get("polling", {}).get("batch_size", 10)

    @property
    def polling_stagger_delay(self) -> int:
        return self.config_data.get("polling", {}).get("stagger_delay", 2)

    # Alert Thresholds
    @property
    def gpu_temp_threshold(self) -> float:
        return self.config_data.get("alerts", {}).get("gpu_temp_threshold", 85.0)

    @property
    def gpu_util_threshold(self) -> float:
        return self.config_data.get("alerts", {}).get("gpu_util_threshold", 95.0)

    # CORS
    @property
    def cors_origins(self) -> List[str]:
        import os

        # Allow all origins in Docker, or specific origins from environment variable
        cors_env = os.getenv("CORS_ORIGINS", "*")
        if cors_env == "*":
            return ["*"]
        else:
            return cors_env.split(",")
        # Default for development
        # return ["http://localhost:3000", "http://localhost:5173"]

    # App settings
    @property
    def app_name(self) -> str:
        return "CVS Cluster Monitor"

    @property
    def debug(self) -> bool:
        return False

    @property
    def api_prefix(self) -> str:
        return "/api"

    @property
    def nodes(self) -> List[str]:
        return self.load_nodes_from_file()

    # SSH sub-object for compatibility
    @property
    def ssh(self):
        class SSHConfig:
            def __init__(self, parent):
                self.parent = parent
                self.username = parent.ssh_username
                self.password = parent.ssh_password
                self.key_file = parent.ssh_key_file
                self.timeout = parent.ssh_timeout

                # Jump host sub-object
                class JumpHost:
                    def __init__(self, parent):
                        self.enabled = parent.jump_host_enabled
                        self.host = parent.jump_host
                        self.username = parent.jump_host_username
                        self.password = parent.jump_host_password
                        self.key_file = parent.jump_host_key_file

                self.jump_host = JumpHost(parent)

        return SSHConfig(self)

    # Polling sub-object
    @property
    def polling(self):
        class PollingConfig:
            def __init__(self, parent):
                import os

                # Allow environment variable override
                self.interval = int(os.getenv('POLLING__INTERVAL', parent.polling_interval))
                self.batch_size = parent.polling_batch_size
                self.stagger_delay = parent.polling_stagger_delay
                self.failure_threshold = int(
                    os.getenv(
                        'POLLING__FAILURE_THRESHOLD', parent.config_data.get('polling', {}).get('failure_threshold', 5)
                    )
                )

        return PollingConfig(self)


# Global config instance
config = SimpleConfig()
