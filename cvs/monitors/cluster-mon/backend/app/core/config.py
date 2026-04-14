"""
Configuration management for GPU cluster monitor.
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class JumpHostConfig(BaseSettings):
    """Jump host (bastion) configuration."""

    enabled: bool = Field(default=False, description="Enable jump host")
    host: Optional[str] = Field(default=None, description="Jump host IP/hostname")
    username: str = Field(default="root", description="Jump host username")
    password: Optional[str] = Field(default=None, description="Jump host password")
    key_file: str = Field(default="~/.ssh/id_rsa", description="Jump host SSH key")


class ClusterSSHConfig(BaseSettings):
    """SSH configuration for cluster nodes."""

    username: str = Field(default="root", description="SSH username")
    password: Optional[str] = Field(default=None, description="SSH password")
    key_file: str = Field(default="~/.ssh/id_rsa", description="SSH private key file")
    timeout: int = Field(default=30, description="SSH timeout in seconds")
    jump_host: JumpHostConfig = Field(default_factory=JumpHostConfig, description="Jump host config")


class PollingConfig(BaseSettings):
    """Polling configuration."""

    interval: int = Field(default=15, description="Polling interval in seconds")
    batch_size: int = Field(default=10, description="Number of nodes per batch")
    stagger_delay: int = Field(default=2, description="Delay between batches in seconds")


class RedisConfig(BaseSettings):
    """Redis configuration."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    ttl: int = Field(default=900, description="TTL for cached data in seconds (15 min)")


class InfluxDBConfig(BaseSettings):
    """InfluxDB configuration."""

    url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    token: Optional[str] = Field(default=None, description="InfluxDB auth token")
    org: str = Field(default="gpu-monitor", description="InfluxDB organization")
    bucket: str = Field(default="gpu_cluster", description="InfluxDB bucket")


class AlertConfig(BaseSettings):
    """Alert threshold configuration."""

    gpu_temp_threshold: float = Field(default=85.0, description="GPU temperature threshold (C)")
    gpu_util_threshold: float = Field(default=95.0, description="GPU utilization threshold (%)")
    gpu_mem_threshold: float = Field(default=95.0, description="GPU memory threshold (%)")
    error_count_threshold: int = Field(default=10, description="Error count threshold")
    nic_error_threshold: int = Field(default=100, description="NIC error count threshold")


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = "GPU Cluster Monitor"
    debug: bool = False

    # Cluster nodes
    nodes_file: str = Field(default="../config/nodes.txt", description="File with node IPs")
    nodes: List[str] = Field(default_factory=list, description="List of node IPs")

    # Sub-configurations
    ssh: ClusterSSHConfig = Field(default_factory=ClusterSSHConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)

    # API
    api_prefix: str = "/api"
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            # Configured via environment or simple_config.py
        ]
    )

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "allow"  # Allow extra fields for YAML loading compatibility

    def load_nodes_from_file(self) -> List[str]:
        """Load node IPs from file."""
        nodes_path = Path(self.nodes_file)
        if nodes_path.exists():
            with open(nodes_path) as f:
                nodes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            self.nodes = nodes
            return nodes
        return []

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> "Settings":
        """Load settings from YAML file."""
        with open(yaml_file) as f:
            config_data = yaml.safe_load(f)

        # Flatten nested structure for Pydantic
        settings_dict = {}

        if "cluster" in config_data:
            cluster = config_data["cluster"]

            if "nodes_file" in cluster:
                settings_dict["nodes_file"] = cluster["nodes_file"]

            if "ssh" in cluster:
                for key, value in cluster["ssh"].items():
                    if key == "jump_host" and isinstance(value, dict):
                        # Handle nested jump_host configuration
                        for jh_key, jh_value in value.items():
                            settings_dict[f"ssh__jump_host__{jh_key}"] = jh_value
                    else:
                        settings_dict[f"ssh__{key}"] = value

            if "polling" in cluster:
                for key, value in cluster["polling"].items():
                    settings_dict[f"polling__{key}"] = value

            if "storage" in cluster:
                storage = cluster["storage"]
                if "redis" in storage:
                    for key, value in storage["redis"].items():
                        settings_dict[f"redis__{key}"] = value
                if "influxdb" in storage:
                    for key, value in storage["influxdb"].items():
                        settings_dict[f"influxdb__{key}"] = value

            if "alerts" in cluster:
                for key, value in cluster["alerts"].items():
                    settings_dict[f"alerts__{key}"] = value

        return cls(**settings_dict)


# Global settings instance
# Try to load from YAML first, fall back to defaults
try:
    # Get absolute path to config file (relative to backend directory)
    yaml_path = Path("../config/cluster.yaml").resolve()

    if yaml_path.exists():
        logger.info(f"Loading configuration from: {yaml_path}")
        settings = Settings.load_from_yaml(str(yaml_path))
        logger.info(f"Jump host enabled: {settings.ssh.jump_host.enabled}")
        logger.info(f"Jump host: {settings.ssh.jump_host.host}")
    else:
        logger.warning(f"YAML file not found at: {yaml_path}, using defaults")
        settings = Settings()
except Exception as e:
    logger.error(f"Error loading YAML config: {e}")
    import traceback

    traceback.print_exc()
    settings = Settings()
