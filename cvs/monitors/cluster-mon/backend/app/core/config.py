"""
Configuration management for CVS Cluster Monitor.
Uses pydantic-settings with a YAML source and env var overrides.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from typing import Any, Optional, Tuple, Type, List
import yaml
from pathlib import Path
import os


class JumpHostConfig(BaseModel):
    enabled: bool = False
    host: Optional[str] = None
    username: str = "root"
    password: Optional[str] = None
    key_file: str = "~/.ssh/id_rsa"
    node_username: str = "root"       # replaces: node_username_via_jumphost
    node_key_file: str = "~/.ssh/id_rsa"  # replaces: node_key_file_on_jumphost


class SSHConfig(BaseModel):
    username: str = "root"
    key_file: str = "~/.ssh/id_rsa"
    password: Optional[str] = None
    timeout: int = 30
    jump_host: JumpHostConfig = Field(default_factory=JumpHostConfig)


class PollingConfig(BaseModel):
    interval: int = 60
    batch_size: int = 10
    stagger_delay: int = 2
    failure_threshold: int = 5


class AlertsConfig(BaseModel):
    gpu_temp_threshold: float = 85.0
    gpu_util_threshold: float = 95.0


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379"
    db: int = 0
    snapshot_max_entries: int = 1000
    event_max_entries: int = 10000


class StorageConfig(BaseModel):
    redis: RedisConfig = Field(default_factory=RedisConfig)


class RCCLConfig(BaseModel):
    """Forward-declaration for RCCL extension config. No runtime behaviour in base robustness spec."""
    ras_port: int = 28028
    poll_interval: int = 30
    collective_timeout_secs: int = 10
    debug_log_path: Optional[str] = None


class _YamlSource(PydanticBaseSettingsSource):
    """
    Loads cluster.yaml as a pydantic-settings source.
    Compatible with pydantic-settings 2.1.0.
    """

    def __init__(self, settings_cls: Type[BaseSettings], yaml_path: Path):
        super().__init__(settings_cls)
        self._path = yaml_path

    def __call__(self) -> dict[str, Any]:
        if self._path.exists():
            raw = yaml.safe_load(self._path.read_text()) or {}
            return raw.get("cluster", {})
        return {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Required by PydanticBaseSettingsSource ABC in pydantic-settings 2.1.0.
        # Not called when __call__() returns the full dict; stub satisfies the ABC.
        raise NotImplementedError

    def field_is_complex(self, field: Any) -> bool:
        return True


class Settings(BaseSettings):
    app_name: str = "CVS Cluster Monitor"
    api_prefix: str = "/api"
    cors_origins: List[str] = Field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
    ])
    nodes_file: str = "config/nodes.txt"
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    rccl: RCCLConfig = Field(default_factory=RCCLConfig)

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Try Docker path first, then dev path
        yaml_path = Path("/app/config/cluster.yaml")
        if not yaml_path.exists():
            yaml_path = Path("../config/cluster.yaml")
        return (
            init_settings,
            env_settings,                           # env vars override YAML
            _YamlSource(settings_cls, yaml_path),   # YAML is primary source
            file_secret_settings,
        )

    def load_nodes_from_file(self) -> List[str]:
        """Load node IPs from nodes file, trying multiple paths."""
        possible_paths = [
            Path("/app/config/nodes.txt"),
            Path("../config/nodes.txt"),
            Path(self.nodes_file),
        ]
        for p in possible_paths:
            p = p.resolve()
            if p.exists():
                nodes = [
                    line.strip()
                    for line in p.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                if nodes:
                    return nodes
        return []

    # Backward-compat properties used in existing main.py and api/config.py
    # These will be removed after main.py is fully migrated.
    @property
    def node_username_via_jumphost(self) -> str:
        return self.ssh.jump_host.node_username

    @property
    def node_key_file_on_jumphost(self) -> str:
        return self.ssh.jump_host.node_key_file

    @property
    def ssh_username(self) -> str:
        return self.ssh.username

    @property
    def ssh_password(self) -> Optional[str]:
        return self.ssh.password

    @property
    def ssh_key_file(self) -> str:
        return self.ssh.key_file

    @property
    def jump_host_enabled(self) -> bool:
        return self.ssh.jump_host.enabled

    @property
    def jump_host(self) -> Optional[str]:
        return self.ssh.jump_host.host

    @property
    def jump_host_username(self) -> str:
        return self.ssh.jump_host.username

    @property
    def jump_host_key_file(self) -> str:
        return self.ssh.jump_host.key_file


# Global settings instance
settings = Settings()
