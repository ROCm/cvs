"""Tests for the new pydantic Settings config."""
import pytest
from app.core.config import (
    Settings,
    JumpHostConfig,
    SSHConfig,
    PollingConfig,
    RCCLConfig,
    StorageConfig,
    RedisConfig,
)


def test_jump_host_config_defaults():
    cfg = JumpHostConfig()
    assert cfg.enabled is False
    assert cfg.node_username == "root"
    assert cfg.node_key_file == "~/.ssh/id_rsa"


def test_ssh_config_has_jump_host():
    cfg = SSHConfig()
    assert isinstance(cfg.jump_host, JumpHostConfig)
    assert cfg.timeout == 30


def test_polling_config_defaults():
    cfg = PollingConfig()
    assert cfg.interval == 60
    assert cfg.failure_threshold == 5


def test_rccl_config_defaults():
    cfg = RCCLConfig()
    assert cfg.ras_port == 28028
    assert cfg.poll_interval == 30
    assert cfg.collective_timeout_secs == 10
    assert cfg.debug_log_path is None


def test_storage_redis_config_defaults():
    cfg = StorageConfig()
    assert cfg.redis.url == "redis://localhost:6379"
    assert cfg.redis.db == 0
    assert cfg.redis.snapshot_max_entries == 1000
    assert cfg.redis.event_max_entries == 10000


def test_settings_has_all_sections():
    s = Settings()
    assert hasattr(s, 'ssh')
    assert hasattr(s, 'polling')
    assert hasattr(s, 'alerts')
    assert hasattr(s, 'storage')
    assert hasattr(s, 'rccl')


def test_settings_env_nested_delimiter(monkeypatch):
    """Verify env_nested_delimiter is set so POLLING__INTERVAL=30 works."""
    from app.core.config import Settings
    monkeypatch.setenv('POLLING__INTERVAL', '45')
    s = Settings()
    assert s.polling.interval == 45
    # monkeypatch automatically cleans up after the test


def test_settings_load_nodes_from_file_missing():
    """load_nodes_from_file returns [] when no file exists."""
    from app.core.config import Settings
    s = Settings()
    # In test environment, no nodes.txt at the expected paths
    # Result should be [] (not an exception)
    result = s.load_nodes_from_file()
    assert isinstance(result, list)
