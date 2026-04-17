"""Tests for reload_configuration topology-diff logic."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


def test_settings_model_dump_comparison():
    """Verify that Settings.model_dump() can detect config changes."""
    from app.core.config import Settings
    s1 = Settings()
    s2 = Settings()
    # Same defaults = equal dumps
    assert s1.ssh.model_dump() == s2.ssh.model_dump()
    assert s1.polling.model_dump() == s2.polling.model_dump()
    assert s1.rccl.model_dump() == s2.rccl.model_dump()


def test_settings_model_dump_detects_change(monkeypatch):
    """Different env vars should produce different model_dump()."""
    from app.core.config import Settings
    s1 = Settings()
    monkeypatch.setenv('POLLING__INTERVAL', '99')
    s2 = Settings()
    assert s1.polling.model_dump() != s2.polling.model_dump()
