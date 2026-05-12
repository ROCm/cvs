import unittest
import os
from unittest.mock import patch
from cvs.lib.parallel.config import ParallelConfig


class TestParallelConfig(unittest.TestCase):
    def test_default_values(self):
        """Test default configuration values."""
        config = ParallelConfig()
        self.assertEqual(config.hosts_per_shard, 32)
        self.assertEqual(config.max_workers_per_cpu, 4)

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8)
        self.assertEqual(config.hosts_per_shard, 64)
        self.assertEqual(config.max_workers_per_cpu, 8)

    @patch('os.cpu_count')
    def test_max_workers_calculation(self, mock_cpu_count):
        """Test max_workers calculation based on CPU count."""
        mock_cpu_count.return_value = 8
        config = ParallelConfig(max_workers_per_cpu=4)

        # max(32, 8 * 4) = max(32, 32) = 32
        self.assertEqual(config.max_workers, 32)

    @patch('os.cpu_count')
    def test_max_workers_with_high_cpu_count(self, mock_cpu_count):
        """Test max_workers with high CPU count."""
        mock_cpu_count.return_value = 16
        config = ParallelConfig(max_workers_per_cpu=4)

        # max(32, 16 * 4) = max(32, 64) = 64
        self.assertEqual(config.max_workers, 64)

    @patch('os.cpu_count')
    def test_max_workers_when_cpu_count_none(self, mock_cpu_count):
        """Test max_workers when os.cpu_count() returns None."""
        mock_cpu_count.return_value = None
        config = ParallelConfig(max_workers_per_cpu=4)

        # max(32, 4 * 4) = max(32, 16) = 32 (fallback to 4 CPUs)
        self.assertEqual(config.max_workers, 32)

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_no_env_vars(self):
        """Test from_env with no environment variables set."""
        config = ParallelConfig.from_env()
        self.assertEqual(config.hosts_per_shard, 32)  # default
        self.assertEqual(config.max_workers_per_cpu, 4)  # default

    @patch.dict(os.environ, {'CVS_HOSTS_PER_SHARD': '64', 'CVS_WORKERS_PER_CPU': '8'})
    def test_from_env_with_env_vars(self):
        """Test from_env with environment variables set."""
        config = ParallelConfig.from_env()
        self.assertEqual(config.hosts_per_shard, 64)
        self.assertEqual(config.max_workers_per_cpu, 8)

    @patch.dict(os.environ, {'CVS_HOSTS_PER_SHARD': '16'})
    def test_from_env_partial_env_vars(self):
        """Test from_env with only some environment variables set."""
        config = ParallelConfig.from_env()
        self.assertEqual(config.hosts_per_shard, 16)
        self.assertEqual(config.max_workers_per_cpu, 4)  # default

    def test_config_attributes(self):
        """Test that ParallelConfig attributes are set correctly."""
        config1 = ParallelConfig(hosts_per_shard=32, max_workers_per_cpu=4)
        config2 = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8)

        self.assertEqual(config1.hosts_per_shard, 32)
        self.assertEqual(config1.max_workers_per_cpu, 4)
        self.assertEqual(config2.hosts_per_shard, 64)
        self.assertEqual(config2.max_workers_per_cpu, 8)


if __name__ == "__main__":
    unittest.main()
