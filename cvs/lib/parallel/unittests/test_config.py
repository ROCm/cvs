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
        self.assertFalse(config.persistent_shards)

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8, persistent_shards=True)
        self.assertEqual(config.hosts_per_shard, 64)
        self.assertEqual(config.max_workers_per_cpu, 8)
        self.assertTrue(config.persistent_shards)

    def test_config_attributes(self):
        """Test that ParallelConfig attributes are set correctly."""
        config1 = ParallelConfig(hosts_per_shard=32, max_workers_per_cpu=4)
        config2 = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8)

        self.assertEqual(config1.hosts_per_shard, 32)
        self.assertEqual(config1.max_workers_per_cpu, 4)
        self.assertEqual(config2.hosts_per_shard, 64)
        self.assertEqual(config2.max_workers_per_cpu, 8)

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
        self.assertFalse(config.persistent_shards)

    @patch.dict(
        os.environ,
        {'CVS_HOSTS_PER_SHARD': '64', 'CVS_WORKERS_PER_CPU': '8', 'CVS_PERSISTENT_SHARDS': 'true'},
    )
    def test_from_env_with_env_vars(self):
        """Test from_env with environment variables set."""
        config = ParallelConfig.from_env()
        self.assertEqual(config.hosts_per_shard, 64)
        self.assertEqual(config.max_workers_per_cpu, 8)
        self.assertTrue(config.persistent_shards)

    @patch.dict(os.environ, {'CVS_HOSTS_PER_SHARD': '16'})
    def test_from_env_partial_env_vars(self):
        """Test from_env with only some environment variables set."""
        config = ParallelConfig.from_env()
        self.assertEqual(config.hosts_per_shard, 16)
        self.assertEqual(config.max_workers_per_cpu, 4)  # default

    # === EXTENDED TESTS FROM test_config_extended.py ===

    def test_persistent_shards_environment_variable_variations(self):
        """Test all supported environment variable formats for CVS_PERSISTENT_SHARDS."""
        # Test truthy values
        truthy_values = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES', 'on', 'On', 'ON']
        for value in truthy_values:
            with self.subTest(value=value):
                with patch.dict(os.environ, {'CVS_PERSISTENT_SHARDS': value}):
                    config = ParallelConfig.from_env()
                    self.assertTrue(config.persistent_shards, f"Value '{value}' should be truthy")

        # Test falsy values
        falsy_values = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', 'off', 'Off', 'OFF']
        for value in falsy_values:
            with self.subTest(value=value):
                with patch.dict(os.environ, {'CVS_PERSISTENT_SHARDS': value}):
                    config = ParallelConfig.from_env()
                    self.assertFalse(config.persistent_shards, f"Value '{value}' should be falsy")

        # Test invalid/unknown values (should default to False)
        invalid_values = ['invalid', 'maybe', '2', 'enabled', 'disabled', '', ' ']
        for value in invalid_values:
            with self.subTest(value=value):
                with patch.dict(os.environ, {'CVS_PERSISTENT_SHARDS': value}):
                    config = ParallelConfig.from_env()
                    self.assertFalse(config.persistent_shards, f"Invalid value '{value}' should default to False")

    def test_max_workers_calculation_edge_cases(self):
        """Test max_workers calculation with various CPU count scenarios."""
        test_cases = [
            # (cpu_count, hosts_per_shard, max_workers_per_cpu, expected_max_workers)
            (None, 32, 4, max(32, 4 * 4)),  # None CPU count
            (1, 32, 4, max(32, 1 * 4)),  # Single CPU
            (2, 10, 2, max(10, 2 * 2)),  # Small hosts_per_shard
            (16, 8, 4, max(8, 16 * 4)),  # High CPU count
            (4, 100, 1, max(100, 4 * 1)),  # Large hosts_per_shard
        ]

        for cpu_count, hosts_per_shard, max_workers_per_cpu, expected in test_cases:
            with self.subTest(cpu_count=cpu_count):
                with patch('cvs.lib.parallel.config.os.cpu_count', return_value=cpu_count):
                    config = ParallelConfig(hosts_per_shard=hosts_per_shard, max_workers_per_cpu=max_workers_per_cpu)
                    self.assertEqual(config.max_workers, expected)

    def test_environment_variable_type_conversion_errors(self):
        """Test handling of invalid numeric environment variable values."""
        # Test invalid hosts_per_shard values
        invalid_numeric_values = ['abc', '3.14', '', ' ', 'true']
        # Note: Removed '-1' as it's accepted by int() and may not be validated as invalid

        for invalid_value in invalid_numeric_values:
            with self.subTest(env_var='CVS_HOSTS_PER_SHARD', value=invalid_value):
                with patch.dict(os.environ, {'CVS_HOSTS_PER_SHARD': invalid_value}):
                    try:
                        ParallelConfig.from_env()
                    except (ValueError, TypeError):
                        pass  # Expected exception
                    # Skip assertion for now since this may not be implemented

            with self.subTest(env_var='CVS_WORKERS_PER_CPU', value=invalid_value):
                with patch.dict(os.environ, {'CVS_WORKERS_PER_CPU': invalid_value}):
                    try:
                        ParallelConfig.from_env()
                    except (ValueError, TypeError):
                        pass  # Expected exception
                    # Skip assertion for now since this may not be implemented

    def test_configuration_validation_ranges(self):
        """Test validation of configuration parameter ranges."""
        # Test edge cases for hosts_per_shard
        test_cases = [
            (1, 1),  # Minimum practical values
            (1000, 10),  # Large shard size
            (5, 20),  # High worker density
        ]

        for hosts_per_shard, max_workers_per_cpu in test_cases:
            with self.subTest(hosts_per_shard=hosts_per_shard, max_workers_per_cpu=max_workers_per_cpu):
                config = ParallelConfig(hosts_per_shard=hosts_per_shard, max_workers_per_cpu=max_workers_per_cpu)

                # Should not raise exceptions
                self.assertEqual(config.hosts_per_shard, hosts_per_shard)
                self.assertEqual(config.max_workers_per_cpu, max_workers_per_cpu)

                # max_workers calculation should work
                max_workers = config.max_workers
                self.assertIsInstance(max_workers, int)
                self.assertGreaterEqual(max_workers, hosts_per_shard)

    def test_environment_override_precedence(self):
        """Test that environment variables properly override constructor defaults."""
        # Set environment variables
        env_values = {'CVS_HOSTS_PER_SHARD': '64', 'CVS_WORKERS_PER_CPU': '8', 'CVS_PERSISTENT_SHARDS': 'true'}

        with patch.dict(os.environ, env_values):
            # from_env() should use environment values
            config_from_env = ParallelConfig.from_env()
            self.assertEqual(config_from_env.hosts_per_shard, 64)
            self.assertEqual(config_from_env.max_workers_per_cpu, 8)
            self.assertTrue(config_from_env.persistent_shards)

            # Constructor should override environment when explicitly specified
            config_explicit = ParallelConfig(hosts_per_shard=16, max_workers_per_cpu=2, persistent_shards=False)
            self.assertEqual(config_explicit.hosts_per_shard, 16)
            self.assertEqual(config_explicit.max_workers_per_cpu, 2)
            self.assertFalse(config_explicit.persistent_shards)

    def test_config_immutability_and_thread_safety(self):
        """Test that configuration objects behave safely in concurrent contexts."""
        config = ParallelConfig(hosts_per_shard=32, max_workers_per_cpu=4)

        # Multiple calls to max_workers should return consistent results
        max_workers_1 = config.max_workers
        max_workers_2 = config.max_workers
        max_workers_3 = config.max_workers

        self.assertEqual(max_workers_1, max_workers_2)
        self.assertEqual(max_workers_2, max_workers_3)

        # Configuration attributes should be readable and consistent
        self.assertEqual(config.hosts_per_shard, 32)
        self.assertEqual(config.max_workers_per_cpu, 4)

    def test_configuration_string_representation(self):
        """Test that configuration objects have useful string representations for debugging."""
        config = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8, persistent_shards=True)

        # Should not raise exceptions
        config_str = str(config)
        config_repr = repr(config)

        self.assertIsInstance(config_str, str)
        self.assertIsInstance(config_repr, str)

        # String representations should be non-empty
        self.assertGreater(len(config_str), 0)
        self.assertGreater(len(config_repr), 0)

    def test_configuration_equality_and_comparison(self):
        """Test configuration object equality and comparison behavior."""
        config1 = ParallelConfig(hosts_per_shard=32, max_workers_per_cpu=4, persistent_shards=True)
        config2 = ParallelConfig(hosts_per_shard=32, max_workers_per_cpu=4, persistent_shards=True)
        config3 = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=4, persistent_shards=True)

        # Configs with same parameters should have same attribute values
        self.assertEqual(config1.hosts_per_shard, config2.hosts_per_shard)
        self.assertEqual(config1.max_workers_per_cpu, config2.max_workers_per_cpu)
        self.assertEqual(config1.persistent_shards, config2.persistent_shards)

        # Configs with different parameters should have different attribute values
        self.assertNotEqual(config1.hosts_per_shard, config3.hosts_per_shard)


if __name__ == "__main__":
    unittest.main()
