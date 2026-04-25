import unittest
from unittest.mock import patch, MagicMock
from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh
from cvs.lib.parallel.config import ParallelConfig


class TestMultiProcessPsshInitialization(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2", "host3", "host4", "host5"]

    @patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__')
    def test_init_no_sharding_small_host_list(self, mock_parent_init):
        """Test initialization without sharding for small host lists."""
        mock_parent_init.return_value = None
        small_host_list = ["host1", "host2"]
        config = ParallelConfig(hosts_per_shard=32)

        pssh = MultiProcessPssh(self.mock_log, small_host_list, user="test", config=config)

        # Should call parent __init__ for small lists
        mock_parent_init.assert_called_once_with(
            self.mock_log, small_host_list, "test", None, 'id_rsa', False, True, None
        )
        # For small lists, no sharder should be created
        self.assertFalse(hasattr(pssh, 'sharder'))

    @patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded')
    @patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder')
    def test_init_with_sharding_large_host_list(self, mock_sharder_class, mock_init_sharded):
        """Test initialization with sharding for large host lists."""
        config = ParallelConfig(hosts_per_shard=2)  # Force sharding
        mock_sharder = MagicMock()
        mock_sharder_class.return_value = mock_sharder

        MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

        # Should call _init_sharded for large lists
        mock_init_sharded.assert_called_once_with(
            self.mock_log, self.host_list, "test", None, 'id_rsa', False, True, None
        )
        # Verify sharder was created
        mock_sharder_class.assert_called_once_with(config)

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8)

        with patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__') as mock_parent_init:
            mock_parent_init.return_value = None
            pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", password="pass", config=config)

            self.assertEqual(pssh.config, config)
            mock_parent_init.assert_called_once()

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__') as mock_parent_init:
            mock_parent_init.return_value = None
            pssh = MultiProcessPssh(self.mock_log, ["host1"], user="test")

            # Should create default config
            self.assertIsNotNone(pssh.config)
            self.assertIsInstance(pssh.config, ParallelConfig)


class TestMultiProcessPsshInitSharded(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2"]

    @patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder')
    def test_init_sharded_sets_attributes(self, mock_sharder_class):
        """Test that _init_sharded sets all required attributes."""
        pssh = MultiProcessPssh.__new__(MultiProcessPssh)  # Create without calling __init__
        # Set up config and sharder that _init_sharded expects
        pssh.config = ParallelConfig(hosts_per_shard=2, max_workers_per_cpu=1)
        mock_sharder = MagicMock()
        pssh.sharder = mock_sharder

        pssh._init_sharded(
            self.mock_log, self.host_list, "testuser", "testpass", "custom_key", True, False, {"VAR": "value"}
        )

        self.assertEqual(pssh.log, self.mock_log)
        self.assertEqual(pssh.host_list, self.host_list)
        self.assertEqual(pssh.reachable_hosts, self.host_list)
        self.assertEqual(pssh.user, "testuser")
        self.assertEqual(pssh.password, "testpass")
        self.assertEqual(pssh.pkey, "custom_key")
        self.assertTrue(pssh.host_key_check)
        self.assertFalse(pssh.stop_on_errors)
        self.assertEqual(pssh.env_vars, {"VAR": "value"})
        self.assertTrue(pssh._use_process_sharding)


class TestMultiProcessPsshExec(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2"]

    def test_exec_no_sharding(self):
        """Test exec without sharding calls parent method."""
        with patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__') as mock_parent_init:
            with patch('cvs.lib.parallel.multiprocess_pssh.Pssh.exec') as mock_parent_exec:
                mock_parent_init.return_value = None
                mock_parent_exec.return_value = {"host1": "result"}

                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test")
                result = pssh.exec("uptime")

                mock_parent_exec.assert_called_once_with("uptime", timeout=None, print_console=True)
                self.assertEqual(result, {"host1": "result"})

    def test_exec_with_sharding(self):
        """Test exec with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder') as mock_sharder_class:
                mock_sharder = MagicMock()
                mock_sharder_class.return_value = mock_sharder

                # Set up the mock sharder behavior
                mock_sharder.chunk_hosts.return_value = [["host1"], ["host2"]]
                mock_sharder.create_payloads.return_value = [
                    {"operation": "exec", "init": {}, "cmd": "uptime"},
                    {"operation": "exec", "init": {}, "cmd": "uptime"},
                ]
                mock_sharder.execute_sharded.return_value = [
                    {"result": {"host1": "up1"}, "reachable_hosts": ["host1"], "unreachable_hosts": []},
                    {"result": {"host2": "up2"}, "reachable_hosts": ["host2"], "unreachable_hosts": []},
                ]
                mock_sharder.merge_results.return_value = {"host1": "up1", "host2": "up2"}

                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.log = self.mock_log
                pssh.env_prefix = None
                pssh.host_list = self.host_list
                pssh.reachable_hosts = self.host_list
                pssh.sharder = mock_sharder
                pssh._use_process_sharding = True  # Enable sharding for this test
                # Add attributes needed by _shard_init_kwargs()
                pssh.user = "test"
                pssh.password = None
                pssh.pkey = "id_rsa"
                pssh.host_key_check = False
                pssh.stop_on_errors = True
                pssh.env_vars = None

                result = pssh.exec("uptime")

                # Verify sharder methods were called
                mock_sharder.chunk_hosts.assert_called_once_with(self.host_list)
                mock_sharder.execute_sharded.assert_called_once()
                mock_sharder.merge_results.assert_called_once()
                self.assertEqual(result, {"host1": "up1", "host2": "up2"})

    def test_exec_cmd_list_with_sharding(self):
        """Test exec_cmd_list with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)
        cmd_list = ["uptime", "date"]

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder') as mock_sharder_class:
                mock_sharder = MagicMock()
                mock_sharder_class.return_value = mock_sharder

                # Set up the mock sharder behavior - create_payloads returns a list, we need the [0] element
                mock_sharder.create_payloads.return_value = [
                    {"operation": "cmd_list", "init": {}, "cmd_list": cmd_list}
                ]
                mock_sharder.execute_sharded.return_value = [
                    {
                        "result": {"host1": "up1", "host2": "date2"},
                        "reachable_hosts": ["host1", "host2"],
                        "unreachable_hosts": [],
                    },
                ]
                mock_sharder.merge_results.return_value = {"host1": "up1", "host2": "date2"}

                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.log = self.mock_log
                pssh.env_prefix = None
                pssh.host_list = self.host_list
                pssh.reachable_hosts = self.host_list
                pssh.config = config
                pssh.sharder = mock_sharder
                pssh._use_process_sharding = True  # Enable sharding for this test
                # Additional attributes needed by _shard_init_kwargs
                pssh.user = "test"
                pssh.password = None
                pssh.pkey = "id_rsa"
                pssh.host_key_check = False
                pssh.stop_on_errors = True
                pssh.env_vars = None

                result = pssh.exec_cmd_list(cmd_list)

                # Verify sharder methods were called
                mock_sharder.execute_sharded.assert_called_once()
                mock_sharder.merge_results.assert_called_once()
                self.assertEqual(result, {"host1": "up1", "host2": "date2"})

    def test_scp_file_with_sharding(self):
        """Test scp_file with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder') as mock_sharder_class:
                mock_sharder = MagicMock()
                mock_sharder_class.return_value = mock_sharder

                # Set up the mock sharder behavior
                mock_sharder.chunk_hosts.return_value = [["host1"], ["host2"]]
                mock_sharder.create_payloads.return_value = [
                    {"operation": "scp", "init": {}, "local_file": "test.txt", "remote_file": "/tmp/test.txt"},
                ]
                mock_sharder.execute_sharded.return_value = [
                    {
                        "result": {"host1": "scp1", "host2": "scp2"},
                        "reachable_hosts": ["host1", "host2"],
                        "unreachable_hosts": [],
                    },
                ]
                mock_sharder.merge_results.return_value = {"host1": "scp1", "host2": "scp2"}

                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.log = self.mock_log
                pssh.host_list = self.host_list
                pssh.reachable_hosts = self.host_list
                pssh.sharder = mock_sharder
                pssh._use_process_sharding = True  # Enable sharding for this test
                # Add attributes needed by _shard_init_kwargs()
                pssh.user = "test"
                pssh.password = None
                pssh.pkey = "id_rsa"
                pssh.host_key_check = False
                pssh.stop_on_errors = True
                pssh.env_vars = None

                result = pssh.scp_file("test.txt", "/tmp/test.txt")

                # Verify sharder methods were called
                mock_sharder.chunk_hosts.assert_called_once_with(self.host_list)
                mock_sharder.execute_sharded.assert_called_once()
                mock_sharder.merge_results.assert_called_once()
                self.assertEqual(result, {"host1": "scp1", "host2": "scp2"})

    def test_reboot_connections_with_sharding(self):
        """Test reboot_connections with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder') as mock_sharder_class:
                mock_sharder = MagicMock()
                mock_sharder_class.return_value = mock_sharder

                # Set up the mock sharder behavior
                mock_sharder.chunk_hosts.return_value = [["host1"], ["host2"]]
                mock_sharder.create_payloads.return_value = [{"operation": "reboot", "init": {}}]
                mock_sharder.execute_sharded.return_value = [
                    {
                        "result": {"host1": "reboot1", "host2": "reboot2"},
                        "reachable_hosts": ["host1", "host2"],
                        "unreachable_hosts": [],
                    },
                ]
                mock_sharder.merge_results.return_value = {"host1": "reboot1", "host2": "reboot2"}

                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.log = self.mock_log
                pssh.host_list = self.host_list
                pssh.reachable_hosts = self.host_list
                pssh.sharder = mock_sharder
                pssh._use_process_sharding = True  # Enable sharding for this test
                # Add attributes needed by _shard_init_kwargs()
                pssh.user = "test"
                pssh.password = None
                pssh.pkey = "id_rsa"
                pssh.host_key_check = False
                pssh.stop_on_errors = True
                pssh.env_vars = None

                result = pssh.reboot_connections()

                # Verify sharder methods were called
                mock_sharder.chunk_hosts.assert_called_once_with(self.host_list)
                mock_sharder.execute_sharded.assert_called_once()
                mock_sharder.merge_results.assert_called_once()
                self.assertEqual(result, {"host1": "reboot1", "host2": "reboot2"})


class TestMultiProcessPsshHelperMethods(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2"]

    def test_shard_init_kwargs(self):
        """Test _shard_init_kwargs creates correct initialization arguments."""
        config = ParallelConfig(hosts_per_shard=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder'):
                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.log = self.mock_log
                pssh.user = "test"
                pssh.password = "pass"
                pssh.pkey = "key"
                pssh.host_key_check = True
                pssh.stop_on_errors = False
                pssh.env_vars = {"TEST": "value"}

                kwargs = pssh._shard_init_kwargs()

                expected = {
                    'log': self.mock_log,
                    'user': "test",
                    'password': "pass",
                    'pkey': "key",
                    'host_key_check': True,
                    'stop_on_errors': False,
                    'env_vars': {"TEST": "value"},
                }
                self.assertEqual(kwargs, expected)

    def test_merge_shard_returns(self):
        """Test _merge_shard_returns updates host lists correctly."""
        config = ParallelConfig(hosts_per_shard=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder'):
                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.host_list = self.host_list
                pssh.reachable_hosts = []
                pssh.unreachable_hosts = []

                shard_returns = [
                    {"reachable_hosts": ["host1"], "unreachable_hosts": []},
                    {"reachable_hosts": ["host2"], "unreachable_hosts": []},
                ]

                pssh._merge_shard_returns(shard_returns)

                self.assertEqual(set(pssh.reachable_hosts), {"host1", "host2"})
                self.assertEqual(pssh.unreachable_hosts, [])


if __name__ == "__main__":
    unittest.main()
