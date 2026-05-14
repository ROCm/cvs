import unittest
from unittest.mock import patch, MagicMock
from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh
from cvs.lib.parallel.config import ParallelConfig


class TestMultiProcessPsshInitialization(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2", "host3", "host4", "host5"]
        # Mock Pssh.__init__ for all tests in this class
        self.pssh_patcher = patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__')
        self.mock_pssh_init = self.pssh_patcher.start()
        self.mock_pssh_init.return_value = None

    def tearDown(self):
        self.pssh_patcher.stop()

    def test_init_no_sharding_small_host_list(self):
        """Test initialization without sharding for small host lists."""
        small_host_list = ["host1", "host2"]
        config = ParallelConfig(hosts_per_shard=32)

        pssh = MultiProcessPssh(self.mock_log, small_host_list, user="test", config=config)

        # Always creates composed Pssh instance with ABC+composition
        self.mock_pssh_init.assert_called_once_with(
            self.mock_log, small_host_list, "test", None, 'id_rsa', False, True, None, process_output=True
        )
        # For small lists, no sharder should be created (key difference!)
        self.assertFalse(hasattr(pssh, 'sharder'))
        self.assertFalse(pssh._use_process_sharding)
        # But pssh instance should always exist
        self.assertTrue(hasattr(pssh, 'pssh'))

    @patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded')
    @patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder')
    def test_init_with_sharding_large_host_list(self, mock_sharder_class, mock_init_sharded):
        """Test initialization with sharding for large host lists."""
        config = ParallelConfig(hosts_per_shard=2)  # Force sharding
        mock_sharder = MagicMock()
        mock_sharder_class.return_value = mock_sharder

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

        # In sharded mode, should NOT create Pssh instance (avoids bottleneck)
        self.mock_pssh_init.assert_not_called()
        # Should call _init_sharded for large lists
        mock_init_sharded.assert_called_once_with(
            self.mock_log, self.host_list, "test", None, 'id_rsa', False, True, None
        )
        # Verify sharder was created
        mock_sharder_class.assert_called_once_with(config)
        self.assertTrue(pssh._use_process_sharding)
        # In sharded mode: pssh is None, sharder exists
        self.assertIsNone(pssh.pssh)
        self.assertTrue(hasattr(pssh, 'sharder'))

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ParallelConfig(hosts_per_shard=64, max_workers_per_cpu=8)

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", password="pass", config=config)

        self.assertEqual(pssh.config, config)


class TestMultiProcessPsshExec(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2"]
        # Mock Pssh.__init__ for all tests in this class
        self.pssh_patcher = patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__')
        self.mock_pssh_init = self.pssh_patcher.start()
        self.mock_pssh_init.return_value = None

        # Mock execute_sharded to avoid process spawning while testing real payload creation
        self.execute_sharded_patcher = patch('cvs.lib.parallel.pssh_sharder.PsshSharder.execute_sharded')
        self.mock_execute_sharded = self.execute_sharded_patcher.start()

    def tearDown(self):
        self.pssh_patcher.stop()
        self.execute_sharded_patcher.stop()

    def test_exec_with_sharding(self):
        """Test exec with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        # Mock execute_sharded to return realistic shard results
        self.mock_execute_sharded.return_value = [
            {'result': {"host1": "up1"}, 'reachable_hosts': ["host1"], 'unreachable_hosts': []},
            {'result': {"host2": "up2"}, 'reachable_hosts': ["host2"], 'unreachable_hosts': []},
        ]

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.exec("uptime", timeout=30, detailed=True)

        # Verify execute_sharded was called with real payloads from create_payloads
        self.mock_execute_sharded.assert_called_once()
        payloads = self.mock_execute_sharded.call_args[0][0]

        # Check that real payloads were created correctly
        self.assertEqual(len(payloads), 2)  # Two shards (hosts_per_shard=1)
        first_payload = payloads[0]
        self.assertEqual(first_payload['operation'], 'exec')
        self.assertEqual(first_payload['cmd'], 'uptime')
        self.assertEqual(first_payload['timeout'], 30)
        self.assertEqual(first_payload['detailed'], True)
        self.assertIn('init', first_payload)

        # Verify real init kwargs are properly constructed by _shard_init_kwargs
        init_kwargs = first_payload['init']
        self.assertEqual(init_kwargs['user'], 'test')
        self.assertIn('host_list', init_kwargs)

        # Verify result is merged from shards
        self.assertEqual(result, {"host1": "up1", "host2": "up2"})

    def test_exec_cmd_list_with_sharding(self):
        """Test exec_cmd_list with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)
        cmd_list = ["uptime", "date"]

        # Mock execute_sharded to capture real payloads and return results
        def capture_payloads_and_return_results(payloads):
            self.captured_payloads = payloads
            return [
                {
                    'result': {"host1": "up1", "host2": "date2"},
                    'reachable_hosts': ["host1", "host2"],
                    'unreachable_hosts': [],
                }
            ]

        self.mock_execute_sharded.side_effect = capture_payloads_and_return_results

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.exec_cmd_list(cmd_list, timeout=60)

        # Verify execute_sharded was called with real payloads
        self.mock_execute_sharded.assert_called_once()

        # Check that real payloads were created correctly by create_payloads
        self.assertEqual(len(self.captured_payloads), 2)  # Two shards
        first_payload = self.captured_payloads[0]
        self.assertEqual(first_payload['operation'], 'exec_cmd_list')
        # With hosts_per_shard=1, each shard gets one command from the list
        self.assertEqual(first_payload['cmd_list'], ['uptime'])  # First shard gets first command
        self.assertEqual(first_payload['timeout'], 60)
        self.assertIn('init', first_payload)

        # Verify real init kwargs
        init_kwargs = first_payload['init']
        self.assertEqual(init_kwargs['user'], 'test')

        # Verify result is merged from shards
        self.assertEqual(result, {"host1": "up1", "host2": "date2"})

    def test_exec_cmd_list_with_unreachable_hosts(self):
        """Test exec_cmd_list correctly maps commands when some hosts are unreachable."""
        config = ParallelConfig(hosts_per_shard=2, max_workers_per_cpu=1)
        cmd_list = ["cmd1", "cmd2", "cmd3", "cmd4"]

        # Simulate host2 being unreachable (removed from reachable_hosts)
        original_hosts = ["host1", "host2", "host3", "host4"]
        reachable_hosts_only = ["host1", "host3", "host4"]  # host2 is unreachable

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder') as mock_sharder_class:
                mock_sharder = MagicMock()
                mock_sharder_class.return_value = mock_sharder

                # Mock sharder behavior
                mock_sharder.chunk_hosts.return_value = [["host1", "host3"], ["host4"]]
                mock_sharder.create_payloads.return_value = [
                    {
                        "operation": "exec_cmd_list",
                        "init": {},
                        "cmd_list": ["cmd1", "cmd3"],
                    },  # Expected: correct commands
                ]
                mock_sharder.execute_sharded.return_value = [
                    {
                        "result": {"host1": "result1", "host3": "result3", "host4": "result4"},
                        "reachable_hosts": reachable_hosts_only,
                        "unreachable_hosts": [],
                    }
                ]
                mock_sharder.merge_results.return_value = {"host1": "result1", "host3": "result3", "host4": "result4"}

                pssh = MultiProcessPssh(self.mock_log, original_hosts, user="test", config=config)

                # Set up test state with unreachable host
                pssh.env_prefix = None
                pssh.host_list = original_hosts
                pssh.reachable_hosts = reachable_hosts_only  # host2 is missing
                pssh.config = config
                pssh.sharder = mock_sharder
                pssh._use_process_sharding = True
                # Additional attributes needed by _shard_init_kwargs
                pssh.user = "test"
                pssh.password = None
                pssh.pkey = "id_rsa"
                pssh.host_key_check = False
                pssh.stop_on_errors = True
                pssh.env_vars = None

                result = pssh.exec_cmd_list(cmd_list)

                # Verify create_payloads was called and check the command mapping
                mock_sharder.create_payloads.assert_called()

                # The cmd_list should be filtered to match reachable hosts
                # Original: cmd_list=["cmd1", "cmd2", "cmd3", "cmd4"] for ["host1", "host2", "host3", "host4"]
                # Filtered: should be ["cmd1", "cmd3", "cmd4"] for ["host1", "host3", "host4"]
                # This validates that host3 gets cmd3 (not cmd2) and host4 gets cmd4 (not cmd3)

                self.assertEqual(result, {"host1": "result1", "host3": "result3", "host4": "result4"})

    def test_scp_file_with_sharding(self):
        """Test scp_file delegates to upload_file with sharding."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        # Mock execute_sharded for void scp_file operation
        self.mock_execute_sharded.return_value = [
            {'result': None, 'reachable_hosts': ["host1"], 'unreachable_hosts': []},
            {'result': None, 'reachable_hosts': ["host2"], 'unreachable_hosts': []},
        ]

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.scp_file("test.txt", "/tmp/test.txt", recurse=False)

        # Verify execute_sharded was called with real payloads
        self.mock_execute_sharded.assert_called_once()
        payloads = self.mock_execute_sharded.call_args[0][0]

        # Check that real payloads were created for upload_file operation
        self.assertEqual(len(payloads), 2)  # Two shards
        first_payload = payloads[0]
        self.assertEqual(first_payload['operation'], 'upload_file')  # scp_file delegates to upload_file
        self.assertEqual(first_payload['local_file'], 'test.txt')
        self.assertEqual(first_payload['remote_file'], '/tmp/test.txt')
        self.assertEqual(first_payload['recurse'], False)

        # scp_file should return empty dict (merged void results)
        self.assertEqual(result, {})

    def test_reboot_connections_with_sharding(self):
        """Test reboot_connections with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        # Mock execute_sharded for void reboot_connections operation
        self.mock_execute_sharded.return_value = [
            {'result': None, 'reachable_hosts': ["host1"], 'unreachable_hosts': []},
            {'result': None, 'reachable_hosts': ["host2"], 'unreachable_hosts': []},
        ]

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.reboot_connections()

        # Verify execute_sharded was called with real payloads
        self.mock_execute_sharded.assert_called_once()
        payloads = self.mock_execute_sharded.call_args[0][0]

        # Check that real payloads were created correctly
        self.assertEqual(len(payloads), 2)  # Two shards
        first_payload = payloads[0]
        self.assertEqual(first_payload['operation'], 'reboot_connections')
        self.assertIn('init', first_payload)

        # reboot_connections should return empty dict (merged void results)
        self.assertEqual(result, {})


class TestMultiProcessPsshHelperMethods(unittest.TestCase):
    def setUp(self):
        self.mock_log = MagicMock()
        self.host_list = ["host1", "host2"]
        # Mock Pssh.__init__ for all tests in this class
        self.pssh_patcher = patch('cvs.lib.parallel.multiprocess_pssh.Pssh.__init__')
        self.mock_pssh_init = self.pssh_patcher.start()
        self.mock_pssh_init.return_value = None

        # Mock execute_sharded to avoid process spawning while testing real payload creation
        self.execute_sharded_patcher = patch('cvs.lib.parallel.pssh_sharder.PsshSharder.execute_sharded')
        self.mock_execute_sharded = self.execute_sharded_patcher.start()

    def tearDown(self):
        self.pssh_patcher.stop()
        self.execute_sharded_patcher.stop()

    def test_shard_init_kwargs(self):
        """Test _shard_init_kwargs creates correct initialization arguments."""
        config = ParallelConfig(hosts_per_shard=1)

        with patch('cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh._init_sharded'):
            with patch('cvs.lib.parallel.multiprocess_pssh.PsshSharder'):
                pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)

                # Manually set attributes that _init_sharded would set
                pssh.user = "test"
                pssh.password = "pass"
                pssh.pkey = "key"
                pssh.host_key_check = True
                pssh.stop_on_errors = False
                pssh.env_vars = {"TEST": "value"}

                kwargs = pssh._shard_init_kwargs()

                expected = {
                    'log': None,  # No longer pass logger to child processes (fixes pickling issue)
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

    def test_upload_file_with_sharding(self):
        """Test upload_file with sharding uses sharder."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        # Mock execute_sharded for void upload_file operation
        self.mock_execute_sharded.return_value = [
            {'result': None, 'reachable_hosts': ["host1"], 'unreachable_hosts': []},
            {'result': None, 'reachable_hosts': ["host2"], 'unreachable_hosts': []},
        ]

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.upload_file("test.txt", "/tmp/test.txt", recurse=True)

        # Verify execute_sharded was called with real payloads
        self.mock_execute_sharded.assert_called_once()
        payloads = self.mock_execute_sharded.call_args[0][0]

        # Check that real payload was created correctly
        self.assertEqual(len(payloads), 2)  # Two shards
        first_payload = payloads[0]
        self.assertEqual(first_payload['operation'], 'upload_file')
        self.assertEqual(first_payload['local_file'], 'test.txt')
        self.assertEqual(first_payload['remote_file'], '/tmp/test.txt')
        self.assertEqual(first_payload['recurse'], True)
        self.assertIn('init', first_payload)

        # upload_file should return empty dict (merged void results)
        self.assertEqual(result, {})

    def test_download_file_with_sharding(self):
        """Test download_file with sharding uses sharder and merges results."""
        config = ParallelConfig(hosts_per_shard=1, max_workers_per_cpu=1)

        # Mock execute_sharded to return realistic download results
        self.mock_execute_sharded.return_value = [
            {'result': {"host1": "/tmp/test.txt_host1"}, 'reachable_hosts': ["host1"], 'unreachable_hosts': []},
            {'result': {"host2": "/tmp/test.txt_host2"}, 'reachable_hosts': ["host2"], 'unreachable_hosts': []},
        ]

        pssh = MultiProcessPssh(self.mock_log, self.host_list, user="test", config=config)
        result = pssh.download_file("/remote/test.txt", "/tmp/test.txt", suffix_separator="_")

        # Verify execute_sharded was called with real payloads
        self.mock_execute_sharded.assert_called_once()
        payloads = self.mock_execute_sharded.call_args[0][0]

        # Check that real payload was created correctly
        self.assertEqual(len(payloads), 2)  # Two shards
        first_payload = payloads[0]
        self.assertEqual(first_payload['operation'], 'download_file')
        self.assertEqual(first_payload['remote_file'], '/remote/test.txt')
        self.assertEqual(first_payload['local_file'], '/tmp/test.txt')
        self.assertEqual(first_payload['suffix_separator'], '_')
        self.assertIn('init', first_payload)

        # download_file should return merged host->path dict
        expected_result = {"host1": "/tmp/test.txt_host1", "host2": "/tmp/test.txt_host2"}
        self.assertEqual(result, expected_result)
