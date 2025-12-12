import unittest
from unittest.mock import MagicMock, patch, mock_open
import argparse

from cvs.debuggers.gdb_backtrace_collector import GdbBacktraceCollectorDebugger


class TestGdbBacktraceCollectorDebugger(unittest.TestCase):
    def setUp(self):
        self.debugger = GdbBacktraceCollectorDebugger()

    def test_get_name(self):
        """Test debugger name"""
        self.assertEqual(self.debugger.get_name(), "gdb_backtrace_collector")

    def test_get_description(self):
        """Test debugger description"""
        desc = self.debugger.get_description()
        self.assertIn("gdb", desc.lower())
        self.assertIn("backtrace", desc.lower())

    def test_get_parser(self):
        """Test parser creation"""
        parser = self.debugger.get_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

        # Test parsing valid arguments
        args = parser.parse_args(["--cluster_file", "/path/to/cluster.json", "--process_filter", "mpirun"])
        self.assertEqual(args.cluster_file, "/path/to/cluster.json")
        self.assertEqual(args.process_filter, "mpirun")
        self.assertEqual(args.pid, None)
        self.assertEqual(args.output_format, "console")
        self.assertEqual(args.timeout, 30)

    @patch("cvs.debuggers.gdb_backtrace_collector.resolve_cluster_config_placeholders")
    @patch("cvs.debuggers.gdb_backtrace_collector.parallel_ssh_lib.Pssh")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_debug_with_process_filter(
        self, mock_json_load, mock_open_file, mock_pssh_class, mock_resolve_placeholders
    ):
        """Test debugging with process filter"""
        # Mock cluster config
        cluster_dict = {
            'node_dict': {'node1': '192.168.1.1', 'node2': '192.168.1.2'},
            'username': 'testuser',
            'priv_key_file': '/path/to/key',
        }
        mock_json_load.return_value = cluster_dict
        mock_resolve_placeholders.return_value = cluster_dict

        # Mock PSSH instance
        mock_pssh_instance = MagicMock()
        mock_pssh_instance.reachable_hosts = ['192.168.1.1', '192.168.1.2']
        mock_pssh_class.return_value = mock_pssh_instance

        # Mock pgrep command results
        ps_results = {
            '192.168.1.1': '1234\n5678\n',
            '192.168.1.2': '5678\n',
        }
        mock_pssh_instance.exec.return_value = ps_results

        # Mock gdb command results - one result per PID per host
        gdb_results = {
            0: '#0 in main()\n',  # First command: host 192.168.1.1, PID 1234
            1: '#0 in MPI_Barrier()\n',  # Second command: host 192.168.1.1, PID 5678
            2: '#0 in allreduce()\n',  # Third command: host 192.168.1.2, PID 5678
        }
        # Convert to list format expected by exec_cmd_list
        gdb_results_list = [
            gdb_results[0],  # host 192.168.1.1, PID 1234
            gdb_results[1],  # host 192.168.1.1, PID 5678
            gdb_results[2],  # host 192.168.1.2, PID 5678
        ]
        mock_pssh_instance.exec_cmd_list.return_value = gdb_results_list

        args = MagicMock()
        args.cluster_file = "/path/to/cluster.json"
        args.process_filter = "mpirun"
        args.pid = None
        args.node = None
        args.output_format = "console"
        args.timeout = 30

        with patch("builtins.print") as mock_print:
            self.debugger.debug(args)

        # Verify json.load was called
        mock_json_load.assert_called_once()
        mock_open_file.assert_called_once_with(args.cluster_file)

        # Verify resolve_placeholders was called
        mock_resolve_placeholders.assert_called_once_with(cluster_dict)

        # Verify Pssh was created correctly
        mock_pssh_class.assert_called_once_with(
            unittest.mock.ANY,  # log
            ['192.168.1.1', '192.168.1.2'],  # node_list
            user='testuser',
            pkey='/path/to/key',
            stop_on_errors=False,
            timeout=30,  # Now includes timeout parameter
        )

        # Verify ps command was executed
        ps_cmd = "pgrep -f 'mpirun'"
        mock_pssh_instance.exec.assert_any_call(ps_cmd, timeout=10)

        # Verify gdb commands were executed via exec_cmd_list
        gdb_cmds = [
            "timeout 30 sudo gdb -p 1234 --batch -ex 'thread apply all bt' 2>&1 || echo 'GDB_ERROR'",
            "timeout 30 sudo gdb -p 5678 --batch -ex 'thread apply all bt' 2>&1 || echo 'GDB_ERROR'",
            "timeout 30 sudo gdb -p 5678 --batch -ex 'thread apply all bt' 2>&1 || echo 'GDB_ERROR'",
        ]
        mock_pssh_instance.exec_cmd_list.assert_called_once_with(gdb_cmds, timeout=10)

        # Verify output was printed
        mock_print.assert_called()

    @patch("cvs.debuggers.gdb_backtrace_collector.resolve_cluster_config_placeholders")
    @patch("cvs.debuggers.gdb_backtrace_collector.parallel_ssh_lib.Pssh")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_debug_with_specific_pid(self, mock_json_load, mock_open_file, mock_pssh_class, mock_resolve_placeholders):
        """Test debugging with specific PID and node"""
        # Mock cluster config
        cluster_dict = {'node_dict': {'node1': '192.168.1.1'}, 'username': 'testuser', 'password': 'testpass'}
        mock_json_load.return_value = cluster_dict
        mock_resolve_placeholders.return_value = cluster_dict

        # Mock initial PSSH instance for initialization
        mock_pssh_initial = MagicMock()
        mock_pssh_initial.reachable_hosts = ['192.168.1.1']
        mock_pssh_initial.user = 'testuser'
        mock_pssh_initial.password = 'testpass'
        mock_pssh_initial.pkey = None

        # Mock target PSSH instance for specific node
        mock_pssh_target = MagicMock()
        mock_pssh_target.reachable_hosts = ['192.168.1.1']

        # Configure mock_pssh_class to return different instances on consecutive calls
        mock_pssh_class.side_effect = [mock_pssh_initial, mock_pssh_target]

        # Mock gdb result
        gdb_results = {'192.168.1.1': ['#0 in main()\n']}  # Dict format for exec_cmd_list
        mock_pssh_target.exec_cmd_list.return_value = gdb_results

        args = MagicMock()
        args.cluster_file = "/path/to/cluster.json"
        args.process_filter = None
        args.pid = 1234
        args.node = '192.168.1.1'
        args.output_format = "console"
        args.timeout = 30

        with patch("builtins.print"):
            self.debugger.debug(args)

        # Verify Pssh was created twice: once for initialization, once for target node
        self.assertEqual(mock_pssh_class.call_count, 2)

        # First call: initialization
        mock_pssh_class.assert_any_call(
            unittest.mock.ANY,  # log
            ['192.168.1.1'],  # node_list
            user='testuser',
            password='testpass',
            stop_on_errors=False,
            timeout=30,  # Now uses the timeout parameter
        )

        # Second call: target node execution
        mock_pssh_class.assert_any_call(
            unittest.mock.ANY,  # log
            ['192.168.1.1'],  # target node only
            user='testuser',
            password='testpass',
            stop_on_errors=False,
            timeout=30,  # Now uses the timeout parameter
        )


if __name__ == "__main__":
    unittest.main()
