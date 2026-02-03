import unittest
from unittest.mock import patch, MagicMock
from cvs.lib.parallel_ssh_lib import Pssh


class TestPsshExec(unittest.TestCase):
    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.mock_pssh_client = mock_pssh_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")

    def test_exec_successful(self):
        # Test: Execute command successfully on all hosts
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["output1 line1", "output1 line2"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["output2 line1"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec("echo hello")

        self.mock_client.run_command.assert_called_once_with("echo hello", stop_on_errors=True)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("output1 line1", result["host1"])
        self.assertIn("output2 line1", result["host2"])

    def test_exec_with_connection_error_stop_on_errors_true(self):
        # Test: Handle exceptions with stop_on_errors=True (default)
        # Exception should be raised, and no result returned (no partial results)
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        # With stop_on_errors=True, run_command raises on exception, no result returned
        with self.assertRaises(ConnectionError) as cm:
            result = self.pssh.exec("echo hello")  # This should raise, so result is not assigned

        self.assertIn("Connection failed", str(cm.exception))
        # Since exception was raised, result was not returned
        self.assertNotIn("result", locals())

    @patch.object(Pssh, "check_connectivity")
    def test_exec_with_connection_error_stop_on_errors_false(self, mock_check_connectivity):
        # Test Case 2.2: Execute command with connection error and stop_on_errors=False
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.pssh.stop_on_errors = False
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        self.mock_check_connectivity = mock_check_connectivity
        self.mock_check_connectivity.return_value = []  # No pruning

        result = self.pssh.exec("echo hello", timeout=10)

        self.mock_client.run_command.assert_called_once_with("echo hello", read_timeout=10, stop_on_errors=False)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Connection failed", result["host2"])

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_with_pruning_unreachable_host(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False,  on host2, and check_connectivity fails for host2, prune it
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = ["host2"]  # Simulate unreachable

        result = self.pssh.exec("echo hello", timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(self.pssh.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        # Client should be recreated once (init + prune)
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_no_pruning_when_reachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False, timeout on host2, but check_connectivity succeeds, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        from pssh.exceptions import Timeout

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout("Command timed out")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = []  # Always reachable

        result = self.pssh.exec("echo hello", timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1", "host2"])  # No change
        self.assertEqual(self.pssh.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_pruning_with_multiple_unreachable_hosts(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False, multiple hosts (host2, host3) timeout and are unreachable, prune all
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2", "host3"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        mock_output3 = MagicMock()
        mock_output3.host = "host3"
        mock_output3.stdout = []
        mock_output3.stderr = []
        mock_output3.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2, mock_output3]
        mock_check_connectivity.return_value = ["host2", "host3"]  # Simulate all unreachable

        result = self.pssh.exec("echo hello", timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(sorted(self.pssh.unreachable_hosts), ["host2", "host3"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host3", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(result["host3"], "Connection failed\n\nABORT: Host Unreachable Error")
        # Client should be recreated once (init + prune)
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch.object(Pssh, "check_connectivity")
    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    def test_exec_no_pruning_on_timeout_exception_reachable(self, mock_pssh_client, mock_check_connectivity):
        # Test: exec with timeout exception, no pruning if host is reachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        from pssh.exceptions import Timeout

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout("Command timed out")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = []  # No pruning

        result = self.pssh.exec("echo hello", timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1", "host2"])  # No pruning
        self.assertEqual(self.pssh.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch.object(Pssh, "check_connectivity")
    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    def test_exec_pruning_on_timeout_exception_unreachable(self, mock_pssh_client, mock_check_connectivity):
        # Test: exec with timeout exception, pruning occurs if host unreachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        from pssh.exceptions import Timeout

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout("Command timed out")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = ["host2"]  # Simulate unreachable

        result = self.pssh.exec("echo hello", timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(self.pssh.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(
            result["host2"], "Command timed out\nABORT: Timeout Error in Host: host2\n\nABORT: Host Unreachable Error"
        )
        # Client recreated after pruning
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch.object(Pssh, "prune_unreachable_hosts")
    @patch.object(Pssh, "inform_unreachability")
    def test_exec_no_pruning_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: With stop_on_errors=True, no pruning even with connection error
        # Since stop_on_errors=True, run_command raises immediately, so prune_unreachable_hosts and inform_unreachability are not invoked
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.pssh.exec("echo hello", timeout=10)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch.object(Pssh, "prune_unreachable_hosts")
    @patch.object(Pssh, "inform_unreachability")
    def test_exec_timeout_exception_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: With stop_on_errors=True, Timeout exception is re-raised
        from pssh.exceptions import Timeout

        self.mock_client.run_command.side_effect = Timeout("Command timed out")

        with self.assertRaises(Timeout):
            self.pssh.exec("echo hello", timeout=10)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch("builtins.print")
    def test_exec_print_console_false(self, mock_print):
        # Test: Execute command with print_console=False, verify output lines are not printed
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["output1 line1", "output1 line2"]
        mock_output1.stderr = ["error line1"]
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["output2 line1"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec("echo hello", print_console=False)

        # Verify output is collected correctly
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("output1 line1", result["host1"])
        self.assertIn("output1 line2", result["host1"])
        self.assertIn("error line1", result["host1"])
        self.assertIn("output2 line1", result["host2"])

        # Verify stdout/stderr lines are NOT printed (only headers and command are printed)
        printed_calls = [str(call) for call in mock_print.call_args_list]
        for call in printed_calls:
            # These output lines should NOT be printed
            self.assertNotIn("output1 line1", call)
            self.assertNotIn("output1 line2", call)
            self.assertNotIn("error line1", call)
            self.assertNotIn("output2 line1", call)


class TestPsshExecCmdList(unittest.TestCase):
    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.mock_pssh_client = mock_pssh_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")

    def test_exec_cmd_list_successful(self):
        # Test: Execute different commands on different hosts successfully
        cmd_list = ["echo host1", "echo host2"]
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["host1"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["host2"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec_cmd_list(cmd_list)

        self.mock_client.run_command.assert_called_once_with("%s", host_args=cmd_list, stop_on_errors=True)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host1", result["host1"])
        self.assertIn("host2", result["host2"])

    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_with_connection_error_stop_on_errors_false(self, mock_check_connectivity):
        # Test: Handle exceptions with stop_on_errors=False for exec_cmd_list
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.pssh.stop_on_errors = False
        cmd_list = ["echo success", "echo fail"]
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        self.mock_check_connectivity = mock_check_connectivity
        self.mock_check_connectivity.return_value = []  # Simulate reachable, no pruning

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.mock_client.run_command.assert_called_once_with(
            "%s", host_args=cmd_list, read_timeout=10, stop_on_errors=False
        )
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertIn("Connection failed", result["host2"])

    def test_exec_cmd_list_with_connection_error_stop_on_errors_true(self):
        # Test: Handle exceptions with stop_on_errors=True for exec_cmd_list
        # Exception should be raised, and no result returned (no partial results)
        cmd_list = ["echo test"]
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError) as cm:
            result = self.pssh.exec_cmd_list(cmd_list, timeout=5)

        self.assertIn("Connection failed", str(cm.exception))
        self.assertNotIn("result", locals())

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_no_pruning_when_reachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with stop_on_errors=False, timeout on host2, but check_connectivity succeeds, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        cmd_list = ["echo success", "echo fail"]
        from pssh.exceptions import Timeout

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout("Command timed out")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = []  # Always reachable

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1", "host2"])  # No change
        self.assertEqual(self.pssh.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_pruning_on_timeout_exception_unreachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with timeout exception, pruning occurs if host unreachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        cmd_list = ["echo success", "echo fail"]
        from pssh.exceptions import Timeout

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout("Command timed out")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = ["host2"]  # Simulate unreachable

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(self.pssh.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(
            result["host2"], "Command timed out\nABORT: Timeout Error in Host: host2\n\nABORT: Host Unreachable Error"
        )
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_with_pruning(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        cmd_list = ["echo success", "echo fail"]
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = ["host2"]

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(self.pssh.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_pruning_with_multiple_unreachable_hosts(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with pruning for multiple unreachable hosts
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2", "host3"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        cmd_list = ["echo success", "echo fail1", "echo fail2"]
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        mock_output3 = MagicMock()
        mock_output3.host = "host3"
        mock_output3.stdout = []
        mock_output3.stderr = []
        mock_output3.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2, mock_output3]
        mock_check_connectivity.return_value = ["host2", "host3"]

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1"])
        self.assertEqual(sorted(self.pssh.unreachable_hosts), ["host2", "host3"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host3", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(result["host3"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    @patch.object(Pssh, "check_connectivity")
    def test_exec_cmd_list_no_pruning_on_connection_error_when_reachable(
        self, mock_check_connectivity, mock_pssh_client
    ):
        # Test: exec_cmd_list with ConnectionError exception, but check_connectivity succeeds, no pruning occurs
        # ConnectionError exceptions are checked for reachability, and if reachable, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")
        self.pssh.stop_on_errors = False
        self.pssh.check_connectivity = mock_check_connectivity
        cmd_list = ["echo success", "echo fail"]
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = []  # Simulate reachable, no pruning

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.pssh.reachable_hosts, ["host1", "host2"])  # No pruning
        self.assertEqual(self.pssh.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertIn("Connection failed", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch.object(Pssh, "prune_unreachable_hosts")
    @patch.object(Pssh, "inform_unreachability")
    def test_exec_cmd_list_no_pruning_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: exec_cmd_list with stop_on_errors=True, no pruning even with connection error
        # Since stop_on_errors=True, run_command raises immediately, so prune_unreachable_hosts and inform_unreachability are not invoked
        cmd_list = ["echo test"]
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.pssh.exec_cmd_list(cmd_list, timeout=5)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch.object(Pssh, "prune_unreachable_hosts")
    @patch.object(Pssh, "inform_unreachability")
    def test_exec_cmd_list_timeout_exception_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: exec_cmd_list with stop_on_errors=True, Timeout exception is re-raised
        cmd_list = ["echo test"]
        from pssh.exceptions import Timeout

        self.mock_client.run_command.side_effect = Timeout("Command timed out")

        with self.assertRaises(Timeout):
            self.pssh.exec_cmd_list(cmd_list, timeout=5)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch("builtins.print")
    def test_exec_cmd_list_print_console_false(self, mock_print):
        # Test: Execute command list with print_console=False, verify output lines are not printed
        cmd_list = ["echo host1", "echo host2"]
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["host1 output line1", "host1 output line2"]
        mock_output1.stderr = ["host1 error line1"]
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["host2 output line1"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec_cmd_list(cmd_list, print_console=False)

        # Verify output is collected correctly
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host1 output line1", result["host1"])
        self.assertIn("host1 output line2", result["host1"])
        self.assertIn("host1 error line1", result["host1"])
        self.assertIn("host2 output line1", result["host2"])

        # Verify stdout/stderr lines are NOT printed (only headers and commands are printed)
        printed_calls = [str(call) for call in mock_print.call_args_list]
        for call in printed_calls:
            # These output lines should NOT be printed
            self.assertNotIn("host1 output line1", call)
            self.assertNotIn("host1 output line2", call)
            self.assertNotIn("host1 error line1", call)
            self.assertNotIn("host2 output line1", call)


class TestSafeIterator(unittest.TestCase):
    @patch("cvs.lib.parallel_ssh_lib.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1"]
        self.pssh = Pssh("log", self.host_list, user="user", password="pass")

    def test_safe_iterator_with_valid_lines(self):
        # Test: All lines are valid UTF-8, should yield all lines
        valid_lines = ["line1", "line2", "line3"]
        result = list(self.pssh._safe_iterator(iter(valid_lines)))
        self.assertEqual(result, valid_lines)

    @patch('builtins.print')
    def test_safe_iterator_with_unicode_error(self, mock_print):
        # Test: Iterator raises UnicodeDecodeError, should skip and continue

        class SimpleIterator:
            def __init__(self):
                self.data = [b'line1\n', b'bad\x96utf8\n', b'line3\n']
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                line_bytes = self.data[self.index]
                self.index += 1
                return line_bytes.decode('utf-8').strip()

        result = list(self.pssh._safe_iterator(SimpleIterator()))

        # Should get lines before and after the bad byte
        self.assertEqual(result, ["line1", "line3"])

        # Should have printed warning
        mock_print.assert_called()
        warning_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("UnicodeDecodeError" in call for call in warning_calls))

    @patch('builtins.print')
    def test_safe_iterator_with_multiple_unicode_errors(self, mock_print):
        # Test: Iterator that simulates multiple UnicodeDecodeErrors during iteration

        class ByteLineIterator:
            """Simulates pssh iterator that decodes bytes line-by-line"""

            def __init__(self):
                # Mix of valid UTF-8 and invalid bytes
                self.data = [
                    b'good1\n',  # Valid UTF-8
                    b'bad\x96line\n',  # Invalid UTF-8 (0x96 not valid in UTF-8)
                    b'good2\n',  # Valid UTF-8
                    b'\xff\xfe bad\n',  # Invalid UTF-8 (0xff, 0xfe not valid start bytes)
                    b'good3\n',  # Valid UTF-8
                ]
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                line_bytes = self.data[self.index]
                self.index += 1
                # Decode like pssh does - raises UnicodeDecodeError on invalid UTF-8
                return line_bytes.decode('utf-8').strip()

        result = list(self.pssh._safe_iterator(ByteLineIterator()))

        # Should get only the valid UTF-8 lines
        self.assertEqual(result, ["good1", "good2", "good3"])

        # Should have printed warnings for both decode errors
        self.assertEqual(mock_print.call_count, 2)

    def test_safe_iterator_with_empty_iterator(self):
        # Test: Empty iterator should return empty list
        empty_iter = iter([])
        result = list(self.pssh._safe_iterator(empty_iter))
        self.assertEqual(result, [])

    @patch('builtins.print')
    def test_safe_iterator_integration_with_process_output(self, mock_print):
        # Test: Integration test - _safe_iterator used in _process_output handles UnicodeDecodeError

        class ByteLineIterator:
            """Simulates pssh stdout with mixed valid/invalid UTF-8"""

            def __init__(self):
                self.data = [
                    b'valid line 1\n',
                    b'tqdm\x96progress\n',  # Invalid UTF-8 like from tqdm progress bar
                    b'valid line 2\n',
                ]
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                line_bytes = self.data[self.index]
                self.index += 1
                return line_bytes.decode('utf-8').strip()

        mock_output = MagicMock()
        mock_output.host = "host1"
        mock_output.stdout = ByteLineIterator()
        mock_output.stderr = []
        mock_output.exception = None

        result = self.pssh._process_output([mock_output], cmd="test command", print_console=False)

        # Should have collected valid lines only
        self.assertIn("host1", result)
        self.assertIn("valid line 1", result["host1"])
        self.assertIn("valid line 2", result["host1"])
        self.assertNotIn("tqdm", result["host1"])

        # Should have printed warning for UnicodeDecodeError
        warning_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("UnicodeDecodeError" in call for call in warning_calls))


if __name__ == "__main__":
    unittest.main()
