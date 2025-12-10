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


class TestScpFunction(unittest.TestCase):
    @patch('cvs.lib.parallel_ssh_lib.SCPClient')
    @patch('cvs.lib.parallel_ssh_lib.paramiko.SSHClient')
    def test_scp_success(self, mock_ssh_client, mock_scp_client):
        mock_ssh = MagicMock()
        mock_ssh_client.return_value = mock_ssh
        mock_scp = MagicMock()
        mock_scp_client.return_value = mock_scp

        from cvs.lib.parallel_ssh_lib import scp

        scp('src.txt', 'dst.txt', 'user', 'pass')

        mock_ssh.connect.assert_called()
        mock_scp.put.assert_called_with('src.txt', 'dst.txt')
        mock_scp.close.assert_called()


if __name__ == "__main__":
    unittest.main()
