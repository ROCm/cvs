import unittest
from unittest.mock import patch, MagicMock
from cvs.lib.parallel.phandle import ParallelHandle  # Test basic ParallelHandle class directly


class TestParallelHandleExec(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("cvs.lib.parallel.phandle.ParallelSSHClient")
        self.mock_pssh_client = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.mock_client = MagicMock()
        self.mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.log = self.mock_log

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

        result = self.handle.exec("echo hello")

        self.mock_client.run_command.assert_called_once_with("echo hello", stop_on_errors=True)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("output1 line1", result["host1"])
        self.assertIn("output2 line1", result["host2"])

    def test_exec_retries_once_on_session_error(self):
        from pssh.exceptions import SessionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["ok1"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["ok2"]
        mock_output2.stderr = []
        mock_output2.exception = None

        stale = SessionError("stale session")
        self.mock_client.run_command.side_effect = [stale, [mock_output1, mock_output2]]

        result = self.handle.exec("echo hello")

        self.assertEqual(self.mock_client.run_command.call_count, 2)
        self.assertEqual(self.mock_pssh_client.call_count, 2)
        self.assertIn("ok1", result["host1"])
        self.assertIn("ok2", result["host2"])
        self.mock_log.info.assert_any_call(
            "ParallelSSH: SessionError on first attempt; recreating client and retrying once (%s).",
            stale,
        )
        self.mock_log.debug.assert_any_call("ParallelSSH session retry detail", exc_info=True)

    def test_exec_session_retry_destroys_stale_client_before_recreating(self):
        # The stale client's greenlets/transports must be torn down before a
        # replacement is built, or the old sshd sessions leak.
        from pssh.exceptions import SessionError

        self.mock_client.run_command.side_effect = [SessionError("stale session"), []]
        with patch.object(self.handle, "destroy_clients", wraps=self.handle.destroy_clients) as mock_destroy:
            self.handle.exec("echo hello")

        mock_destroy.assert_called_once()
        self.assertEqual(self.mock_pssh_client.call_count, 2)

    def test_exec_retries_once_on_session_error_with_timeout(self):
        from pssh.exceptions import SessionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["ok"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["ok"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.side_effect = [
            SessionError("stale session"),
            [mock_output1, mock_output2],
        ]

        self.handle.exec("slow_cmd", timeout=120)

        self.assertEqual(self.mock_client.run_command.call_count, 2)
        self.mock_client.run_command.assert_called_with("slow_cmd", read_timeout=120, stop_on_errors=True)

    def test_exec_propagates_session_error_after_failed_retry(self):
        from pssh.exceptions import SessionError

        self.mock_client.run_command.side_effect = [
            SessionError("first"),
            SessionError("second"),
        ]

        with self.assertRaises(SessionError) as cm:
            self.handle.exec("echo hello")

        self.assertIn("second", str(cm.exception))
        self.assertEqual(self.mock_client.run_command.call_count, 2)

    def test_exec_with_connection_error_stop_on_errors_true(self):
        # Test: Handle exceptions with stop_on_errors=True (default)
        # Exception should be raised, and no result returned (no partial results)
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        # With stop_on_errors=True, run_command raises on exception, no result returned
        with self.assertRaises(ConnectionError) as cm:
            result = self.handle.exec("echo hello")  # This should raise, so result is not assigned

        self.assertIn("Connection failed", str(cm.exception))
        # Since exception was raised, result was not returned
        self.assertNotIn("result", locals())

    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_with_connection_error_stop_on_errors_false(self, mock_check_connectivity):
        # Test Case 2.2: Execute command with connection error and stop_on_errors=False
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.handle.stop_on_errors = False
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

        result = self.handle.exec("echo hello", timeout=10)

        self.mock_client.run_command.assert_called_once_with("echo hello", read_timeout=10, stop_on_errors=False)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Connection failed", result["host2"])

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_with_pruning_unreachable_host(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False,  on host2, and check_connectivity fails for host2, prune it
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec("echo hello", timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(self.handle.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        # Client should be recreated once (init + prune)
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_no_pruning_when_reachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False, timeout on host2, but check_connectivity succeeds, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec("echo hello", timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1", "host2"])  # No change
        self.assertEqual(self.handle.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_pruning_with_multiple_unreachable_hosts(self, mock_check_connectivity, mock_pssh_client):
        # Test: With stop_on_errors=False, multiple hosts (host2, host3) timeout and are unreachable, prune all
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2", "host3"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec("echo hello", timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(sorted(self.handle.unreachable_hosts), ["host2", "host3"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host3", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(result["host3"], "Connection failed\n\nABORT: Host Unreachable Error")
        # Client should be recreated once (init + prune)
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch.object(ParallelHandle, "check_connectivity")
    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def test_exec_no_pruning_on_timeout_exception_reachable(self, mock_pssh_client, mock_check_connectivity):
        # Test: exec with timeout exception, no pruning if host is reachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
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

        result = self.handle.exec("echo hello", timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1", "host2"])  # No pruning
        self.assertEqual(self.handle.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch.object(ParallelHandle, "check_connectivity")
    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def test_exec_pruning_on_timeout_exception_unreachable(self, mock_pssh_client, mock_check_connectivity):
        # Test: exec with timeout exception, pruning occurs if host unreachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
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

        result = self.handle.exec("echo hello", timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(self.handle.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success output", result["host1"])
        self.assertEqual(
            result["host2"], "Command timed out\nABORT: Timeout Error in Host: host2\n\nABORT: Host Unreachable Error"
        )
        # Client recreated after pruning
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch.object(ParallelHandle, "prune_unreachable_hosts")
    @patch.object(ParallelHandle, "inform_unreachability")
    def test_exec_no_pruning_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: With stop_on_errors=True, no pruning even with connection error
        # Since stop_on_errors=True, run_command raises immediately, so prune_unreachable_hosts and inform_unreachability are not invoked
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.handle.exec("echo hello", timeout=10)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch.object(ParallelHandle, "prune_unreachable_hosts")
    @patch.object(ParallelHandle, "inform_unreachability")
    def test_exec_timeout_exception_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: With stop_on_errors=True, Timeout exception is re-raised
        from pssh.exceptions import Timeout

        self.mock_client.run_command.side_effect = Timeout("Command timed out")

        with self.assertRaises(Timeout):
            self.handle.exec("echo hello", timeout=10)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch.object(ParallelHandle, "prune_unreachable_hosts")
    @patch.object(ParallelHandle, "inform_unreachability")
    def test_exec_stdout_timeout_during_iteration_stop_on_errors_false(self, mock_inform, mock_prune):
        # Regression: when Timeout is raised mid-stdout-iteration AND stop_on_errors=False,
        # _process_output must call _handle_timeout_exception (which got accidentally removed
        # in an earlier refactor). Without the helper restored, this raises AttributeError.
        from pssh.exceptions import Timeout

        self.handle.stop_on_errors = False

        timeout_error = Timeout("Read timed out")

        # host1: stdout iteration raises Timeout mid-stream
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        # Iterating stdout raises Timeout
        mock_output1.stdout = MagicMock()
        mock_output1.stdout.__iter__ = MagicMock(side_effect=timeout_error)
        mock_output1.stderr = []
        mock_output1.exception = None

        # host2: clean run, no exception yet
        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["ok"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        # Must NOT raise AttributeError (the bug we are guarding against).
        # The ParallelHandle code catches Timeout when stop_on_errors=False and routes through
        # _handle_timeout_exception, which propagates the timeout to all items so the
        # subsequent item.exception handling block formats and records it.
        result = self.handle.exec("echo hello", timeout=10)

        # _handle_timeout_exception must populate item.exception on items that had None
        # so the subsequent formatting block records the timeout for both hosts.
        self.assertIs(mock_output1.exception, timeout_error)
        self.assertIs(mock_output2.exception, timeout_error)
        self.assertIn("host1", result)
        self.assertIn("host2", result)

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

        result = self.handle.exec("echo hello", print_console=False)

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

    def test_exec_detailed_true_successful(self):
        # Test: Execute command successfully with detailed=True
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["output1 line1", "output1 line2"]
        mock_output1.stderr = ["error1 line1"]
        mock_output1.exception = None
        mock_output1.exit_code = 0

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["output2 line1"]
        mock_output2.stderr = []
        mock_output2.exception = None
        mock_output2.exit_code = 0

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.handle.exec("echo hello", detailed=True)

        self.assertIn("host1", result)
        self.assertIn("host2", result)

        # Check structure
        self.assertIsInstance(result["host1"], dict)
        self.assertIsInstance(result["host2"], dict)
        self.assertIn("output", result["host1"])
        self.assertIn("exit_code", result["host1"])
        self.assertIn("output", result["host2"])
        self.assertIn("exit_code", result["host2"])

        # Check content
        self.assertIn("output1 line1", result["host1"]["output"])
        self.assertIn("output1 line2", result["host1"]["output"])
        self.assertIn("error1 line1", result["host1"]["output"])
        self.assertIn("output2 line1", result["host2"]["output"])
        self.assertEqual(result["host1"]["exit_code"], 0)
        self.assertEqual(result["host2"]["exit_code"], 0)

    def test_exec_detailed_true_with_exit_code_failure(self):
        # Test: Execute command with non-zero exit code and detailed=True
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None
        mock_output1.exit_code = 0

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = ["command failed"]
        mock_output2.exception = None
        mock_output2.exit_code = 1

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.handle.exec("failing command", detailed=True)

        self.assertEqual(result["host1"]["exit_code"], 0)
        self.assertEqual(result["host2"]["exit_code"], 1)
        self.assertIn("success output", result["host1"]["output"])
        self.assertIn("command failed", result["host2"]["output"])

    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_detailed_true_with_exception(self, mock_check_connectivity):
        # Test: Execute command with exception and detailed=True
        self.handle.stop_on_errors = False
        from pssh.exceptions import ConnectionError

        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["success output"]
        mock_output1.stderr = []
        mock_output1.exception = None
        mock_output1.exit_code = 0

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = ConnectionError("Connection failed")
        mock_output2.exit_code = None  # No exit code available for exceptions

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]
        mock_check_connectivity.return_value = []  # No pruning

        result = self.handle.exec("echo hello", detailed=True)

        self.assertEqual(result["host1"]["exit_code"], 0)
        self.assertEqual(result["host2"]["exit_code"], -1)  # -1 for exceptions
        self.assertIn("success output", result["host1"]["output"])
        self.assertIn("Connection failed", result["host2"]["output"])

    def test_process_output_normalizes_none_exit_code(self):
        # Regression (L3): when the SSH channel has not reached EOF,
        # parallel-ssh's HostOutput.exit_code property returns None
        # (pssh/clients/native/single.py: get_exit_status returns None
        # when not channel.eof()). The previous implementation used
        # getattr(item, 'exit_code', -1) -- but the -1 default only
        # fires on AttributeError, never when the property returns None.
        # Result: cmd_output[host]['exit_code'] could be None, breaking
        # downstream consumers in docker.py/container.py that compare
        # exit_code against 0 (None != 0 -> spurious failure;
        # None == 0 -> spurious success).
        # The contract is exit_code: int, with -1 meaning "unknown/aborted".
        mock_output = MagicMock()
        mock_output.host = "host1"
        mock_output.stdout = ["partial output"]
        mock_output.stderr = []
        mock_output.exception = None
        mock_output.exit_code = None  # channel not EOF'd yet

        self.mock_client.run_command.return_value = [mock_output]

        result = self.handle.exec("slow command", detailed=True)

        # Must be -1 (the documented "unknown" sentinel), not None.
        self.assertEqual(result["host1"]["exit_code"], -1)
        self.assertIsNotNone(result["host1"]["exit_code"])

    def test_exec_detailed_false_backward_compatibility(self):
        # Test: detailed=False (default) maintains backward compatibility
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["output1 line1"]
        mock_output1.stderr = []
        mock_output1.exception = None
        mock_output1.exit_code = 0

        self.mock_client.run_command.return_value = [mock_output1]

        result = self.handle.exec("echo hello", detailed=False)

        # Should return string, not dict
        self.assertIsInstance(result["host1"], str)
        self.assertNotIsInstance(result["host1"], dict)
        self.assertIn("output1 line1", result["host1"])

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def test_init_does_not_alias_caller_host_list(self, mock_pssh_client):
        # Regression (B2): ParallelHandle.__init__ must not alias the caller's host_list.
        # Before the fix, self.reachable_hosts and self.host_list both pointed to
        # the caller's list object, so prune_unreachable_hosts (which calls
        # self.reachable_hosts.remove(...)) silently mutated the caller's list.
        # This bit BaremetalOrchestrator on the stop_on_errors=False path:
        # a single transient ConnectionError/Timeout/SessionError permanently
        # shrunk the orchestrator's view of the cluster.
        mock_pssh_client.return_value = MagicMock()
        original = ["a", "b", "c"]
        handle = ParallelHandle(self.mock_log, original, user="user", password="pass")
        # Simulate what prune_unreachable_hosts does internally.
        handle.reachable_hosts.remove("b")
        # (i) Caller's list must be untouched.
        self.assertEqual(original, ["a", "b", "c"])
        # (ii) Internal reachable view reflects the prune.
        self.assertEqual(handle.reachable_hosts, ["a", "c"])
        # (iii) host_list is a stable snapshot of the original input.
        self.assertEqual(handle.host_list, ["a", "b", "c"])


class TestParallelHandleExecCmdList(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("cvs.lib.parallel.phandle.ParallelSSHClient")
        self.mock_pssh_client = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.mock_client = MagicMock()
        self.mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.log = self.mock_log

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

        result = self.handle.exec_cmd_list(cmd_list)

        self.mock_client.run_command.assert_called_once_with("%s", host_args=cmd_list, stop_on_errors=True)
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host1", result["host1"])
        self.assertIn("host2", result["host2"])

    def test_exec_cmd_list_retries_once_on_session_error(self):
        from pssh.exceptions import SessionError

        cmd_list = ["echo host1", "echo host2"]
        mock_output1 = MagicMock()
        mock_output1.host = "host1"
        mock_output1.stdout = ["h1"]
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = "host2"
        mock_output2.stdout = ["h2"]
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.side_effect = [
            SessionError("stale session"),
            [mock_output1, mock_output2],
        ]

        result = self.handle.exec_cmd_list(cmd_list)

        self.assertEqual(self.mock_client.run_command.call_count, 2)
        self.assertEqual(self.mock_pssh_client.call_count, 2)
        self.assertIn("h1", result["host1"])
        self.assertIn("h2", result["host2"])

    def test_exec_cmd_list_propagates_session_error_after_failed_retry(self):
        from pssh.exceptions import SessionError

        cmd_list = ["echo a", "echo b"]
        self.mock_client.run_command.side_effect = [
            SessionError("first"),
            SessionError("second"),
        ]

        with self.assertRaises(SessionError) as cm:
            self.handle.exec_cmd_list(cmd_list)

        self.assertIn("second", str(cm.exception))
        self.assertEqual(self.mock_client.run_command.call_count, 2)

    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_with_connection_error_stop_on_errors_false(self, mock_check_connectivity):
        # Test: Handle exceptions with stop_on_errors=False for exec_cmd_list
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.handle.stop_on_errors = False
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

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
            result = self.handle.exec_cmd_list(cmd_list, timeout=5)

        self.assertIn("Connection failed", str(cm.exception))
        self.assertNotIn("result", locals())

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_no_pruning_when_reachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with stop_on_errors=False, timeout on host2, but check_connectivity succeeds, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1", "host2"])  # No change
        self.assertEqual(self.handle.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertIn("Command timed out", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_pruning_on_timeout_exception_unreachable(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with timeout exception, pruning occurs if host unreachable
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(self.handle.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(
            result["host2"], "Command timed out\nABORT: Timeout Error in Host: host2\n\nABORT: Host Unreachable Error"
        )
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_with_pruning(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(self.handle.unreachable_hosts, ["host2"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_pruning_with_multiple_unreachable_hosts(self, mock_check_connectivity, mock_pssh_client):
        # Test: exec_cmd_list with pruning for multiple unreachable hosts
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2", "host3"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1"])
        self.assertEqual(sorted(self.handle.unreachable_hosts), ["host2", "host3"])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("host3", result)
        self.assertIn("success", result["host1"])
        self.assertEqual(result["host2"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(result["host3"], "Connection failed\n\nABORT: Host Unreachable Error")
        self.assertEqual(mock_pssh_client.call_count, 2)

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    @patch.object(ParallelHandle, "check_connectivity")
    def test_exec_cmd_list_no_pruning_on_connection_error_when_reachable(
        self, mock_check_connectivity, mock_pssh_client
    ):
        # Test: exec_cmd_list with ConnectionError exception, but check_connectivity succeeds, no pruning occurs
        # ConnectionError exceptions are checked for reachability, and if reachable, no pruning
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")
        self.handle.stop_on_errors = False
        self.handle.check_connectivity = mock_check_connectivity
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

        result = self.handle.exec_cmd_list(cmd_list, timeout=10)

        self.assertEqual(self.handle.reachable_hosts, ["host1", "host2"])  # No pruning
        self.assertEqual(self.handle.unreachable_hosts, [])
        self.assertIn("host1", result)
        self.assertIn("host2", result)
        self.assertIn("success", result["host1"])
        self.assertIn("Connection failed", result["host2"])  # Original exception
        # Client not recreated
        self.assertEqual(mock_pssh_client.call_count, 1)

    @patch.object(ParallelHandle, "prune_unreachable_hosts")
    @patch.object(ParallelHandle, "inform_unreachability")
    def test_exec_cmd_list_no_pruning_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: exec_cmd_list with stop_on_errors=True, no pruning even with connection error
        # Since stop_on_errors=True, run_command raises immediately, so prune_unreachable_hosts and inform_unreachability are not invoked
        cmd_list = ["echo test"]
        from pssh.exceptions import ConnectionError

        self.mock_client.run_command.side_effect = ConnectionError("Connection failed")

        with self.assertRaises(ConnectionError):
            self.handle.exec_cmd_list(cmd_list, timeout=5)

        # Assert that pruning methods were not called
        mock_prune.assert_not_called()
        mock_inform.assert_not_called()

    @patch.object(ParallelHandle, "prune_unreachable_hosts")
    @patch.object(ParallelHandle, "inform_unreachability")
    def test_exec_cmd_list_timeout_exception_when_stop_on_errors_true(self, mock_inform, mock_prune):
        # Test: exec_cmd_list with stop_on_errors=True, Timeout exception is re-raised
        cmd_list = ["echo test"]
        from pssh.exceptions import Timeout

        self.mock_client.run_command.side_effect = Timeout("Command timed out")

        with self.assertRaises(Timeout):
            self.handle.exec_cmd_list(cmd_list, timeout=5)

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

        result = self.handle.exec_cmd_list(cmd_list, print_console=False)

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


class TestParallelHandleFileTransfer(unittest.TestCase):
    """
    Unit tests for upload_file / download_file / scp_file.

    Strategy: mock ParallelSSHClient.copy_file and copy_remote_file. Each call
    returns a list of greenlet-like objects whose .get() either returns or
    raises. We verify the {host: local_path} dict contract and the IOError
    aggregation message format on partial/full failure.
    """

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1", "host2"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")

    def _ok_greenlet(self):
        g = MagicMock()
        g.get.return_value = None
        return g

    def _fail_greenlet(self, exc):
        g = MagicMock()
        g.get.side_effect = exc
        return g

    # -------------------- upload_file --------------------

    def test_upload_file_success_multi_host(self):
        # Both hosts succeed -> no exception, copy_file called with right args
        self.mock_client.copy_file.return_value = [self._ok_greenlet(), self._ok_greenlet()]

        self.handle.upload_file("/tmp/local.json", "/remote/dest.json")

        self.mock_client.copy_file.assert_called_once_with("/tmp/local.json", "/remote/dest.json", recurse=False)
        self.mock_client.pool.join.assert_called_once()

    def test_upload_file_recurse_passes_through(self):
        # recurse=True must be propagated to copy_file
        self.mock_client.copy_file.return_value = [self._ok_greenlet(), self._ok_greenlet()]

        self.handle.upload_file("/tmp/dir", "/remote/dir", recurse=True)

        self.mock_client.copy_file.assert_called_once_with("/tmp/dir", "/remote/dir", recurse=True)

    def test_upload_file_partial_failure_raises_ioerror(self):
        # host1 ok, host2 raises -> IOError listing offending host
        boom = IOError("permission denied")
        self.mock_client.copy_file.return_value = [self._ok_greenlet(), self._fail_greenlet(boom)]

        with self.assertRaises(IOError) as cm:
            self.handle.upload_file("/tmp/local.json", "/remote/dest.json")

        msg = str(cm.exception)
        self.assertIn("upload_file '/tmp/local.json' -> '/remote/dest.json'", msg)
        self.assertIn("failed on 1/2 hosts", msg)
        self.assertIn("host2", msg)
        self.assertIn("permission denied", msg)

    def test_upload_file_all_hosts_fail(self):
        # Every host fails -> N/N in the message
        self.mock_client.copy_file.return_value = [
            self._fail_greenlet(IOError("disk full")),
            self._fail_greenlet(IOError("disk full")),
        ]

        with self.assertRaises(IOError) as cm:
            self.handle.upload_file("/tmp/local.json", "/remote/dest.json")

        self.assertIn("upload_file '/tmp/local.json' -> '/remote/dest.json'", str(cm.exception))
        self.assertIn("failed on 2/2 hosts", str(cm.exception))

    def test_upload_file_non_ioerror_exception_aggregated(self):
        # Any Exception (not just IOError) from cmd.get() is caught and
        # surfaced through the IOError aggregation. This locks in the
        # broad `except Exception` we use deliberately.
        self.mock_client.copy_file.return_value = [
            self._ok_greenlet(),
            self._fail_greenlet(RuntimeError("libssh2 channel closed")),
        ]

        with self.assertRaises(IOError) as cm:
            self.handle.upload_file("/tmp/local.json", "/remote/dest.json")

        self.assertIn("libssh2 channel closed", str(cm.exception))

    # -------------------- download_file --------------------

    def test_download_file_success_returns_host_to_path_dict(self):
        # On success returns {host: local_file<sep>host} for every host
        self.mock_client.copy_remote_file.return_value = [
            self._ok_greenlet(),
            self._ok_greenlet(),
        ]

        result = self.handle.download_file("/remote/file.json", "/tmp/local.json")

        self.assertEqual(
            result,
            {"host1": "/tmp/local.json_host1", "host2": "/tmp/local.json_host2"},
        )
        self.mock_client.copy_remote_file.assert_called_once_with(
            "/remote/file.json", "/tmp/local.json", recurse=False, suffix_separator="_"
        )
        self.mock_client.pool.join.assert_called_once()

    def test_download_file_custom_suffix_separator(self):
        # Honors a non-default suffix_separator both in the API call and the returned paths
        self.mock_client.copy_remote_file.return_value = [
            self._ok_greenlet(),
            self._ok_greenlet(),
        ]

        result = self.handle.download_file("/remote/file.json", "/tmp/local.json", suffix_separator=".")

        self.assertEqual(
            result,
            {"host1": "/tmp/local.json.host1", "host2": "/tmp/local.json.host2"},
        )
        self.mock_client.copy_remote_file.assert_called_once_with(
            "/remote/file.json", "/tmp/local.json", recurse=False, suffix_separator="."
        )

    def test_download_file_targets_only_requested_host(self):
        target_client = MagicMock()
        target_client.copy_remote_file.return_value = [self._ok_greenlet()]
        with patch("cvs.lib.parallel.phandle.ParallelSSHClient", return_value=target_client) as mock_client:
            result = self.handle.download_file("/remote/file.json", "/tmp/local.json", hosts=["host2"])

        self.assertEqual(result, {"host2": "/tmp/local.json_host2"})
        self.mock_client.copy_remote_file.assert_not_called()
        mock_client.assert_called_once_with(["host2"], user="user", password="pass", keepalive_seconds=30)
        target_client.copy_remote_file.assert_called_once_with(
            "/remote/file.json", "/tmp/local.json", recurse=False, suffix_separator="_"
        )
        target_client.pool.join.assert_called_once()

    def test_download_file_rejects_unknown_target_host(self):
        with self.assertRaisesRegex(ValueError, "unreachable host"):
            self.handle.download_file("/remote/file.json", "/tmp/local.json", hosts=["host3"])
        self.mock_client.copy_remote_file.assert_not_called()

    def test_download_file_partial_failure_raises_ioerror(self):
        # Failed host -> IOError lists it; succeeded host's path is NOT returned
        # (we raise before constructing a partial return value)
        boom = IOError("file not found")
        self.mock_client.copy_remote_file.return_value = [
            self._ok_greenlet(),
            self._fail_greenlet(boom),
        ]

        with self.assertRaises(IOError) as cm:
            self.handle.download_file("/remote/file.json", "/tmp/local.json")

        msg = str(cm.exception)
        self.assertIn("download_file '/remote/file.json' -> '/tmp/local.json'", msg)
        self.assertIn("failed on 1/2 hosts", msg)
        self.assertIn("host2", msg)
        self.assertIn("file not found", msg)

    def test_download_file_all_hosts_fail(self):
        self.mock_client.copy_remote_file.return_value = [
            self._fail_greenlet(IOError("nope")),
            self._fail_greenlet(IOError("nope")),
        ]

        with self.assertRaises(IOError) as cm:
            self.handle.download_file("/remote/file.json", "/tmp/local.json")

        self.assertIn("download_file '/remote/file.json' -> '/tmp/local.json'", str(cm.exception))
        self.assertIn("failed on 2/2 hosts", str(cm.exception))

    def test_download_file_recurse_passes_through(self):
        self.mock_client.copy_remote_file.return_value = [
            self._ok_greenlet(),
            self._ok_greenlet(),
        ]

        self.handle.download_file("/remote/dir", "/tmp/local", recurse=True)

        self.mock_client.copy_remote_file.assert_called_once_with(
            "/remote/dir", "/tmp/local", recurse=True, suffix_separator="_"
        )

    # -------------------- scp_file (alias) --------------------

    def test_scp_file_delegates_to_upload_file(self):
        # scp_file must call upload_file with the same args. We patch upload_file
        # to confirm delegation rather than reimplementing the underlying mock.
        with patch.object(ParallelHandle, "upload_file") as mock_upload:
            self.handle.scp_file("/tmp/local.json", "/remote/dest.json", recurse=True)
            mock_upload.assert_called_once_with("/tmp/local.json", "/remote/dest.json", recurse=True)

    def test_scp_file_propagates_ioerror_from_upload_file(self):
        # When upload_file raises, scp_file lets it propagate (no swallowing)
        with patch.object(ParallelHandle, "upload_file", side_effect=IOError("boom")):
            with self.assertRaises(IOError):
                self.handle.scp_file("/tmp/local.json", "/remote/dest.json")


class TestParallelHandleScpFile(unittest.TestCase):
    """
    Regression test from main (commit 79d7b20) covering scp_file's exception
    semantics. With our PR, scp_file is a backward-compatible alias for
    upload_file, so this test exercises upload_file's IOError contract via
    scp_file: type is IOError (not bare Exception), message includes both
    file paths, and the original exception is chained via __cause__.
    """

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        # Mock the pool.join() method that gets called in scp_file
        self.mock_client.pool = MagicMock()
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, ["host1"], user="user", password="pass")

    def test_scp_file_preserves_original_io_error(self):
        # Regression (B3): scp_file's exception handler used to read:
        #
        #     except IOError:
        #         raise Exception("Expected IOError exception, got none")
        #
        # which is logically inverted (the message fires precisely when an
        # IOError WAS caught), raises bare Exception instead of IOError,
        # drops the original traceback (no `from e`), and includes neither
        # the host nor the file path. This made debugging scp failures
        # nearly impossible -- the user saw "Expected IOError exception,
        # got none" and had no way to recover the actual cause.
        original_error = IOError("Permission denied")

        fake_cmd = MagicMock()
        fake_cmd.get.side_effect = original_error
        self.mock_client.copy_file.return_value = [fake_cmd]

        with self.assertRaises(IOError) as ctx:
            self.handle.scp_file("/local/x", "/remote/y")

        # Must be IOError, not bare Exception (so callers catching IOError
        # actually catch it).
        self.assertIs(type(ctx.exception), IOError)
        # Must surface both file paths in the message so the user knows
        # which copy failed.
        self.assertIn("/local/x", str(ctx.exception))
        self.assertIn("/remote/y", str(ctx.exception))
        # Must chain the original exception via __cause__ so the original
        # traceback is recoverable.
        self.assertIs(ctx.exception.__cause__, original_error)


class TestParallelHandleInactivityTimeout(unittest.TestCase):
    """Per-line inactivity timeout: resets on output, fires only on a stall."""

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ["host1"]
        self.mock_log = MagicMock()
        self.handle = ParallelHandle(self.mock_log, self.host_list, user="user", password="pass")

    @staticmethod
    def _slow_stream(gaps, lines):
        """Yield each line after sleeping the paired gap (gevent-cooperative)."""
        from gevent import sleep as gsleep

        for gap, line in zip(gaps, lines):
            gsleep(gap)
            yield line

    def test_active_stream_survives_short_gaps(self):
        # Gaps (0.05s) are well under the inactivity window (0.5s): the timer
        # resets on every line, so a long-but-active stream is NOT aborted.
        out = MagicMock()
        out.host = "host1"
        out.stdout = self._slow_stream([0.05, 0.05, 0.05], ["a", "b", "c"])
        out.stderr = []
        out.exception = None
        self.mock_client.run_command.return_value = [out]

        result = self.handle.exec("run", inactivity_timeout=0.5)

        # No total cap should be passed to run_command when inactivity is set.
        self.mock_client.run_command.assert_called_once_with("run", stop_on_errors=True)
        self.assertIn("a", result["host1"])
        self.assertIn("c", result["host1"])

    def test_stall_longer_than_window_aborts(self):
        # First line is quick, then a gap (0.6s) exceeds the window (0.3s): the
        # per-line timer fires and (stop_on_errors=True) the Timeout propagates.
        from pssh.exceptions import Timeout

        out = MagicMock()
        out.host = "host1"
        out.stdout = self._slow_stream([0.02, 0.6], ["first", "second"])
        out.stderr = []
        out.exception = None
        self.mock_client.run_command.return_value = [out]

        with self.assertRaises(Timeout):
            self.handle.exec("run", inactivity_timeout=0.3)

    def test_exec_rejects_timeout_and_inactivity_timeout_together(self):
        with self.assertRaises(ValueError):
            self.handle.exec("run", timeout=10, inactivity_timeout=0.3)
        self.mock_client.run_command.assert_not_called()

    def test_exec_cmd_list_rejects_timeout_and_inactivity_timeout_together(self):
        with self.assertRaises(ValueError):
            self.handle.exec_cmd_list(["run"], timeout=10, inactivity_timeout=0.3)
        self.mock_client.run_command.assert_not_called()


class TestParallelHandleDestroyClients(unittest.TestCase):
    """destroy_clients must actually tear the SSH transport down.

    A timed-out exec leaves the per-host greenlet in client.cmds unfinished.
    That greenlet's callable is a bound method of the ParallelSSHClient, so the
    client stays reachable, SSHClient.__del__ never runs, and the sshd session
    survives the ParallelHandle object -- verified against a live sshd. Dropping the
    reference is therefore not enough; the teardown has to be explicit.
    """

    @patch("cvs.lib.parallel.phandle.ParallelSSHClient")
    def _make_handle(self, mock_pssh_client, host_clients=None, cmds=None):
        mock_client = MagicMock()
        mock_client.cmds = cmds
        mock_client._host_clients = host_clients if host_clients is not None else {}
        mock_pssh_client.return_value = mock_client
        handle = ParallelHandle(MagicMock(), ["host1"], user="user", password="pass")
        return handle, mock_client

    def test_destroy_clients_disconnects_each_host_client(self):
        # The per-host SSHClient owns the socket; without an explicit
        # _disconnect() the session is left open on the server.
        host_client = MagicMock()
        handle, client = self._make_handle(host_clients={(0, "host1"): host_client})

        handle.destroy_clients()

        host_client._disconnect.assert_called_once_with()

    def test_destroy_clients_kills_pending_command_greenlets(self):
        # Unfinished greenlets from a timed-out exec are what pin the client;
        # they must be killed or the disconnect above is unreachable.
        greenlet = MagicMock()
        handle, client = self._make_handle(cmds=[greenlet])

        with patch("cvs.lib.parallel.phandle.killall") as mock_killall:
            handle.destroy_clients()

        mock_killall.assert_called_once()
        self.assertEqual(list(mock_killall.call_args.args[0]), [greenlet])

    def test_destroy_clients_survives_disconnect_errors(self):
        # A host that is already gone must not abort teardown of the others.
        dead = MagicMock()
        dead._disconnect.side_effect = OSError("connection already gone")
        alive = MagicMock()
        handle, client = self._make_handle(host_clients={(0, "h1"): dead, (1, "h2"): alive})

        handle.destroy_clients()

        alive._disconnect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
