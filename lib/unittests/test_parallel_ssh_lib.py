import unittest
from unittest.mock import patch, MagicMock
from lib.parallel_ssh_lib import Pssh

class TestPsshExec(unittest.TestCase):

    @patch('lib.parallel_ssh_lib.ParallelSSHClient')
    def setUp(self, mock_pssh_client):
        self.mock_client = MagicMock()
        mock_pssh_client.return_value = self.mock_client
        self.host_list = ['host1', 'host2']
        self.pssh = Pssh('log', self.host_list, user='user', password='pass')

    def test_exec_successful(self):
        # Test: Execute command successfully on all hosts
        mock_output1 = MagicMock()
        mock_output1.host = 'host1'
        mock_output1.stdout = ['output1 line1', 'output1 line2']
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = 'host2'
        mock_output2.stdout = ['output2 line1']
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec('echo hello')

        self.mock_client.run_command.assert_called_once_with('echo hello', stop_on_errors=True)
        self.assertIn('host1', result)
        self.assertIn('host2', result)
        self.assertIn('output1 line1', result['host1'])
        self.assertIn('output2 line1', result['host2'])

    def test_exec_with_exception_stop_on_errors_true(self):
        # Test: Handle exceptions with stop_on_errors=True (default)
        # Exception should be raised, and no result returned (no partial results)
        from pssh.exceptions import Timeout
        self.mock_client.run_command.side_effect = Timeout('Connection failed')

        # With stop_on_errors=True, run_command raises on exception, no result returned
        with self.assertRaises(Timeout) as cm:
            result = self.pssh.exec('echo hello')  # This should raise, so result is not assigned

        self.assertIn('Connection failed', str(cm.exception))
        # Since exception was raised, result was not returned
        self.assertNotIn('result', locals())

    def test_exec_with_exception_stop_on_errors_false(self):
        # Test Case 2.2: Execute command with timeout and stop_on_errors=False
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.pssh.stop_on_errors = False
        from pssh.exceptions import Timeout
        mock_output1 = MagicMock()
        mock_output1.host = 'host1'
        mock_output1.stdout = ['success output']
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = 'host2'
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout('Command timed out')

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec('echo hello', timeout=10)

        self.mock_client.run_command.assert_called_once_with('echo hello', read_timeout=10, stop_on_errors=False)
        self.assertIn('host1', result)
        self.assertIn('host2', result)
        self.assertIn('success output', result['host1'])
        self.assertIn('Command timed out', result['host2'])

    def test_exec_cmd_list_successful(self):
        # Test: Execute different commands on different hosts successfully
        cmd_list = ['echo host1', 'echo host2']
        mock_output1 = MagicMock()
        mock_output1.host = 'host1'
        mock_output1.stdout = ['host1']
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = 'host2'
        mock_output2.stdout = ['host2']
        mock_output2.stderr = []
        mock_output2.exception = None

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec_cmd_list(cmd_list)

        self.mock_client.run_command.assert_called_once_with('%s', host_args=cmd_list, stop_on_errors=True)
        self.assertIn('host1', result)
        self.assertIn('host2', result)
        self.assertIn('host1', result['host1'])
        self.assertIn('host2', result['host2'])

    def test_exec_cmd_list_with_exception_stop_on_errors_false(self):
        # Test: Handle exceptions with stop_on_errors=False for exec_cmd_list
        # Exception should not be raised instead populated in output for failed hosts, success for others
        self.pssh.stop_on_errors = False
        cmd_list = ['echo success', 'echo fail']
        from pssh.exceptions import Timeout
        mock_output1 = MagicMock()
        mock_output1.host = 'host1'
        mock_output1.stdout = ['success']
        mock_output1.stderr = []
        mock_output1.exception = None

        mock_output2 = MagicMock()
        mock_output2.host = 'host2'
        mock_output2.stdout = []
        mock_output2.stderr = []
        mock_output2.exception = Timeout('Command timed out')

        self.mock_client.run_command.return_value = [mock_output1, mock_output2]

        result = self.pssh.exec_cmd_list(cmd_list, timeout=10)

        self.mock_client.run_command.assert_called_once_with('%s', host_args=cmd_list, read_timeout=10, stop_on_errors=False)
        self.assertIn('host1', result)
        self.assertIn('host2', result)
        self.assertIn('success', result['host1'])
        self.assertIn('Command timed out', result['host2'])

    def test_exec_cmd_list_with_exception_stop_on_errors_true(self):
        # Test: Handle exceptions with stop_on_errors=True for exec_cmd_list
        # Exception should be raised, and no result returned (no partial results)
        cmd_list = ['echo test']
        from pssh.exceptions import Timeout
        self.mock_client.run_command.side_effect = Timeout('Command timed out')

        with self.assertRaises(Timeout) as cm:
            result = self.pssh.exec_cmd_list(cmd_list, timeout=5)

        self.assertIn('Command timed out', str(cm.exception))
        self.assertNotIn('result', locals())

if __name__ == '__main__':
    unittest.main()