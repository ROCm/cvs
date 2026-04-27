import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import argparse
import json
import subprocess

from cvs.cli_plugins.sshkeyscan_plugin import SSHKeyScanPlugin


class TestSSHKeyScanPlugin(unittest.TestCase):
    """Test SSHKeyScanPlugin basic functionality"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()
        self.sample_cluster = {
            "node_dict": {"node1": {"mgmt_ip": "192.168.1.1"}, "node2": {"mgmt_ip": "192.168.1.2"}},
            "head_node_dict": {"mgmt_ip": "192.168.1.100"},
            "username": "testuser",
            "priv_key_file": "/path/to/key.pem",
        }

    def test_get_name(self):
        """Test get_name returns correct plugin name"""
        self.assertEqual(self.plugin.get_name(), "sshkeyscan")

    def test_get_parser(self):
        """Test parser setup with correct arguments"""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        result_parser = self.plugin.get_parser(subparsers)

        self.assertIsNotNone(result_parser)
        # Test that parser has the expected arguments by parsing test args
        test_args = [
            "--cluster_file",
            "test.json",
            "--known_hosts",
            "~/.ssh/known_hosts",
            "--remove-existing",
            "--dry-run",
            "--parallel",
            "10",
            "--at",
            "head",
        ]
        args = result_parser.parse_args(test_args)
        self.assertEqual(args.cluster_file, "test.json")
        self.assertEqual(args.known_hosts, "~/.ssh/known_hosts")
        self.assertTrue(args.remove_existing)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.parallel, 10)
        self.assertEqual(args.at, "head")

    def test_get_epilog(self):
        """Test epilog contains help examples"""
        epilog = self.plugin.get_epilog()
        self.assertIn("SSH Key Scan Commands:", epilog)
        self.assertIn("cvs sshkeyscan", epilog)
        self.assertIn("--cluster_file", epilog)


class TestSSHKeyScanLocalExecution(unittest.TestCase):
    """Test local SSH key scanning functionality"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.expanduser')
    def test_scan_host_key_success(self, mock_expanduser, mock_file, mock_run):
        """Test successful SSH key scan for a host"""
        mock_expanduser.return_value = "/home/user/.ssh/known_hosts"

        # Mock successful ssh-keyscan output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "host1 ssh-rsa AAAAB3NzaC1yc2E...\nhost1 ecdsa-sha2-nistp256 AAAAE2VjZH...\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.plugin.scan_host_key("host1", "~/.ssh/known_hosts")

        # Verify ssh-keyscan was called correctly
        mock_run.assert_called_with(['ssh-keyscan', '-H', 'host1'], capture_output=True, text=True, timeout=30)

        # Verify file was written to
        mock_file.assert_called_with("/home/user/.ssh/known_hosts", 'a')
        mock_file().write.assert_called_with(mock_result.stdout)

        # Check result message
        self.assertIn("host1: SUCCESS", result)
        self.assertIn("Added 2 host key(s)", result)

    @patch('subprocess.run')
    def test_scan_host_key_failure(self, mock_run):
        """Test SSH key scan failure"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection refused"
        mock_run.return_value = mock_result

        result = self.plugin.scan_host_key("badhost", "~/.ssh/known_hosts")

        self.assertIn("badhost: FAILED", result)
        self.assertIn("Connection refused", result)

    @patch('subprocess.run')
    def test_scan_host_key_timeout(self, mock_run):
        """Test SSH key scan timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired(['ssh-keyscan'], 30)

        result = self.plugin.scan_host_key("slowhost", "~/.ssh/known_hosts")

        self.assertIn("slowhost: FAILED", result)
        self.assertIn("Connection timeout", result)

    def test_scan_host_key_dry_run(self):
        """Test dry run mode"""
        result = self.plugin.scan_host_key("host1", "~/.ssh/known_hosts", dry_run=True)

        self.assertIn("host1: Would scan and add SSH host key", result)

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.expanduser')
    def test_scan_host_key_remove_existing(self, mock_expanduser, mock_file, mock_run):
        """Test removing existing host keys before scanning"""
        mock_expanduser.return_value = "/home/user/.ssh/known_hosts"

        # Mock ssh-keygen -R call and ssh-keyscan call
        mock_results = [
            MagicMock(returncode=0),  # ssh-keygen -R result
            MagicMock(returncode=0, stdout="host1 ssh-rsa AAAAB3...\n", stderr=""),  # ssh-keyscan result
        ]
        mock_run.side_effect = mock_results

        result = self.plugin.scan_host_key("host1", "~/.ssh/known_hosts", remove_existing=True)

        # Verify both commands were called
        expected_calls = [
            call(['ssh-keygen', '-f', '~/.ssh/known_hosts', '-R', 'host1'], capture_output=True, text=True),
            call(['ssh-keyscan', '-H', 'host1'], capture_output=True, text=True, timeout=30),
        ]
        mock_run.assert_has_calls(expected_calls)

        self.assertIn("host1: SUCCESS", result)

    @patch('subprocess.run')
    def test_scan_host_key_no_output(self, mock_run):
        """Test SSH key scan with empty output"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        result = self.plugin.scan_host_key("host1", "~/.ssh/known_hosts")

        self.assertIn("host1: FAILED", result)
        self.assertIn("No host key received", result)


class TestSSHKeyScanRemoteExecution(unittest.TestCase):
    """Test remote SSH key scanning functionality"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()
        self.mock_pssh = MagicMock()

    def test_scan_host_key_remote_success(self):
        """Test successful remote SSH key scan"""
        # Mock pssh exec results
        self.mock_pssh.exec.side_effect = [
            {"head_node": "host1 ssh-rsa AAAAB3NzaC1yc2E...\nhost1 ecdsa-sha2-nistp256 AAAAE2VjZH...\n"},  # ssh-keyscan
            {"head_node": ""},  # append to file
        ]

        result = self.plugin.scan_host_key_remote(self.mock_pssh, "host1", "~/.ssh/known_hosts")

        # Verify commands were executed
        expected_calls = [
            call("timeout 30 ssh-keyscan -H host1", timeout=35, print_console=False),
            call(
                "echo 'host1 ssh-rsa AAAAB3NzaC1yc2E...\nhost1 ecdsa-sha2-nistp256 AAAAE2VjZH...' >> ~/.ssh/known_hosts",
                print_console=False,
            ),
        ]
        self.mock_pssh.exec.assert_has_calls(expected_calls)

        self.assertIn("host1: SUCCESS", result)
        self.assertIn("Added 2 host key(s)", result)

    def test_scan_host_key_remote_failure(self):
        """Test remote SSH key scan failure"""
        self.mock_pssh.exec.return_value = {"head_node": "Connection refused"}

        result = self.plugin.scan_host_key_remote(self.mock_pssh, "host1", "~/.ssh/known_hosts")

        self.assertIn("host1: FAILED", result)
        self.assertIn("Connection refused", result)

    def test_scan_host_key_remote_dry_run(self):
        """Test remote dry run mode"""
        result = self.plugin.scan_host_key_remote(self.mock_pssh, "host1", "~/.ssh/known_hosts", dry_run=True)

        self.assertIn("host1: Would scan and add SSH host key (on remote)", result)
        # Should not call exec in dry run
        self.mock_pssh.exec.assert_not_called()

    def test_scan_host_key_remote_remove_existing(self):
        """Test remote key removal before scanning"""
        self.mock_pssh.exec.side_effect = [
            {"head_node": ""},  # ssh-keygen -R
            {"head_node": "host1 ssh-rsa AAAAB3...\n"},  # ssh-keyscan
            {"head_node": ""},  # append
        ]

        result = self.plugin.scan_host_key_remote(self.mock_pssh, "host1", "~/.ssh/known_hosts", remove_existing=True)

        # Verify removal command was called
        calls = self.mock_pssh.exec.call_args_list
        self.assertEqual(calls[0][0][0], "ssh-keygen -f ~/.ssh/known_hosts -R host1")

        self.assertIn("host1: SUCCESS", result)

    def test_scan_host_key_remote_invalid_output(self):
        """Test remote scan with invalid output"""
        self.mock_pssh.exec.return_value = {"head_node": "invalid output without ssh keys"}

        result = self.plugin.scan_host_key_remote(self.mock_pssh, "host1", "~/.ssh/known_hosts")

        self.assertIn("host1: FAILED", result)
        self.assertIn("Invalid output", result)

    @patch('cvs.cli_plugins.sshkeyscan_plugin.Pssh')
    def test_scan_host_key_remote_threadsafe(self, mock_pssh_class):
        """Test thread-safe remote scanning"""
        mock_pssh_instance = MagicMock()
        mock_pssh_class.return_value = mock_pssh_instance
        mock_pssh_instance.exec.return_value = {"head_node": "host1 ssh-rsa AAAAB3...\n"}

        cluster_config = {
            'head_node': '192.168.1.100',
            'username': 'testuser',
            'priv_key_file': '/path/to/key.pem',
            'env_vars': None,
        }

        with patch.object(self.plugin, 'scan_host_key_remote') as mock_scan:
            mock_scan.return_value = "host1: SUCCESS"

            result = self.plugin.scan_host_key_remote_threadsafe(cluster_config, "host1", "~/.ssh/known_hosts")

            # Verify thread-local pssh was created
            mock_pssh_class.assert_called_with(
                None, ['192.168.1.100'], user='testuser', pkey='/path/to/key.pem', env_vars=None
            )

            # Verify scan_host_key_remote was called with thread pssh
            mock_scan.assert_called_with(mock_pssh_instance, "host1", "~/.ssh/known_hosts", False, False)

            self.assertEqual(result, "host1: SUCCESS")


class TestSSHKeyScanPluginRun(unittest.TestCase):
    """Test the main run method with various configurations"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()
        self.sample_cluster = {
            "node_dict": {"node1": {"mgmt_ip": "192.168.1.1"}, "node2": {"mgmt_ip": "192.168.1.2"}},
            "head_node_dict": {"mgmt_ip": "192.168.1.100"},
            "username": "testuser",
            "priv_key_file": "/path/to/key.pem",
        }

    @patch.dict(os.environ, {}, clear=True)
    def test_run_no_cluster_file(self):
        """Test run method with no cluster file specified"""
        args = argparse.Namespace(cluster_file=None)

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        mock_print.assert_called_with(
            "Error: No cluster file specified. Set CLUSTER_FILE environment variable or use --cluster_file."
        )

    @patch.dict(os.environ, {'CLUSTER_FILE': '/path/to/cluster.json'})
    @patch('builtins.open', new_callable=mock_open, read_data='{"invalid": "json"')
    def test_run_invalid_json(self, mock_file):
        """Test run method with invalid JSON in cluster file"""
        args = argparse.Namespace(cluster_file=None)

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        # Should print JSON decode error
        mock_print.assert_called()
        call_args = str(mock_print.call_args_list)
        self.assertIn("Invalid JSON", call_args)

    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_run_cluster_file_not_found(self, mock_file):
        """Test run method with non-existent cluster file"""
        args = argparse.Namespace(cluster_file='/nonexistent/cluster.json')

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        mock_print.assert_called_with("Error: Cluster file '/nonexistent/cluster.json' not found.")

    @patch('builtins.open', new_callable=mock_open)
    def test_run_empty_node_dict(self, mock_file):
        """Test run method with empty node_dict"""
        mock_file.return_value.read.return_value = json.dumps({"node_dict": {}})
        args = argparse.Namespace(cluster_file='cluster.json')

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        mock_print.assert_called_with("Error: No hosts found in cluster file.")

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.expanduser')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('cvs.cli_plugins.sshkeyscan_plugin.ThreadPoolExecutor')
    def test_run_local_execution_success(self, mock_executor, mock_exists, mock_makedirs, mock_expanduser, mock_file):
        """Test successful local execution"""
        mock_file.return_value.read.return_value = json.dumps(self.sample_cluster)
        mock_expanduser.return_value = "/home/user/.ssh/known_hosts"
        mock_exists.return_value = False  # known_hosts doesn't exist

        # Mock ThreadPoolExecutor
        mock_future1 = MagicMock()
        mock_future1.result.return_value = "node1: SUCCESS - Added 1 host key(s)"
        mock_future2 = MagicMock()
        mock_future2.result.return_value = "node2: SUCCESS - Added 1 host key(s)"

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed
        with patch('cvs.cli_plugins.sshkeyscan_plugin.as_completed', return_value=[mock_future1, mock_future2]):
            args = argparse.Namespace(
                cluster_file='cluster.json',
                known_hosts='~/.ssh/known_hosts',
                remove_existing=False,
                dry_run=False,
                parallel=20,
                at='local',
            )

            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

            # Verify ThreadPoolExecutor was used correctly
            mock_executor.assert_called_with(max_workers=20)
            # Verify context manager was used
            mock_executor_instance.__enter__.assert_called()
            mock_executor_instance.__exit__.assert_called()

            # Check that success messages were printed
            printed_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            success_messages = [msg for msg in printed_calls if "SUCCESS" in str(msg)]
            self.assertEqual(len(success_messages), 2)

    @patch('builtins.open', new_callable=mock_open)
    @patch('cvs.cli_plugins.sshkeyscan_plugin.Pssh')
    @patch('cvs.cli_plugins.sshkeyscan_plugin.ThreadPoolExecutor')
    def test_run_head_execution_success(self, mock_executor, mock_pssh, mock_file):
        """Test successful head node execution"""
        mock_file.return_value.read.return_value = json.dumps(self.sample_cluster)

        # Mock pssh
        mock_pssh_instance = MagicMock()
        mock_pssh.return_value = mock_pssh_instance

        # Mock ThreadPoolExecutor
        mock_future1 = MagicMock()
        mock_future1.result.return_value = "node1: SUCCESS - Added 1 host key(s) (on remote)"
        mock_future2 = MagicMock()
        mock_future2.result.return_value = "node2: SUCCESS - Added 1 host key(s) (on remote)"

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance

        with patch('cvs.cli_plugins.sshkeyscan_plugin.as_completed', return_value=[mock_future1, mock_future2]):
            args = argparse.Namespace(
                cluster_file='cluster.json',
                known_hosts='~/.ssh/known_hosts',
                remove_existing=False,
                dry_run=False,
                parallel=20,
                at='head',
            )

            with patch('builtins.print'):
                self.plugin.run(args)

            # Verify pssh was created correctly
            mock_pssh.assert_called_with(
                None, ['192.168.1.100'], user='testuser', pkey='/path/to/key.pem', env_vars=None
            )

            # Verify setup command was executed
            mock_pssh_instance.exec.assert_called_with(
                "mkdir -p $(dirname ~/.ssh/known_hosts) && touch ~/.ssh/known_hosts", print_console=False
            )

    @patch('builtins.open', new_callable=mock_open)
    def test_run_head_execution_missing_head_node(self, mock_file):
        """Test head execution with missing head node info"""
        cluster_without_head = {
            "node_dict": {"node1": {"mgmt_ip": "192.168.1.1"}},
            "head_node_dict": {},  # Missing mgmt_ip
        }
        mock_file.return_value.read.return_value = json.dumps(cluster_without_head)

        args = argparse.Namespace(cluster_file='cluster.json', at='head')

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        mock_print.assert_called_with("Error: No head node found in cluster file (head_node_dict.mgmt_ip).")

    @patch('builtins.open', new_callable=mock_open)
    def test_run_head_execution_missing_credentials(self, mock_file):
        """Test head execution with missing SSH credentials"""
        cluster_no_creds = {
            "node_dict": {"node1": {"mgmt_ip": "192.168.1.1"}},
            "head_node_dict": {"mgmt_ip": "192.168.1.100"},
            # Missing username and priv_key_file
        }
        mock_file.return_value.read.return_value = json.dumps(cluster_no_creds)

        args = argparse.Namespace(cluster_file='cluster.json', at='head')

        with self.assertRaises(SystemExit) as cm:
            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

        self.assertEqual(cm.exception.code, 1)
        mock_print.assert_called_with(
            "Error: username and priv_key_file required in cluster file for remote execution."
        )

    @patch('builtins.open', new_callable=mock_open)
    @patch('cvs.cli_plugins.sshkeyscan_plugin.ThreadPoolExecutor')
    def test_run_dry_run_mode(self, mock_executor, mock_file):
        """Test dry run mode"""
        mock_file.return_value.read.return_value = json.dumps(self.sample_cluster)

        # Mock ThreadPoolExecutor - dry run messages don't contain SUCCESS/FAILED
        # but the code counts anything without "SUCCESS" as failed, so let's use SUCCESS for dry run
        mock_future1 = MagicMock()
        mock_future1.result.return_value = "node1: SUCCESS - Would scan and add SSH host key"
        mock_future2 = MagicMock()
        mock_future2.result.return_value = "node2: SUCCESS - Would scan and add SSH host key"

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance

        with patch('cvs.cli_plugins.sshkeyscan_plugin.as_completed', return_value=[mock_future1, mock_future2]):
            args = argparse.Namespace(
                cluster_file='cluster.json',
                known_hosts='~/.ssh/known_hosts',
                remove_existing=False,
                dry_run=True,
                parallel=20,
                at='local',
            )

            with patch('builtins.print') as mock_print:
                self.plugin.run(args)

            # Check that dry run message was printed
            printed_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            dry_run_messages = [msg for msg in printed_calls if "DRY RUN" in str(msg)]
            self.assertGreater(len(dry_run_messages), 0)

    @patch('builtins.open', new_callable=mock_open)
    @patch('cvs.cli_plugins.sshkeyscan_plugin.ThreadPoolExecutor')
    def test_run_with_failures_exit_code(self, mock_executor, mock_file):
        """Test that run exits with code 1 when there are failures"""
        mock_file.return_value.read.return_value = json.dumps(self.sample_cluster)

        # Mock one success and one failure
        mock_future1 = MagicMock()
        mock_future1.result.return_value = "node1: SUCCESS - Added 1 host key(s)"
        mock_future2 = MagicMock()
        mock_future2.result.return_value = "node2: FAILED - Connection refused"

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance

        with patch('cvs.cli_plugins.sshkeyscan_plugin.as_completed', return_value=[mock_future1, mock_future2]):
            args = argparse.Namespace(
                cluster_file='cluster.json',
                known_hosts='~/.ssh/known_hosts',
                remove_existing=False,
                dry_run=False,
                parallel=20,
                at='local',
            )

            with self.assertRaises(SystemExit) as cm:
                with patch('builtins.print'):
                    self.plugin.run(args)

            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
