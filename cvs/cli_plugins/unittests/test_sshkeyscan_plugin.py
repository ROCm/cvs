import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import argparse
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

        subparsers = argparse.ArgumentParser().add_subparsers()
        parser = self.plugin.get_parser(subparsers)

        # Test that parser was created
        self.assertIsNotNone(parser)


class TestSSHKeyScanNewImplementation(unittest.TestCase):
    """Test the new unified SSH key scanning implementation"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()

    def test_build_scan_command(self):
        """Test building the scan command"""
        cmd = self.plugin.build_scan_command("/tmp/hosts", "~/.ssh/known_hosts", 20, 50, 30)

        # Verify command structure
        self.assertIn("xargs -P 20 -n 50", cmd)
        self.assertIn("ssh-keyscan -T 30 -H", cmd)
        self.assertIn("~/.ssh/known_hosts", cmd)

    def test_build_remove_keys_command(self):
        """Test building the remove keys command"""
        cmd = self.plugin.build_remove_keys_command("/tmp/hosts", "~/.ssh/known_hosts")

        self.assertIn("ssh-keygen", cmd)
        self.assertIn("-R", cmd)
        self.assertIn("~/.ssh/known_hosts", cmd)

    @patch("tempfile.mkstemp")
    @patch("os.fdopen")
    @patch("os.unlink")
    def test_create_hosts_file_local(self, mock_unlink, mock_fdopen, mock_mkstemp):
        """Test creating local hosts file"""
        # Mock file operations
        mock_mkstemp.return_value = (3, "/tmp/hosts123")
        mock_file = MagicMock()
        mock_fdopen.return_value.__enter__.return_value = mock_file

        hosts = ["host1", "host2", "host3"]
        result = self.plugin.create_hosts_file_local(hosts)

        # Verify temp file creation
        self.assertEqual(result, "/tmp/hosts123")
        mock_mkstemp.assert_called_with(suffix='.hosts', text=True)

        # Verify hosts were written
        expected_calls = [call("host1\n"), call("host2\n"), call("host3\n")]
        mock_file.write.assert_has_calls(expected_calls)

    @patch("os.unlink")
    @patch("tempfile.mkstemp")
    @patch("os.fdopen")
    def test_create_hosts_file_remote(self, mock_fdopen, mock_mkstemp, mock_unlink):
        """Test creating remote hosts file using SCP transfer"""
        # Mock local temp file creation
        mock_mkstemp.return_value = (3, "/tmp/local_temp_file")
        mock_file = MagicMock()
        mock_fdopen.return_value.__enter__.return_value = mock_file

        # Mock phdl for SCP transfer
        mock_phdl = MagicMock()

        hosts = ["host1", "host2"]
        result = self.plugin.create_hosts_file_remote(hosts, mock_phdl)

        # Verify result is a UUID-based remote filename
        self.assertTrue(result.startswith("/tmp/cvs_hosts_"))
        self.assertEqual(len(result.split("_")[-1]), 8)  # UUID hex[:8] length

        # Verify local temp file was created
        mock_mkstemp.assert_called_once()
        mock_fdopen.assert_called_once_with(3, 'w')

        # Verify hosts were written to local file
        mock_file.write.assert_any_call("host1\n")
        mock_file.write.assert_any_call("host2\n")

        # Verify SCP transfer was called
        mock_phdl.scp_file.assert_called_once_with("/tmp/local_temp_file", result)

        # Verify local temp file cleanup
        mock_unlink.assert_called_once_with("/tmp/local_temp_file")

    @patch("subprocess.run")
    def test_execute_command_local_success(self, mock_run):
        """Test local command execution success"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.plugin.execute_command_local("echo test", 30)

        # Verify result structure
        self.assertTrue(result['success'])
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['stdout'], "success output")

    @patch("subprocess.run")
    def test_execute_command_local_timeout(self, mock_run):
        """Test local command execution timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        result = self.plugin.execute_command_local("sleep 60", 30)

        # Verify timeout handling
        self.assertFalse(result['success'])
        self.assertEqual(result['returncode'], 124)

    def test_execute_command_remote_success(self):
        """Test remote command execution success"""
        mock_phdl = MagicMock()
        mock_phdl.exec.return_value = {"head": "success output"}

        result = self.plugin.execute_command_remote("echo test", mock_phdl, 30)

        # Verify result structure
        self.assertTrue(result['success'])
        self.assertEqual(result['stdout'], "success output")

    def test_parse_ssh_keyscan_output(self):
        """Test parsing SSH keyscan output"""
        # Mock hashed output (what we get with -H flag)
        stdout = "|1|hash1|hash ssh-rsa AAAAB3NzaC1...\n|1|hash2|hash ecdsa-sha2-nistp256 AAAAE2VjZH..."
        stderr = ""
        hosts = ["host1", "host2"]

        successful, failed, results = self.plugin.parse_ssh_keyscan_output(stdout, stderr, hosts)

        # Should detect keys and estimate success
        self.assertGreater(successful, 0)
        self.assertEqual(len(results), len(hosts))

    @patch.object(SSHKeyScanPlugin, 'create_hosts_file_local')
    @patch.object(SSHKeyScanPlugin, 'execute_command_local')
    @patch.object(SSHKeyScanPlugin, 'cleanup_hosts_file')
    def test_scan_ssh_keys_local(self, mock_cleanup, mock_execute, mock_create_hosts):
        """Test SSH key scanning locally"""
        # Mock dependencies
        mock_create_hosts.return_value = "/tmp/hosts123"
        mock_execute.return_value = {'success': True, 'stdout': "|1|hash|hash ssh-rsa AAAAB3...", 'stderr': ''}

        hosts = ["host1", "host2"]
        successful, failed, results = self.plugin.scan_ssh_keys(hosts, "~/.ssh/known_hosts", 20, None, False)

        # Verify execution flow
        mock_create_hosts.assert_called_once_with(hosts)
        mock_execute.assert_called_once()
        mock_cleanup.assert_called_once_with("/tmp/hosts123", None)

        # Verify results
        self.assertGreaterEqual(successful, 0)
        self.assertEqual(len(results), len(hosts))

    def test_scan_ssh_keys_dry_run(self):
        """Test SSH key scanning in dry run mode"""
        hosts = ["host1", "host2"]
        successful, failed, results = self.plugin.scan_ssh_keys(hosts, "~/.ssh/known_hosts", 20, None, True)

        # In dry run, should return success for all hosts
        self.assertEqual(successful, len(hosts))
        self.assertEqual(failed, 0)
        for result in results:
            self.assertIn("Would scan and add SSH host key", result)


class TestSSHKeyScanConfigMethods(unittest.TestCase):
    """Test the configuration and setup methods"""

    def setUp(self):
        self.plugin = SSHKeyScanPlugin()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"node_dict": {"host1": {}}, "head_node_dict": {"mgmt_ip": "192.168.1.100"}}',
    )
    @patch("os.environ.get")
    def test_load_and_validate_cluster_config(self, mock_env, mock_file):
        """Test cluster config loading"""
        mock_env.return_value = None

        args = argparse.Namespace(cluster_file="/path/to/cluster.json")
        cluster_file, cluster, hosts = self.plugin._load_and_validate_cluster_config(args)

        self.assertEqual(cluster_file, "/path/to/cluster.json")
        self.assertIn("node_dict", cluster)
        self.assertEqual(hosts, ["host1"])

    def test_resolve_known_hosts_path_local(self):
        """Test resolving known_hosts path for local execution"""
        args = argparse.Namespace(known_hosts="~/.ssh/known_hosts", at="local")

        with patch("os.path.expanduser", return_value="/home/user/.ssh/known_hosts"):
            path = self.plugin._resolve_known_hosts_path(args)
            self.assertEqual(path, "/home/user/.ssh/known_hosts")

    def test_resolve_known_hosts_path_remote(self):
        """Test resolving known_hosts path for remote execution"""
        args = argparse.Namespace(known_hosts="~/.ssh/known_hosts", at="head")

        path = self.plugin._resolve_known_hosts_path(args)
        self.assertEqual(path, "~/.ssh/known_hosts")  # Should not expand for remote

    def test_setup_remote_connection(self):
        """Test setting up remote SSH connection"""
        cluster = {
            "head_node_dict": {"mgmt_ip": "192.168.1.100"},
            "username": "testuser",
            "priv_key_file": "/path/to/key.pem",
            "env_vars": None,
        }
        args = argparse.Namespace(at="head")

        with patch('cvs.cli_plugins.sshkeyscan_plugin.Pssh') as mock_pssh:
            result = self.plugin._setup_remote_connection(cluster, args)

            self.assertIsNotNone(result)
            mock_pssh.assert_called_with(
                self.plugin.logger, ["192.168.1.100"], user="testuser", pkey="/path/to/key.pem", env_vars=None
            )

    def test_setup_remote_connection_local_mode(self):
        """Test that no connection is created for local mode"""
        args = argparse.Namespace(at="local")

        result = self.plugin._setup_remote_connection({}, args)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
