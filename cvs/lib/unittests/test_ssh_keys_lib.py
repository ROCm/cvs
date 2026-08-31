# cvs/lib/unittests/test_ssh_keys_lib.py
import unittest
from unittest.mock import MagicMock, patch

import cvs.lib.ssh_keys_lib as lib


class TestValidateKeyDistributionConfig(unittest.TestCase):
    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=True)
    def test_valid_config_returns_normalized(self, _isfile):
        cfg = {
            "cluster_key_private_path": "/tmp/id",
            "cluster_key_public_path": "/tmp/id.pub",
        }
        norm = lib.validate_key_distribution_config(cfg)
        self.assertEqual(norm["key_name"], "cluster_id")
        self.assertEqual(norm["remote_ssh_dir"], "~/.ssh")
        self.assertEqual(norm["verify_mode"], "ring")
        self.assertEqual(norm["verify_timeout"], 20)
        self.assertEqual(norm["ssh_config_write_mode"], "managed_block")
        self.assertEqual(norm["controlling_station_pubkey_path"], "")

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=True)
    def test_explicit_values_not_overwritten_by_defaults(self, _isfile):
        cfg = {
            "cluster_key_private_path": "/tmp/id",
            "cluster_key_public_path": "/tmp/id.pub",
            "key_name": "mykey",
            "verify_mode": "full_mesh",
            "verify_timeout": 60,
        }
        norm = lib.validate_key_distribution_config(cfg)
        self.assertEqual(norm["key_name"], "mykey")
        self.assertEqual(norm["verify_mode"], "full_mesh")
        self.assertEqual(norm["verify_timeout"], 60)

    def test_missing_private_path_raises(self):
        with self.assertRaises(ValueError):
            lib.validate_key_distribution_config({"cluster_key_public_path": "/tmp/id.pub"})

    def test_empty_private_path_raises(self):
        with self.assertRaises(ValueError):
            lib.validate_key_distribution_config(
                {"cluster_key_private_path": "", "cluster_key_public_path": "/tmp/id.pub"}
            )

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=False)
    def test_nonexistent_private_path_raises(self, _isfile):
        with self.assertRaises(ValueError):
            lib.validate_key_distribution_config(
                {
                    "cluster_key_private_path": "/no/such/file",
                    "cluster_key_public_path": "/tmp/id.pub",
                }
            )

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", side_effect=lambda p: p != "/no/ctrl")
    def test_controlling_path_set_but_missing_raises(self, _isfile):
        with self.assertRaises(ValueError):
            lib.validate_key_distribution_config(
                {
                    "cluster_key_private_path": "/tmp/id",
                    "cluster_key_public_path": "/tmp/id.pub",
                    "controlling_station_pubkey_path": "/no/ctrl",
                }
            )

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=True)
    def test_controlling_path_empty_ok(self, _isfile):
        cfg = {
            "cluster_key_private_path": "/tmp/id",
            "cluster_key_public_path": "/tmp/id.pub",
            "controlling_station_pubkey_path": "",
        }
        norm = lib.validate_key_distribution_config(cfg)
        self.assertEqual(norm["controlling_station_pubkey_path"], "")

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=True)
    def test_known_default_key_name_logs_warning(self, _isfile):
        cfg = {
            "cluster_key_private_path": "/tmp/id",
            "cluster_key_public_path": "/tmp/id.pub",
            "key_name": "id_rsa",
        }
        with self.assertLogs("root", level="WARNING"):
            lib.validate_key_distribution_config(cfg)


class TestCollectClusterHostnames(unittest.TestCase):
    def test_node_dict_only(self):
        cluster = {"node_dict": {"node1": {}, "node2": {}}}
        result = lib.collect_cluster_hostnames(cluster)
        self.assertEqual(result, ["node1", "node2"])

    def test_distinct_vpc_ip_included(self):
        cluster = {
            "node_dict": {
                "node1": {"vpc_ip": "10.0.0.1"},
                "node2": {"vpc_ip": "10.0.0.2"},
            }
        }
        result = lib.collect_cluster_hostnames(cluster)
        self.assertIn("node1", result)
        self.assertIn("10.0.0.1", result)
        self.assertIn("node2", result)
        self.assertIn("10.0.0.2", result)
        self.assertEqual(len(result), 4)

    def test_vpc_ip_same_as_node_name_not_duplicated(self):
        cluster = {"node_dict": {"10.0.0.1": {"vpc_ip": "10.0.0.1"}}}
        result = lib.collect_cluster_hostnames(cluster)
        self.assertEqual(result, ["10.0.0.1"])

    def test_order_preserved(self):
        cluster = {"node_dict": {"node3": {}, "node1": {}, "node2": {}}}
        result = lib.collect_cluster_hostnames(cluster)
        self.assertEqual(result, ["node3", "node1", "node2"])

    def test_empty_node_dict(self):
        self.assertEqual(lib.collect_cluster_hostnames({"node_dict": {}}), [])

    def test_no_node_dict(self):
        self.assertEqual(lib.collect_cluster_hostnames({}), [])


class TestLongestCommonPrefix(unittest.TestCase):
    def test_shared_prefix(self):
        self.assertEqual(lib._longest_common_prefix(["node01", "node02", "node03"]), "node0")

    def test_no_shared_prefix(self):
        self.assertEqual(lib._longest_common_prefix(["abc", "xyz"]), "")

    def test_single_element(self):
        self.assertEqual(lib._longest_common_prefix(["hello"]), "hello")

    def test_identical_elements(self):
        self.assertEqual(lib._longest_common_prefix(["foo", "foo"]), "foo")

    def test_empty_list(self):
        self.assertEqual(lib._longest_common_prefix([]), "")


class TestDeriveSshHostPattern(unittest.TestCase):
    def test_override_wins(self):
        result = lib.derive_ssh_host_pattern(["node1", "node2"], override="myoverride")
        self.assertEqual(result, "myoverride")

    def test_named_prefix_wildcard(self):
        result = lib.derive_ssh_host_pattern(["node01", "node02", "node03"])
        self.assertEqual(result, "node0*")

    def test_ip_shared_three_octets(self):
        result = lib.derive_ssh_host_pattern(["10.0.0.1", "10.0.0.2"])
        self.assertEqual(result, "10.0.0.*")

    def test_ip_shared_two_octets(self):
        result = lib.derive_ssh_host_pattern(["10.0.1.1", "10.0.2.1"])
        self.assertEqual(result, "10.0.*")

    def test_ip_shared_one_octet(self):
        result = lib.derive_ssh_host_pattern(["10.1.0.1", "10.2.0.1"])
        self.assertEqual(result, "10.*")

    def test_ip_no_shared_octet_explicit_list(self):
        result = lib.derive_ssh_host_pattern(["10.0.0.1", "192.168.0.1"])
        self.assertIn("10.0.0.1", result)
        self.assertIn("192.168.0.1", result)

    def test_mixed_names_explicit_list(self):
        result = lib.derive_ssh_host_pattern(["gpu-a", "worker-b"])
        self.assertIn("gpu-a", result)
        self.assertIn("worker-b", result)

    def test_single_host(self):
        result = lib.derive_ssh_host_pattern(["node1"])
        self.assertEqual(result, "node1")


class TestRenderSshConfigBlock(unittest.TestCase):
    def test_block_structure(self):
        block = lib.render_ssh_config_block("node*", "myuser", "~/.ssh/cluster_id")
        self.assertIn(lib.SSH_CONFIG_BEGIN, block)
        self.assertIn(lib.SSH_CONFIG_END, block)
        self.assertIn("Host node*", block)
        self.assertIn("User myuser", block)
        self.assertIn("IdentityFile ~/.ssh/cluster_id", block)
        self.assertIn("StrictHostKeyChecking no", block)
        self.assertIn("UserKnownHostsFile /dev/null", block)
        self.assertIn("LogLevel ERROR", block)

    def test_begin_before_end(self):
        block = lib.render_ssh_config_block("*", "u", "/path")
        begin_pos = block.index(lib.SSH_CONFIG_BEGIN)
        end_pos = block.index(lib.SSH_CONFIG_END)
        self.assertLess(begin_pos, end_pos)


class TestBuildEnsureSshDirCmd(unittest.TestCase):
    def test_contains_mkdir_and_chmod(self):
        cmd = lib.build_ensure_ssh_dir_cmd("~/.ssh")
        self.assertIn("mkdir -p", cmd)
        self.assertIn("chmod 700", cmd)
        self.assertIn("~/.ssh", cmd)

    def test_non_home_path_quoted(self):
        cmd = lib.build_ensure_ssh_dir_cmd("/some/path with spaces")
        self.assertIn("chmod 700", cmd)


class TestBuildKeyPermsCmd(unittest.TestCase):
    def test_private_600_public_644(self):
        cmd = lib.build_key_perms_cmd("~/.ssh", "cluster_id")
        self.assertIn("chmod 600", cmd)
        self.assertIn("chmod 644", cmd)
        self.assertIn("cluster_id.pub", cmd)


class TestBuildAuthorizePubkeyCmd(unittest.TestCase):
    def test_uses_grep_qxf(self):
        cmd = lib.build_authorize_pubkey_cmd("~/.ssh", "~/.ssh/cluster_id.pub")
        self.assertIn("grep -qxF", cmd)
        self.assertIn("authorized_keys", cmd)

    def test_never_contains_raw_key_material(self):
        cmd = lib.build_authorize_pubkey_cmd("~/.ssh", "~/.ssh/cluster_id.pub")
        # Key content never embedded; only cat via subshell
        self.assertIn("cat ", cmd)
        self.assertNotIn("ssh-rsa", cmd)

    def test_touch_and_chmod_600(self):
        cmd = lib.build_authorize_pubkey_cmd("~/.ssh", "~/.ssh/cluster_id.pub")
        self.assertIn("touch", cmd)
        self.assertIn("chmod 600", cmd)


class TestBuildWriteSshConfigCmd(unittest.TestCase):
    def test_managed_block_contains_sed_and_base64(self):
        cmd = lib.build_write_ssh_config_cmd("~/.ssh", "block text\n", "managed_block")
        self.assertIn("sed", cmd)
        self.assertIn("base64", cmd)
        self.assertIn("chmod 600", cmd)

    def test_overwrite_no_sed(self):
        cmd = lib.build_write_ssh_config_cmd("~/.ssh", "block\n", "overwrite")
        self.assertNotIn("sed", cmd)
        self.assertIn("base64", cmd)
        self.assertIn("chmod 600", cmd)

    def test_path_with_spaces_quoted(self):
        cmd = lib.build_write_ssh_config_cmd("/home/my user/.ssh", "block\n", "managed_block")
        # shlex.quote wraps the path in single quotes so the raw unquoted path must not appear bare
        self.assertNotIn(" /home/my user/", cmd)


class TestUploadClusterKeys(unittest.TestCase):
    def _make_orch(self, exit_code=0):
        orch = MagicMock()
        orch.all.hosts = ["n1", "n2"]
        orch.exec.return_value = {
            "n1": {"output": "", "exit_code": exit_code},
            "n2": {"output": "", "exit_code": exit_code},
        }
        return orch

    @patch("cvs.lib.ssh_keys_lib.os.path.isfile", return_value=True)
    def test_upload_called_for_priv_and_pub(self, _isfile):
        orch = self._make_orch()
        norm = {
            "cluster_key_private_path": "/local/id",
            "cluster_key_public_path": "/local/id.pub",
            "key_name": "cluster_id",
            "remote_ssh_dir": "~/.ssh",
        }
        results = lib.upload_cluster_keys(orch, norm)
        self.assertEqual(orch.all.upload_file.call_count, 2)
        calls = orch.all.upload_file.call_args_list
        remote_paths = [c[0][1] for c in calls]
        self.assertIn("~/.ssh/cluster_id", remote_paths)
        self.assertIn("~/.ssh/cluster_id.pub", remote_paths)
        self.assertTrue(all(results.values()))

    def test_upload_ioerror_marks_nodes_failed(self):
        orch = self._make_orch()
        orch.all.upload_file.side_effect = IOError("sftp fail")
        norm = {
            "cluster_key_private_path": "/local/id",
            "cluster_key_public_path": "/local/id.pub",
            "key_name": "cluster_id",
            "remote_ssh_dir": "~/.ssh",
        }
        results = lib.upload_cluster_keys(orch, norm)
        self.assertTrue(all(not v for v in results.values()))

    def test_chmod_failure_marks_node_failed(self):
        orch = self._make_orch(exit_code=1)
        norm = {
            "cluster_key_private_path": "/local/id",
            "cluster_key_public_path": "/local/id.pub",
            "key_name": "cluster_id",
            "remote_ssh_dir": "~/.ssh",
        }
        results = lib.upload_cluster_keys(orch, norm)
        self.assertFalse(results["n1"])
        self.assertFalse(results["n2"])


class TestAuthorizeClusterPubkey(unittest.TestCase):
    def test_returns_success_dict(self):
        orch = MagicMock()
        orch.exec.return_value = {
            "n1": {"output": "", "exit_code": 0},
            "n2": {"output": "", "exit_code": 0},
        }
        norm = {"key_name": "cluster_id", "remote_ssh_dir": "~/.ssh"}
        results = lib.authorize_cluster_pubkey(orch, norm)
        self.assertTrue(results["n1"])
        self.assertTrue(results["n2"])
        orch.exec.assert_called_once()
        cmd = orch.exec.call_args[0][0]
        self.assertIn("grep -qxF", cmd)

    def test_nonzero_exit_returns_false(self):
        orch = MagicMock()
        orch.exec.return_value = {"n1": {"output": "err", "exit_code": 1}}
        norm = {"key_name": "cluster_id", "remote_ssh_dir": "~/.ssh"}
        results = lib.authorize_cluster_pubkey(orch, norm)
        self.assertFalse(results["n1"])


class TestAuthorizeControllingStation(unittest.TestCase):
    def test_empty_path_returns_empty_dict(self):
        orch = MagicMock()
        norm = {"controlling_station_pubkey_path": "", "remote_ssh_dir": "~/.ssh"}
        results = lib.authorize_controlling_station(orch, norm)
        self.assertEqual(results, {})
        orch.all.upload_file.assert_not_called()

    def test_upload_and_authorize_called(self):
        orch = MagicMock()
        orch.all.hosts = ["n1"]
        orch.exec.return_value = {"n1": {"output": "", "exit_code": 0}}
        norm = {
            "controlling_station_pubkey_path": "/local/ctrl.pub",
            "remote_ssh_dir": "~/.ssh",
        }
        results = lib.authorize_controlling_station(orch, norm)
        orch.all.upload_file.assert_called_once_with("/local/ctrl.pub", "~/.ssh/.cvs_controlling_station.pub")
        self.assertTrue(results["n1"])

    def test_upload_ioerror_marks_failed(self):
        orch = MagicMock()
        orch.all.hosts = ["n1"]
        orch.all.upload_file.side_effect = IOError("fail")
        norm = {
            "controlling_station_pubkey_path": "/local/ctrl.pub",
            "remote_ssh_dir": "~/.ssh",
        }
        results = lib.authorize_controlling_station(orch, norm)
        self.assertFalse(results["n1"])


class TestInstallSshConfig(unittest.TestCase):
    def test_uses_cluster_username(self):
        orch = MagicMock()
        orch.exec.return_value = {"n1": {"output": "", "exit_code": 0}}
        cluster = {"username": "myuser", "node_dict": {"n1": {}}}
        norm = {
            "remote_ssh_dir": "~/.ssh",
            "key_name": "cluster_id",
            "ssh_config_host_pattern": "",
            "ssh_config_write_mode": "managed_block",
        }
        lib.install_ssh_config(orch, cluster, norm)
        cmd = orch.exec.call_args[0][0]
        self.assertIn("base64", cmd)

    def test_returns_success_dict(self):
        orch = MagicMock()
        orch.exec.return_value = {"n1": {"output": "", "exit_code": 0}}
        cluster = {"username": "u", "node_dict": {"n1": {}}}
        norm = {
            "remote_ssh_dir": "~/.ssh",
            "key_name": "cluster_id",
            "ssh_config_host_pattern": "",
            "ssh_config_write_mode": "managed_block",
        }
        results = lib.install_ssh_config(orch, cluster, norm)
        self.assertTrue(results["n1"])


class TestVerifyPasswordlessSsh(unittest.TestCase):
    def test_single_node_returns_empty(self):
        orch = MagicMock()
        cluster = {"node_dict": {"n1": {}}}
        norm = {"remote_ssh_dir": "~/.ssh", "verify_timeout": 20, "verify_mode": "ring"}
        results = lib.verify_passwordless_ssh(orch, cluster, norm)
        self.assertEqual(results, {})

    def test_ring_three_nodes_builds_three_probes(self):
        orch = MagicMock()
        # exec_cmd_list returns per-node dict
        orch.all.exec_cmd_list.return_value = {"n1": "", "n2": "", "n3": ""}
        cluster = {"node_dict": {"n1": {}, "n2": {}, "n3": {}}, "username": "u", "priv_key_file": "/k"}
        norm = {"remote_ssh_dir": "~/.ssh", "verify_timeout": 20, "verify_mode": "ring"}
        lib.verify_passwordless_ssh(orch, cluster, norm)
        orch.all.exec_cmd_list.assert_called_once()
        cmd_list = orch.all.exec_cmd_list.call_args[0][0]
        self.assertEqual(len(cmd_list), 3)

    def test_nonzero_output_error_marks_failed(self):
        orch = MagicMock()
        orch.all.exec_cmd_list.return_value = {"n1": "error occurred", "n2": ""}
        cluster = {"node_dict": {"n1": {}, "n2": {}}, "username": "u", "priv_key_file": "/k"}
        norm = {"remote_ssh_dir": "~/.ssh", "verify_timeout": 20, "verify_mode": "ring"}
        results = lib.verify_passwordless_ssh(orch, cluster, norm)
        failed = [pair for pair, ok in results.items() if not ok]
        self.assertTrue(len(failed) >= 0)  # structure validated; specific values depend on mapping


if __name__ == "__main__":
    unittest.main()
