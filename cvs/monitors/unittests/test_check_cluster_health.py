'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/monitors/check_cluster_health.py.

These cover the cluster-file loader and the CLI argument-resolution logic
so that future drifts in the cluster file shape, the deprecated --hosts_file
fallback, or the credential validation rules are caught here.
'''

import argparse
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from cvs.monitors.check_cluster_health import CheckClusterHealthMonitor, load_cluster_file


def _write(tmpdir, name, contents):
    """Write ``contents`` to ``tmpdir/name`` and return the full path."""
    path = os.path.join(tmpdir, name)
    with open(path, 'w') as f:
        f.write(contents)
    return path


class TestLoadClusterFile(unittest.TestCase):
    """Validate the cluster-file loader used by the monitor."""

    def test_valid_cluster_file_returns_nodes_user_and_key(self):
        cluster = {
            "username": "alice",
            "priv_key_file": "/home/alice/.ssh/id_rsa",
            "node_dict": {
                "10.0.0.1": {"bmc_ip": "NA", "vpc_ip": "10.0.0.1"},
                "10.0.0.2": {"bmc_ip": "NA", "vpc_ip": "10.0.0.2"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", json.dumps(cluster))
            nodes, user, pkey = load_cluster_file(path)
        self.assertEqual(nodes, ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(user, "alice")
        self.assertEqual(pkey, "/home/alice/.ssh/id_rsa")

    def test_user_id_placeholder_is_resolved(self):
        cluster = {
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
            "node_dict": {"host-a": {"bmc_ip": "NA", "vpc_ip": "host-a"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", json.dumps(cluster))
            with patch.dict(os.environ, {"USER": "bob"}, clear=False):
                nodes, user, pkey = load_cluster_file(path)
        self.assertEqual(nodes, ["host-a"])
        self.assertEqual(user, "bob")
        self.assertEqual(pkey, "/home/bob/.ssh/id_rsa")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_cluster_file("/nonexistent/path/cluster.json")

    def test_invalid_json_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", "{not json")
            with self.assertRaises(ValueError):
                load_cluster_file(path)

    def test_missing_username_raises_value_error(self):
        cluster = {
            "priv_key_file": "/k",
            "node_dict": {"h1": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", json.dumps(cluster))
            with self.assertRaisesRegex(ValueError, "username"):
                load_cluster_file(path)

    def test_missing_priv_key_file_raises_value_error(self):
        cluster = {
            "username": "u",
            "node_dict": {"h1": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", json.dumps(cluster))
            with self.assertRaisesRegex(ValueError, "priv_key_file"):
                load_cluster_file(path)

    def test_empty_node_dict_raises_value_error(self):
        cluster = {"username": "u", "priv_key_file": "/k", "node_dict": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "cluster.json", json.dumps(cluster))
            with self.assertRaisesRegex(ValueError, "node_dict"):
                load_cluster_file(path)


class TestArgParser(unittest.TestCase):
    """Pin the CLI surface so a future refactor cannot silently change it."""

    def setUp(self):
        self.parser = CheckClusterHealthMonitor().get_parser()

    def test_cluster_file_alone_is_accepted(self):
        args = self.parser.parse_args(["--cluster_file", "/tmp/c.json"])
        self.assertEqual(args.cluster_file, "/tmp/c.json")
        self.assertIsNone(args.hosts_file)

    def test_hosts_file_alone_is_accepted(self):
        args = self.parser.parse_args(["--hosts_file", "/tmp/h.txt", "--username", "u", "--key_file", "/k"])
        self.assertEqual(args.hosts_file, "/tmp/h.txt")
        self.assertIsNone(args.cluster_file)

    def test_both_cluster_and_hosts_file_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--cluster_file", "/c.json", "--hosts_file", "/h.txt"])

    def test_neither_cluster_nor_hosts_file_parses(self):
        # Argparse no longer enforces "exactly one source" because the
        # CLUSTER_FILE env var can satisfy --cluster_file. Runtime
        # validation lives in _resolve_connection (see TestResolveConnection).
        args = self.parser.parse_args(["--iterations", "1"])
        self.assertIsNone(args.cluster_file)
        self.assertIsNone(args.hosts_file)


class TestResolveConnection(unittest.TestCase):
    """Drive the credential-resolution branch points with synthetic Namespaces."""

    def setUp(self):
        self.monitor = CheckClusterHealthMonitor()
        # Each test gets a clean env so CLUSTER_FILE leakage from the host
        # shell or other tests cannot influence the resolution branches.
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop('CLUSTER_FILE', None)
        self.addCleanup(self._env_patch.stop)

    @staticmethod
    def _ns(**overrides):
        defaults = dict(
            cluster_file=None,
            hosts_file=None,
            username=None,
            password=None,
            key_file=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cluster_file_returns_creds_from_file(self):
        cluster = {
            "username": "u",
            "priv_key_file": "/key",
            "node_dict": {"h1": {}, "h2": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "c.json", json.dumps(cluster))
            nodes, user, pkey, password = self.monitor._resolve_connection(self._ns(cluster_file=path))
        self.assertEqual(nodes, ["h1", "h2"])
        self.assertEqual((user, pkey, password), ("u", "/key", None))

    def test_cluster_file_with_extra_creds_aborts(self):
        cluster = {"username": "u", "priv_key_file": "/k", "node_dict": {"h1": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "c.json", json.dumps(cluster))
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(self._ns(cluster_file=path, username="other"))

    def test_hosts_file_with_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n2.2.2.2\n# comment\n\n")
            nodes, user, pkey, password = self.monitor._resolve_connection(
                self._ns(hosts_file=hosts_path, username="u", key_file="/k")
            )
        self.assertEqual(nodes, ["1.1.1.1", "2.2.2.2"])
        self.assertEqual((user, pkey, password), ("u", "/k", None))

    def test_hosts_file_with_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n")
            nodes, user, pkey, password = self.monitor._resolve_connection(
                self._ns(hosts_file=hosts_path, username="u", password="pw")
            )
        self.assertEqual(nodes, ["1.1.1.1"])
        self.assertEqual((user, pkey, password), ("u", None, "pw"))

    def test_hosts_file_missing_username_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n")
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(self._ns(hosts_file=hosts_path, key_file="/k"))

    def test_hosts_file_both_password_and_key_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n")
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(
                    self._ns(hosts_file=hosts_path, username="u", password="p", key_file="/k")
                )

    def test_hosts_file_neither_password_nor_key_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n")
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(self._ns(hosts_file=hosts_path, username="u"))

    def test_empty_hosts_file_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            hosts_path = _write(tmp, "hosts.txt", "# only a comment\n\n")
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(self._ns(hosts_file=hosts_path, username="u", key_file="/k"))

    def test_cluster_file_env_var_supplies_path(self):
        cluster = {"username": "envuser", "priv_key_file": "/envkey", "node_dict": {"h1": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "c.json", json.dumps(cluster))
            os.environ['CLUSTER_FILE'] = path
            nodes, user, pkey, password = self.monitor._resolve_connection(self._ns())
        self.assertEqual(nodes, ["h1"])
        self.assertEqual((user, pkey, password), ("envuser", "/envkey", None))

    def test_cluster_file_flag_takes_precedence_over_env_var(self):
        # CLI flag wins over CLUSTER_FILE. Pin this so a future refactor
        # cannot silently flip the precedence; it must stay in lockstep
        # with cvs exec / cvs scp.
        env_cluster = {"username": "envuser", "priv_key_file": "/envkey", "node_dict": {"env-host": {}}}
        flag_cluster = {"username": "flaguser", "priv_key_file": "/flagkey", "node_dict": {"flag-host": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            env_path = _write(tmp, "env.json", json.dumps(env_cluster))
            flag_path = _write(tmp, "flag.json", json.dumps(flag_cluster))
            os.environ['CLUSTER_FILE'] = env_path
            nodes, user, pkey, _ = self.monitor._resolve_connection(self._ns(cluster_file=flag_path))
        self.assertEqual(nodes, ["flag-host"])
        self.assertEqual((user, pkey), ("flaguser", "/flagkey"))

    def test_cluster_file_env_var_combined_with_hosts_file_aborts(self):
        cluster = {"username": "u", "priv_key_file": "/k", "node_dict": {"h1": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            cluster_path = _write(tmp, "c.json", json.dumps(cluster))
            hosts_path = _write(tmp, "hosts.txt", "1.1.1.1\n")
            os.environ['CLUSTER_FILE'] = cluster_path
            with self.assertRaises(SystemExit):
                self.monitor._resolve_connection(self._ns(hosts_file=hosts_path, username="u", key_file="/k"))

    def test_no_source_at_all_aborts(self):
        # Neither --cluster_file, nor --hosts_file, nor CLUSTER_FILE in env.
        with self.assertRaises(SystemExit):
            self.monitor._resolve_connection(self._ns())


if __name__ == "__main__":
    unittest.main()
