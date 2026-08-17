import argparse
import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cvs.cli_plugins.exec_plugin import ExecPlugin, _collect_switch_hosts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAIN_CLUSTER = {
    "username": "root",
    "priv_key_file": "/home/user/.ssh/id_rsa",
    "head_node_dict": {"mgmt_ip": "10.0.0.1"},
    "node_dict": {
        "10.0.0.2": {"bmc_ip": "NA", "vpc_ip": "10.0.0.2"},
        "10.0.0.3": {"bmc_ip": "NA", "vpc_ip": "10.0.0.3"},
    },
}

RACK_CLUSTER = {
    "username": "root",
    "priv_key_file": "/home/user/.ssh/id_rsa",
    "head_node_dict": {"mgmt_ip": "10.0.0.1"},
    "node_dict": {
        "10.0.0.2": {"bmc_ip": "NA", "vpc_ip": "10.0.0.2", "rack_id": "rack-01"},
    },
    "racks": {
        "switch_ssh_user": "admin",
        "switch_ssh_password": "password",
        "rack-01": {
            "platform": "HeliosP",
            "switch_trays": ["192.168.1.1", "192.168.1.2"],
        },
        "rack-02": {
            "platform": "HeliosP",
            "switch_trays": ["192.168.2.1"],
        },
    },
}


# ===========================================================================
# _collect_switch_hosts
# ===========================================================================


class TestCollectSwitchHosts(unittest.TestCase):
    # --- No racks block --------------------------------------------------

    def test_no_racks_returns_empty(self):
        hosts = _collect_switch_hosts(PLAIN_CLUSTER)
        self.assertEqual(hosts, [])

    def test_empty_racks_returns_empty(self):
        cluster = dict(PLAIN_CLUSTER, racks={})
        hosts = _collect_switch_hosts(cluster)
        self.assertEqual(hosts, [])

    def test_none_racks_returns_empty(self):
        cluster = dict(PLAIN_CLUSTER, racks=None)
        hosts = _collect_switch_hosts(cluster)
        self.assertEqual(hosts, [])

    # --- Host list correctness -------------------------------------------

    def test_switch_hosts_collected(self):
        hosts = _collect_switch_hosts(RACK_CLUSTER)
        self.assertIn("192.168.1.1", hosts)
        self.assertIn("192.168.1.2", hosts)
        self.assertIn("192.168.2.1", hosts)
        self.assertEqual(len(hosts), 3)

    def test_metadata_keys_skipped(self):
        """Credential keys in racks block are skipped, not treated as rack IDs."""
        hosts = _collect_switch_hosts(RACK_CLUSTER)
        for h in hosts:
            self.assertNotIn("switch_ssh_user", h)
            self.assertNotIn("switch_ssh_password", h)

    def test_rack_without_switch_trays_skipped(self):
        cluster = {
            "username": "root",
            "priv_key_file": "/key",
            "node_dict": {},
            "racks": {
                "rack-01": {"platform": "HeliosP"},
            },
        }
        hosts = _collect_switch_hosts(cluster)
        self.assertEqual(hosts, [])

    # --- Deprecation warning via rack_groups -----------------------------

    def test_rack_groups_key_triggers_deprecation(self):
        """Top-level 'rack_groups' key (the real deprecated scenario) emits a warning."""
        cluster = {
            "username": "root",
            "priv_key_file": "/key",
            "node_dict": {},
            "rack_groups": {  # deprecated top-level key
                "rack-01": {"switch_trays": ["192.168.1.1"]},
            },
        }
        # _collect_switch_hosts itself doesn't emit the warning (exec_plugin.run does),
        # but it still returns the hosts from rack_groups for backward compat.
        hosts = _collect_switch_hosts(cluster)
        self.assertIn("192.168.1.1", hosts)

    def test_racks_key_preferred_over_rack_groups(self):
        """When both 'racks' and 'rack_groups' exist, 'racks' wins."""
        cluster = {
            "username": "root",
            "priv_key_file": "/key",
            "node_dict": {},
            "racks": {
                "rack-01": {"switch_trays": ["10.0.0.1"]},
            },
            "rack_groups": {
                "rack-99": {"switch_trays": ["99.99.99.99"]},
            },
        }
        hosts = _collect_switch_hosts(cluster)
        self.assertIn("10.0.0.1", hosts)
        self.assertNotIn("99.99.99.99", hosts)


# ===========================================================================
# ExecPlugin.run — argument / file loading errors
# ===========================================================================


class TestExecPluginRunErrors(unittest.TestCase):
    def setUp(self):
        self.plugin = ExecPlugin()

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.cluster_file = kwargs.get("cluster_file", None)
        args.cmd = kwargs.get("cmd", "hostname")
        args.target = kwargs.get("target", "computes")
        args.timeout = kwargs.get("timeout", 30)
        args.connect_timeout = kwargs.get("connect_timeout", 15)
        args.json_output = kwargs.get("json_output", False)
        args.verbose = kwargs.get("verbose", False)
        return args

    def test_no_cluster_file_exits(self):
        args = self._make_args()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLUSTER_FILE", None)
            with self.assertRaises(SystemExit) as cm:
                self.plugin.run(args)
            self.assertEqual(cm.exception.code, 1)

    def test_cluster_file_from_env_var(self):
        """CLUSTER_FILE env var is used when --cluster_file is not passed."""
        cluster_json = json.dumps(PLAIN_CLUSTER)
        args = self._make_args(cluster_file=None)
        with patch.dict(os.environ, {"CLUSTER_FILE": "/fake/cluster.json"}):
            with patch("builtins.open", mock_open(read_data=cluster_json)):
                with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})):
                    self.plugin.run(args)  # Should not raise

    def test_file_not_found_exits(self):
        args = self._make_args(cluster_file="/nonexistent/cluster.json")
        with self.assertRaises(SystemExit) as cm:
            self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)

    def test_invalid_json_exits(self):
        args = self._make_args(cluster_file="/fake/cluster.json")
        with patch("builtins.open", mock_open(read_data="{not valid json")):
            with self.assertRaises(SystemExit) as cm:
                self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)

    def test_missing_username_exits(self):
        bad = dict(PLAIN_CLUSTER)
        bad.pop("username")
        args = self._make_args(cluster_file="/fake/cluster.json")
        with patch("builtins.open", mock_open(read_data=json.dumps(bad))):
            with self.assertRaises(SystemExit) as cm:
                self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)

    def test_missing_priv_key_for_computes_exits(self):
        bad = dict(PLAIN_CLUSTER)
        bad.pop("priv_key_file")
        args = self._make_args(cluster_file="/fake/cluster.json", target="computes")
        with patch("builtins.open", mock_open(read_data=json.dumps(bad))):
            with self.assertRaises(SystemExit) as cm:
                self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)

    def test_empty_node_dict_exits(self):
        bad = dict(PLAIN_CLUSTER, node_dict={})
        args = self._make_args(cluster_file="/fake/cluster.json", target="computes")
        with patch("builtins.open", mock_open(read_data=json.dumps(bad))):
            with self.assertRaises(SystemExit) as cm:
                self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)


# ===========================================================================
# ExecPlugin.run — target routing
# ===========================================================================


class TestExecPluginRunTargets(unittest.TestCase):
    def setUp(self):
        self.plugin = ExecPlugin()

    def _make_args(
        self, target="computes", timeout=30, connect_timeout=15, cmd="hostname", json_output=False, verbose=False
    ):
        args = MagicMock()
        args.cluster_file = "/fake/cluster.json"
        args.cmd = cmd
        args.target = target
        args.timeout = timeout
        args.connect_timeout = connect_timeout
        args.json_output = json_output
        args.verbose = verbose
        return args

    def _run_with_cluster(self, cluster, args):
        cluster_json = json.dumps(cluster)
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})) as mock_run:
                self.plugin.run(args)
        return mock_run

    # --- target=computes (default) ---------------------------------------

    def test_computes_calls_run_on_hosts_once(self):
        mock_run = self._run_with_cluster(PLAIN_CLUSTER, self._make_args(target="computes"))
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs['label'], "compute")

    def test_computes_passes_correct_hosts(self):
        mock_run = self._run_with_cluster(PLAIN_CLUSTER, self._make_args(target="computes"))
        hosts_arg = mock_run.call_args[0][0]
        self.assertIn("10.0.0.2", hosts_arg)
        self.assertIn("10.0.0.3", hosts_arg)

    def test_computes_passes_timeout(self):
        mock_run = self._run_with_cluster(PLAIN_CLUSTER, self._make_args(target="computes", timeout=60))
        self.assertEqual(mock_run.call_args.kwargs.get("timeout"), 60)

    def test_computes_passes_connect_timeout(self):
        mock_run = self._run_with_cluster(PLAIN_CLUSTER, self._make_args(target="computes", connect_timeout=5))
        self.assertEqual(mock_run.call_args.kwargs.get("connect_timeout"), 5)

    # --- target=switches -------------------------------------------------

    def test_switches_no_racks_prints_warning(self):
        args = self._make_args(target="switches")
        cluster_json = json.dumps(PLAIN_CLUSTER)
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})) as mock_run:
                with patch("builtins.print") as mock_print:
                    self.plugin.run(args)
        mock_run.assert_not_called()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Warning", printed)

    def test_switches_with_racks_calls_run_on_hosts(self):
        mock_run = self._run_with_cluster(RACK_CLUSTER, self._make_args(target="switches"))
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs['label'], "switch")

    def test_switches_with_racks_passes_correct_hosts(self):
        mock_run = self._run_with_cluster(RACK_CLUSTER, self._make_args(target="switches"))
        hosts_arg = mock_run.call_args[0][0]
        self.assertIn("192.168.1.1", hosts_arg)
        self.assertIn("192.168.2.1", hosts_arg)

    def test_switches_passes_global_user(self):
        """Global switch_ssh_user from racks block is passed as username kwarg."""
        mock_run = self._run_with_cluster(RACK_CLUSTER, self._make_args(target="switches"))
        self.assertEqual(mock_run.call_args[0][1], "admin")

    def test_switches_passes_global_password(self):
        """Global switch_ssh_password is passed when no key file."""
        mock_run = self._run_with_cluster(RACK_CLUSTER, self._make_args(target="switches"))
        self.assertEqual(mock_run.call_args.kwargs.get("password"), "password")

    def test_switches_key_file_suppresses_password(self):
        """When switch_ssh_key_file is set, password must be None."""
        cluster = {
            "username": "root",
            "priv_key_file": "/key",
            "node_dict": {},
            "racks": {
                "switch_ssh_user": "admin",
                "switch_ssh_key_file": "/switch_key",
                "switch_ssh_password": "should_be_ignored",
                "rack-01": {"switch_trays": ["192.168.1.1"]},
            },
        }
        mock_run = self._run_with_cluster(cluster, self._make_args(target="switches"))
        self.assertEqual(mock_run.call_args.kwargs.get("pkey"), "/switch_key")
        self.assertIsNone(mock_run.call_args.kwargs.get("password"))

    def test_deprecation_warning_fired_for_rack_groups(self):
        """Top-level 'rack_groups' key (deprecated) triggers DeprecationWarning."""
        cluster = {
            "username": "root",
            "priv_key_file": "/key",
            "node_dict": {},
            "rack_groups": {
                "switch_ssh_user": "admin",
                "rack-01": {"switch_trays": ["192.168.1.1"]},
            },
        }
        cluster_json = json.dumps(cluster)
        args = self._make_args(target="switches")
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    self.plugin.run(args)
        categories = [str(x.category) for x in w]
        self.assertTrue(
            any("DeprecationWarning" in c for c in categories),
            f"Expected DeprecationWarning, got: {categories}",
        )

    # --- target=all ------------------------------------------------------

    def test_all_calls_run_on_hosts_twice(self):
        mock_run = self._run_with_cluster(RACK_CLUSTER, self._make_args(target="all"))
        self.assertEqual(mock_run.call_count, 2)
        labels = {call.kwargs['label'] for call in mock_run.call_args_list}
        self.assertEqual(labels, {"compute", "switch"})

    def test_all_without_racks_calls_only_compute(self):
        """target=all on a plain cluster should run computes + warn about switches."""
        args = self._make_args(target="all")
        cluster_json = json.dumps(PLAIN_CLUSTER)
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})) as mock_run:
                with patch("builtins.print"):
                    self.plugin.run(args)
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs['label'], "compute")

    # --- exit code behavior ----------------------------------------------

    def test_ssh_failure_exits_nonzero(self):
        """_run_on_hosts returning (False, {}) must cause sys.exit(1)."""
        cluster_json = json.dumps(PLAIN_CLUSTER)
        args = self._make_args(target="computes")
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(False, {})):
                with self.assertRaises(SystemExit) as cm:
                    self.plugin.run(args)
        self.assertEqual(cm.exception.code, 1)

    def test_ssh_success_does_not_exit(self):
        """_run_on_hosts returning (True, {}) must not raise SystemExit."""
        cluster_json = json.dumps(PLAIN_CLUSTER)
        args = self._make_args(target="computes")
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, {})):
                self.plugin.run(args)  # no exception


# ===========================================================================
# ExecPlugin._run_on_hosts — SSH error handling
# ===========================================================================


class TestRunOnHosts(unittest.TestCase):
    def setUp(self):
        self.plugin = ExecPlugin()

    def _call(self, hosts=None, **kwargs):
        return self.plugin._run_on_hosts(
            hosts or ["10.0.0.2"],
            "root",
            None,
            "hostname",
            label=kwargs.pop("label", "compute"),
            pkey=kwargs.pop("pkey", "/key"),
            **kwargs,
        )

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_pssh_init_error_returns_false(self, MockPssh):
        MockPssh.side_effect = RuntimeError("connection refused")
        ok, output = self._call()
        self.assertFalse(ok)
        self.assertEqual(output, {})

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_pssh_exec_error_returns_false(self, MockPssh):
        mock_pssh = MagicMock()
        mock_pssh.exec.side_effect = RuntimeError("timeout")
        MockPssh.return_value = mock_pssh
        ok, output = self._call()
        self.assertFalse(ok)
        self.assertEqual(output, {})

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_success_returns_true(self, MockPssh):
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {"10.0.0.2": "XSJICHRISTO01"}
        MockPssh.return_value = mock_pssh
        ok, output = self._call()
        self.assertTrue(ok)
        self.assertEqual(output, {"10.0.0.2": "XSJICHRISTO01"})

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_timeout_passed_to_exec(self, MockPssh):
        """--timeout (command read timeout) is passed to pssh.exec(), not the Pssh constructor."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call(timeout=45)
        mock_pssh.exec.assert_called_once_with("hostname", timeout=45)
        # Must NOT appear in the Pssh constructor kwargs
        init_kwargs = MockPssh.call_args[1]
        self.assertNotEqual(init_kwargs.get("timeout"), 45)

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_connect_timeout_passed_to_pssh_ctor(self, MockPssh):
        """--connect-timeout is forwarded to the Pssh constructor as 'timeout' (ParallelSSHClient connection timeout)."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call(connect_timeout=10)
        init_kwargs = MockPssh.call_args[1]
        self.assertEqual(init_kwargs.get("timeout"), 10)

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_connect_timeout_none_omitted_from_ctor(self, MockPssh):
        """When connect_timeout is None, 'timeout' is not passed to the Pssh constructor."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call(connect_timeout=None)
        init_kwargs = MockPssh.call_args[1]
        self.assertNotIn("timeout", init_kwargs)

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_num_retries_is_zero(self, MockPssh):
        """num_retries=0 ensures a single SSH attempt, preventing hangs on dead hosts."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call()
        init_kwargs = MockPssh.call_args[1]
        self.assertEqual(init_kwargs.get("num_retries"), 0)

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_password_auth_path(self, MockPssh):
        """When password provided, Pssh is initialized with password kwarg."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call(pkey=None, password="secret")
        init_kwargs = MockPssh.call_args[1]
        self.assertEqual(init_kwargs.get("password"), "secret")
        self.assertIsNone(init_kwargs.get("pkey"))

    @patch("cvs.cli_plugins.exec_plugin.Pssh")
    def test_pkey_auth_path(self, MockPssh):
        """Without password, Pssh is initialized with pkey kwarg."""
        mock_pssh = MagicMock()
        mock_pssh.exec.return_value = {}
        MockPssh.return_value = mock_pssh
        self._call(pkey="/key", password=None)
        init_kwargs = MockPssh.call_args[1]
        self.assertEqual(init_kwargs.get("pkey"), "/key")
        self.assertIsNone(init_kwargs.get("password"))


# ===========================================================================
# ExecPlugin metadata
# ===========================================================================


class TestExecPluginMetadata(unittest.TestCase):
    def setUp(self):
        self.plugin = ExecPlugin()

    def test_name(self):
        self.assertEqual(self.plugin.get_name(), "exec")

    def test_epilog_contains_examples(self):
        epilog = self.plugin.get_epilog()
        self.assertIn("cvs exec", epilog)
        self.assertIn("--target", epilog)

    def test_parser_registers_required_args(self):
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = main_parser.parse_args(["exec", "--cmd", "hostname"])
        self.assertEqual(args.cmd, "hostname")
        self.assertEqual(args.target, "computes")
        self.assertEqual(args.timeout, 30)
        self.assertEqual(args.connect_timeout, 15)
        self.assertFalse(args.json_output)
        self.assertFalse(args.verbose)

    def test_verbose_flag_default_is_false(self):
        """--verbose defaults to False (SSH diagnostics are suppressed by default)."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = main_parser.parse_args(["exec", "--cmd", "hostname"])
        self.assertFalse(args.verbose)

    def test_verbose_flag_short_form(self):
        """-v is the short form for --verbose."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = main_parser.parse_args(["exec", "--cmd", "hostname", "-v"])
        self.assertTrue(args.verbose)

    def test_verbose_flag_long_form(self):
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = main_parser.parse_args(["exec", "--cmd", "hostname", "--verbose"])
        self.assertTrue(args.verbose)

    def test_parser_rejects_invalid_target(self):
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        with self.assertRaises(SystemExit):
            main_parser.parse_args(["exec", "--cmd", "hostname", "--target", "invalid"])


# ===========================================================================
# --json output mode
# ===========================================================================


class TestJsonOutput(unittest.TestCase):
    def setUp(self):
        self.plugin = ExecPlugin()

    def _make_args(
        self, target="computes", timeout=30, connect_timeout=15, cmd="hostname", json_output=False, verbose=False
    ):
        args = MagicMock()
        args.cluster_file = "/fake/cluster.json"
        args.cmd = cmd
        args.target = target
        args.timeout = timeout
        args.connect_timeout = connect_timeout
        args.json_output = json_output
        args.verbose = verbose
        return args

    def test_json_flag_emits_json_to_stdout(self):
        """--json prints a valid JSON envelope to stdout with the expected schema."""
        host_output = {"10.0.0.2": "hostname_result\n", "10.0.0.3": "hostname_result\n"}
        args = self._make_args(cmd="hostname", timeout=30, connect_timeout=15, json_output=True)
        cluster_json = json.dumps(PLAIN_CLUSTER)
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, host_output)):
                with patch("builtins.print") as mock_print:
                    self.plugin.run(args)

        printed_args = [call.args[0] for call in mock_print.call_args_list if call.args]
        self.assertEqual(len(printed_args), 1, "Expected exactly one print() call for JSON output")
        envelope = json.loads(printed_args[0])
        self.assertEqual(envelope["command"], "hostname")
        self.assertEqual(envelope["read_timeout"], 30)
        self.assertEqual(envelope["connect_timeout"], 15)
        self.assertIn("output", envelope)
        self.assertEqual(envelope["output"], host_output)

    def test_json_output_merges_compute_and_switch(self):
        """With --target all and --json, compute and switch host outputs are merged under 'output'."""
        compute_output = {"10.0.0.2": "compute_result\n"}
        switch_output = {"192.168.1.1": "switch_result\n"}
        args = self._make_args(target="all", json_output=True)
        cluster_json = json.dumps(RACK_CLUSTER)

        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            return (True, compute_output) if call_count == 1 else (True, switch_output)

        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", side_effect=side_effect):
                with patch("builtins.print") as mock_print:
                    self.plugin.run(args)

        printed_args = [call.args[0] for call in mock_print.call_args_list if call.args]
        envelope = json.loads(printed_args[0])
        self.assertIn("10.0.0.2", envelope["output"])
        self.assertIn("192.168.1.1", envelope["output"])

    def test_json_flag_default_is_false(self):
        """--json flag defaults to False (text mode is the default)."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = main_parser.parse_args(["exec", "--cmd", "hostname"])
        self.assertFalse(args.json_output)

    def test_text_mode_unchanged(self):
        """Without --json, per-host text format is still printed (not JSON)."""
        host_output = {"10.0.0.2": "myhost\n"}
        args = self._make_args(json_output=False)
        cluster_json = json.dumps(PLAIN_CLUSTER)
        with patch("builtins.open", mock_open(read_data=cluster_json)):
            with patch.object(self.plugin, "_run_on_hosts", return_value=(True, host_output)):
                with patch("builtins.print") as mock_print:
                    self.plugin.run(args)

        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("[compute] Host:", all_printed)
        # Must NOT be a JSON envelope
        for call in mock_print.call_args_list:
            if call.args:
                try:
                    parsed = json.loads(call.args[0])
                    self.assertNotIn("command", parsed, "Unexpected JSON envelope in text mode")
                except (json.JSONDecodeError, TypeError):
                    pass  # non-JSON output is expected in text mode


if __name__ == "__main__":
    unittest.main()
