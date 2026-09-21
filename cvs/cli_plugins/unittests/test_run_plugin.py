import argparse
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import shutil
import tempfile
import json
from pathlib import Path

# Add the parent directory to sys.path to import cli_plugins
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cvs.cli_plugins.run_plugin import RunPlugin, resolve_test_function_name, resolve_test_function_names


class TestRunPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = RunPlugin()
        # run() resolves the run layout before handing off to pytest. Stub it so
        # these tests create no directories and stay independent of the ambient
        # scheduler environment.
        patcher = patch("cvs.cli_plugins.run_plugin.RunLayout")
        mock_layout = patcher.start()
        self._run_dir = tempfile.mkdtemp()
        mock_layout.get.return_value.run_dir = self._run_dir
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self._run_dir, True)
        self.addCleanup(os.environ.pop, "CVS_WORKSPACE", None)
        os.environ.pop("CVS_WORKSPACE", None)

    def _base_args(self):
        return argparse.Namespace(
            test="agfhc_cvs",
            function=["test_func"],
            cluster_file="/path/to/cluster.json",
            config_file="/path/to/config.json",
            html=None,
            no_html=False,
            self_contained_html=False,
            log_file="/tmp/test.log",
            no_log_file=False,
            log_level=None,
            capture="tee-sys",
            extra_pytest_args=[],
            workspace=None,
        )

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_single_function(self, mock_exit, mock_pytest_main):
        """Test running a single test function"""
        args = self._base_args()
        args.function = ["test_func"]

        mock_pytest_main.return_value = 0  # Mock successful pytest run

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        # Verify pytest.main was called with correct arguments
        expected_args = [
            "/mock/path/test.py::test_func",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
            f"--html={self._run_dir}/test.html",
            "--self-contained-html",
            "--log-file=/tmp/test.log",
            "--capture=tee-sys",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_multiple_functions(self, mock_exit, mock_pytest_main):
        """Test running multiple test functions"""
        args = self._base_args()
        args.function = ["test_func1", "test_func2", "test_func3"]

        mock_pytest_main.return_value = 0

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        # Verify pytest.main was called with multiple function targets
        expected_args = [
            "/mock/path/test.py::test_func1",
            "/mock/path/test.py::test_func2",
            "/mock/path/test.py::test_func3",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
            f"--html={self._run_dir}/test.html",
            "--self-contained-html",
            "--log-file=/tmp/test.log",
            "--capture=tee-sys",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_no_log_file_flag_suppresses_log(self, mock_exit, mock_pytest_main):
        """--no-log-file and --no-html suppress auto-derived reports."""
        args = self._base_args()
        args.function = []
        args.log_file = None
        args.no_log_file = True
        args.no_html = True
        args.capture = None

        mock_pytest_main.return_value = 0

        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

        expected_args = [
            "/mock/path/test.py",
            "--cluster_file=/path/to/cluster.json",
            "--config_file=/path/to/config.json",
        ]
        mock_pytest_main.assert_called_once_with(expected_args)
        mock_exit.assert_called_once_with(0)


class TestRunPluginJsonValidation(unittest.TestCase):
    """Tests for RunPlugin._validate_json_config pre-flight checks."""

    def setUp(self):
        self.plugin = RunPlugin()

    @patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1))
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_missing_file(self, mock_print, mock_exit):
        """A missing config file should print a clean error and exit."""
        with self.assertRaises(SystemExit) as ctx:
            self.plugin._validate_json_config("/nonexistent/path.json", "--cluster_file")
        self.assertEqual(ctx.exception.code, 1)
        printed = " ".join(str(c) for c in mock_print.call_args_list[0][0])
        self.assertIn("does not exist", printed)
        self.assertIn("/nonexistent/path.json", printed)

    @patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1))
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_malformed_json(self, mock_print, mock_exit):
        """A malformed JSON file should print a clean error and exit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self.plugin._validate_json_config(path, "--config_file")
        finally:
            os.unlink(path)
        self.assertEqual(ctx.exception.code, 1)
        messages = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertIn("is not valid JSON", messages)
        self.assertIn(path, messages)

    def test_valid_json(self):
        """A valid JSON file should pass validation without exiting."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"host": {}}, f)
            path = f.name
        try:
            self.plugin._validate_json_config(path, "--config_file")
        finally:
            os.unlink(path)


class TestRunPluginWorkspace(unittest.TestCase):
    """--workspace and the RunLayout handoff.

    Worker ranks in a Slurm/Spur job never enter pytest, so the run layout has
    to be resolved by the CLI before pytest.main() is reached.
    """

    def setUp(self):
        self.plugin = RunPlugin()
        self.addCleanup(os.environ.pop, "CVS_WORKSPACE", None)
        os.environ.pop("CVS_WORKSPACE", None)

    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        return parser.parse_args(["run", "health", "--cluster_file", "c.json", "--config_file", "f.json"] + extra)

    def test_parser_accepts_workspace(self):
        self.assertEqual(self._parse(["--workspace", "/shared/ws"]).workspace, "/shared/ws")

    def test_workspace_defaults_to_none(self):
        self.assertIsNone(self._parse([]).workspace)

    def test_parser_allows_omitted_cluster_file(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        self.plugin.get_parser(subparsers)
        args = parser.parse_args(["run", "health", "--config_file", "f.json"])
        self.assertIsNone(args.cluster_file)

    def test_parser_accepts_no_html_and_no_log_file(self):
        args = self._parse(["--no-html", "--no-log-file"])
        self.assertTrue(args.no_html)
        self.assertTrue(args.no_log_file)

    def _make_args(self, workspace, test_name="agfhc_cvs"):
        return argparse.Namespace(
            test=test_name,
            function=[],
            cluster_file="/path/to/cluster.json",
            config_file="/path/to/config.json",
            html=None,
            no_html=False,
            self_contained_html=False,
            log_file=None,
            no_log_file=False,
            log_level=None,
            capture=None,
            extra_pytest_args=[],
            workspace=workspace,
        )

    def _bind_run_dir(self, mock_layout):
        run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda p=run_dir: shutil.rmtree(p, ignore_errors=True))
        mock_layout.get.return_value.run_dir = run_dir
        return run_dir

    def _run_with_workspace(self, workspace, mock_pytest_main, test_name="agfhc_cvs", mock_layout=None):
        args = self._make_args(workspace, test_name)
        mock_pytest_main.return_value = 0
        if mock_layout is not None:
            self._bind_run_dir(mock_layout)
        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_with_workspace(self, mock_exit, mock_pytest_main, mock_layout):
        self._run_with_workspace("/shared/ws", mock_pytest_main, mock_layout=mock_layout)
        mock_layout.get.assert_called_once_with("/shared/ws")

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_with_none_when_not_given(self, mock_exit, mock_pytest_main, mock_layout):
        self._run_with_workspace(None, mock_pytest_main)
        mock_layout.get.assert_called_once_with(None)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_before_pytest_runs(self, mock_exit, mock_pytest_main, mock_layout):
        # Ordering is the whole point: the layout must be resolved and the agent
        # directory must exist before any fixture or agent looks for them.
        manager = MagicMock()
        manager.attach_mock(mock_layout.get, "get_layout")
        manager.attach_mock(mock_pytest_main, "pytest_main")
        self._run_with_workspace("/shared/ws", mock_pytest_main, mock_layout=mock_layout)
        called = [name for name, _args, _kwargs in manager.mock_calls]
        self.assertLess(called.index("get_layout"), called.index("pytest_main"))

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_workspace_is_not_forwarded_to_pytest(self, mock_exit, mock_pytest_main, mock_layout):
        # The layout reaches suites through RunLayout, not as a pytest option.
        # Matched on substring rather than one exact literal, so forwarding it as
        # a separate ["--workspace", value] pair is caught too.
        self._run_with_workspace("/shared/ws", mock_pytest_main, mock_layout=mock_layout)
        forwarded = mock_pytest_main.call_args[0][0]
        self.assertEqual([arg for arg in forwarded if "workspace" in arg], [])

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_layout_resolved_outside_the_pytest_handoff(self, mock_exit, mock_layout):
        # Worker ranks never enter pytest, so resolution cannot sit in the path
        # that launches it. Pinning it to run() keeps it ahead of any future
        # branch that sends a rank somewhere other than the pytest handoff.
        args = self._make_args("/shared/ws")
        self._bind_run_dir(mock_layout)
        with patch.object(self.plugin, "run_test") as mock_run_test:
            with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
                with patch.object(self.plugin, "_validate_json_config"):
                    self.plugin.run(args)
        mock_layout.get.assert_called_once_with("/shared/ws")
        mock_run_test.assert_called_once()

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    def test_unknown_test_creates_no_run_directory(self, mock_pytest_main, mock_layout):
        # The workspace is shared storage on a real cluster, so a mistyped suite
        # name must not leave an empty run tree behind on it. sys.exit has to
        # raise here as it really does, or the early return does not happen.
        with patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit):
                self._run_with_workspace("/shared/ws", mock_pytest_main, test_name="no_such_suite_xyz")
        mock_layout.get.assert_not_called()
        mock_pytest_main.assert_not_called()

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_unusable_workspace_exits_cleanly(self, mock_print, mock_pytest_main, mock_layout):
        # RunLayout raises RuntimeError for an unwritable or non-venv workspace.
        # The user should get that message, not a traceback, and pytest must not run.
        mock_layout.get.side_effect = RuntimeError("workspace is not writable")
        with patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1)) as mock_exit:
            with self.assertRaises(SystemExit):
                self._run_with_workspace("/shared/ws", mock_pytest_main)
        mock_exit.assert_called_once_with(1)
        mock_pytest_main.assert_not_called()
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call[0])
        self.assertIn("workspace is not writable", printed)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_workspace_auto_derives_html_and_log_file(self, mock_exit, mock_pytest_main, mock_layout):
        mock_pytest_main.return_value = 0
        run_dir = self._bind_run_dir(mock_layout)
        args = self._make_args("/shared/ws")
        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/health.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)
        forwarded = mock_pytest_main.call_args[0][0]
        self.assertIn(f"--html={run_dir}/health.html", forwarded)
        self.assertIn("--self-contained-html", forwarded)
        self.assertIn(f"--log-file={run_dir}/health.log", forwarded)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_no_workspace_still_auto_derives_reports(self, mock_exit, mock_pytest_main, mock_layout):
        mock_pytest_main.return_value = 0
        run_dir = self._bind_run_dir(mock_layout)
        args = self._make_args(None)
        with patch.object(self.plugin, "get_test_file", return_value="/mock/path/health.py"):
            with patch.object(self.plugin, "_validate_json_config"):
                self.plugin.run(args)
        forwarded = mock_pytest_main.call_args[0][0]
        self.assertIn(f"--html={run_dir}/health.html", forwarded)
        self.assertIn("--self-contained-html", forwarded)
        self.assertIn(f"--log-file={run_dir}/health.log", forwarded)

    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.print")
    def test_conflicting_html_flags_exit_1(self, mock_print, mock_pytest_main, mock_layout):
        self._bind_run_dir(mock_layout)
        args = self._make_args("/shared/ws")
        args.html = "/tmp/report.html"
        args.no_html = True
        with patch("cvs.cli_plugins.run_plugin.sys.exit", side_effect=SystemExit(1)) as mock_exit:
            with patch.object(self.plugin, "get_test_file", return_value="/mock/path/test.py"):
                with patch.object(self.plugin, "_validate_json_config"):
                    with self.assertRaises(SystemExit):
                        self.plugin.run(args)
        mock_exit.assert_called_once_with(1)
        mock_pytest_main.assert_not_called()
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call[0])
        self.assertIn("cannot combine --html and --no-html", printed)


class TestResolveReportPaths(unittest.TestCase):
    def setUp(self):
        self.plugin = RunPlugin()
        self.addCleanup(os.environ.pop, "CVS_WORKSPACE", None)
        os.environ.pop("CVS_WORKSPACE", None)

    def _args(self, **kwargs):
        ns = argparse.Namespace(
            html=None,
            no_html=False,
            self_contained_html=False,
            log_file=None,
            no_log_file=False,
            workspace=None,
        )
        for key, value in kwargs.items():
            setattr(ns, key, value)
        return ns

    def test_derives_stem_paths_and_self_contained_from_run_dir(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/ws/cvs_runs/1", "/mock/path/health.py", self._args()
        )
        self.assertEqual(html, "/ws/cvs_runs/1/health.html")
        self.assertEqual(log_file, "/ws/cvs_runs/1/health.log")
        self.assertTrue(contained)

    def test_derives_paths_from_run_dir_regardless_of_workspace_env(self):
        os.environ["CVS_WORKSPACE"] = "/from-env"
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/unrelated/cvs_runs/9", "/suite/agfhc_cvs.py", self._args()
        )
        self.assertEqual(html, "/unrelated/cvs_runs/9/agfhc_cvs.html")
        self.assertEqual(log_file, "/unrelated/cvs_runs/9/agfhc_cvs.log")
        self.assertTrue(contained)

    def test_no_workspace_still_auto_derives(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/venv-parent/cvs_runs/1", "/mock/path/health.py", self._args()
        )
        self.assertEqual(html, "/venv-parent/cvs_runs/1/health.html")
        self.assertEqual(log_file, "/venv-parent/cvs_runs/1/health.log")
        self.assertTrue(contained)

    def test_explicit_paths_override_auto_derive(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/ws/cvs_runs/1",
            "/mock/path/health.py",
            self._args(workspace="/ws", html="/tmp/out.html", log_file="/tmp/out.log"),
        )
        self.assertEqual(html, "/tmp/out.html")
        self.assertEqual(log_file, "/tmp/out.log")
        self.assertFalse(contained)

    def test_no_html_and_no_log_file_suppress_auto_derive(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/ws/cvs_runs/1",
            "/mock/path/health.py",
            self._args(workspace="/ws", no_html=True, no_log_file=True, self_contained_html=True),
        )
        self.assertIsNone(html)
        self.assertIsNone(log_file)
        self.assertFalse(contained)

    def test_no_html_keeps_auto_log_file(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/ws/cvs_runs/1", "/mock/path/health.py", self._args(workspace="/ws", no_html=True)
        )
        self.assertIsNone(html)
        self.assertEqual(log_file, "/ws/cvs_runs/1/health.log")
        self.assertFalse(contained)

    def test_html_conflict_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.plugin._resolve_report_paths("/ws/run", "/t.py", self._args(html="/tmp/a.html", no_html=True))
        self.assertIn("--html", str(ctx.exception))

    def test_log_file_conflict_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.plugin._resolve_report_paths("/ws/run", "/t.py", self._args(log_file="/tmp/a.log", no_log_file=True))
        self.assertIn("--log-file", str(ctx.exception))

    def test_blank_values_do_not_count(self):
        html, log_file, contained = self.plugin._resolve_report_paths(
            "/ws/cvs_runs/1",
            "/mock/path/health.py",
            self._args(workspace="  ", html="   ", log_file=""),
        )
        self.assertEqual(html, "/ws/cvs_runs/1/health.html")
        self.assertEqual(log_file, "/ws/cvs_runs/1/health.log")
        self.assertTrue(contained)


class TestManagedRunPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = RunPlugin()
        self.args = argparse.Namespace(
            test="health",
            function=[],
            cluster_file=None,
            config_file="/path/to/config.json",
            html=None,
            no_html=False,
            self_contained_html=False,
            log_file=None,
            no_log_file=False,
            log_level=None,
            capture=None,
            extra_pytest_args=[],
            workspace="/shared/workspace",
        )

    def _prepare_run(self):
        return (
            patch.object(self.plugin, "_validate_json_config"),
            patch.object(self.plugin, "_resolve_test_file", return_value="/mock/path/test.py"),
        )

    def _layout(self, root):
        layout = MagicMock()
        layout.workspace = Path(root)
        layout.run_dir = Path(root) / "run"
        layout.agent_dir = layout.run_dir / "agent"
        layout.agent_dir.mkdir(parents=True)
        return layout

    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    @patch("cvs.cli_plugins.run_plugin.is_managed_compute", return_value=False)
    def test_unmanaged_run_requires_cluster_file(self, _managed, mock_exit):
        self.plugin.run(self.args)
        mock_exit.assert_called_once_with(1)

    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    @patch("cvs.cli_plugins.run_plugin.AgentRunner")
    @patch("cvs.cli_plugins.run_plugin.is_managed_compute", return_value=True)
    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    def test_worker_never_enters_pytest(self, mock_layout, _managed, mock_agent_class, mock_exit):
        mock_layout.get.return_value.agent_dir = "/shared/workspace/agent"
        runner = mock_agent_class.return_value
        runner.is_rank0 = False
        runner.start.return_value = 0
        validate, resolve = self._prepare_run()
        with (
            validate,
            resolve,
            patch.object(self.plugin, "run_test") as mock_run_test,
            patch.object(self.plugin, "_run_pytest") as mock_run_pytest,
        ):
            self.plugin.run(self.args)
        mock_agent_class.assert_called_once()
        runner.start.assert_called_once()
        mock_run_test.assert_not_called()
        mock_run_pytest.assert_not_called()
        mock_exit.assert_called_once_with(0)

    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    @patch("cvs.cli_plugins.run_plugin.AgentRunner")
    @patch("cvs.cli_plugins.run_plugin.is_managed_compute", return_value=True)
    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    def test_rank0_writes_agent_cluster_and_runs_pytest(self, mock_layout, _managed, mock_agent_class, mock_exit):
        with tempfile.TemporaryDirectory() as root:
            layout = self._layout(root)
            mock_layout.get.return_value = layout
            generated = layout.run_dir / "cluster_agents.json"
            generated.write_text('{"node_dict": {"node01": {"agent_port": 9000}}}\n')
            runner = mock_agent_class.return_value
            runner.is_rank0 = True
            runner.wait.return_value = str(generated)
            validate, resolve = self._prepare_run()
            with validate, resolve, patch.object(self.plugin, "run_test", return_value=7) as mock_run_test:
                self.plugin.run(self.args)

            runner.wait.assert_called_once()
            self.assertEqual(mock_run_test.call_args.args[2], str(generated))
            runner.stop.assert_called_once()
            mock_exit.assert_called_once_with(7)

    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    @patch("cvs.cli_plugins.run_plugin.AgentRunner")
    @patch("cvs.cli_plugins.run_plugin.is_managed_compute", return_value=True)
    @patch("cvs.cli_plugins.run_plugin.RunLayout")
    def test_registration_timeout_with_missing_agent_does_not_run_pytest(
        self, mock_layout, _managed, mock_agent_class, mock_exit
    ):
        with tempfile.TemporaryDirectory() as root:
            mock_layout.get.return_value = self._layout(root)
            runner = mock_agent_class.return_value
            runner.is_rank0 = True
            runner.wait.side_effect = ValueError("agents did not register")
            validate, resolve = self._prepare_run()
            with validate, resolve, patch.object(self.plugin, "run_test") as mock_run_test:
                self.plugin.run(self.args)
            mock_run_test.assert_not_called()
            runner.stop.assert_called_once()
            mock_exit.assert_called_once_with(1)


class TestResolveTestFunctionNames(unittest.TestCase):
    def setUp(self):
        self.plugin = RunPlugin()

    def test_legacy_preflight_aliases(self):
        self.assertEqual(resolve_test_function_name("test_node_smoke"), "test_node_smoke_tier1")
        self.assertEqual(resolve_test_function_name("test_tier3_info"), "test_node_smoke_tier3")
        self.assertEqual(resolve_test_function_name("test_node_smoke_tier1"), "test_node_smoke_tier1")

    def test_dedupes_alias_and_canonical(self):
        names = resolve_test_function_names(
            ["test_node_smoke", "test_node_smoke_tier1", "test_tier3_info", "test_node_smoke_tier3"]
        )
        self.assertEqual(names, ["test_node_smoke_tier1", "test_node_smoke_tier3"])

    @patch("cvs.cli_plugins.run_plugin.pytest.main")
    @patch("cvs.cli_plugins.run_plugin.sys.exit")
    def test_run_test_maps_legacy_preflight_names(self, mock_exit, mock_pytest_main):
        mock_pytest_main.return_value = 0
        self.plugin.run_test(
            "/mock/preflight_checks.py",
            ["test_node_smoke", "test_tier3_info"],
            "/path/to/cluster.json",
            "/path/to/config.json",
            None,
            False,
            None,
            None,
            None,
            [],
        )
        mock_pytest_main.assert_called_once()
        pytest_args = mock_pytest_main.call_args[0][0]
        self.assertIn("/mock/preflight_checks.py::test_node_smoke_tier1", pytest_args)
        self.assertIn("/mock/preflight_checks.py::test_node_smoke_tier3", pytest_args)


if __name__ == "__main__":
    unittest.main()
