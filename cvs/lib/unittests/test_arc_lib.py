'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Tests for ARC controller execution, configuration, and generated suites.
'''

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cvs.lib import arc_lib
from cvs.lib.report.profile import load_json_profile


CLUSTER = {"username": "runner", "node_dict": {"rack-a": {}, "node-only": {}}}
CONFIG = {
    "arc": {
        "arc_release_url": "https://example.invalid/arc-tool-heliosr-1.8.1.tar.gz",
        "arc_version": "1.8.1",
        "controller_nodes": ["rack-a"],
        "rack_config": "/racks/rack.json",
        "spec_mode": "partial",
    }
}


def config(**changes):
    cfg = {"arc": dict(CONFIG["arc"])}
    cfg["arc"].update(changes)
    return cfg


class RecordingOrch:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def exec(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return self.outputs.pop(0)


class TestResolveControllers(unittest.TestCase):
    def test_defaults_list_string_and_empty_cluster(self):
        self.assertEqual(arc_lib.resolve_controllers(CLUSTER, {"arc": {}}), ["rack-a"])
        self.assertEqual(
            arc_lib.resolve_controllers(CLUSTER, config(controller_nodes=["node-only", "rack-a"])),
            ["node-only", "rack-a"],
        )
        self.assertEqual(arc_lib.resolve_controllers(CLUSTER, config(controller_nodes="node-only")), ["node-only"])
        self.assertEqual(arc_lib.resolve_controllers({"node_dict": {}}, {"arc": {}}), [])


class TestValidateArcConfig(unittest.TestCase):
    def test_clean_config(self):
        self.assertEqual(arc_lib.validate_arc_config(config(), CLUSTER), [])
        for changes in (dict(run_timeout=0), dict(run_timeout="7200"), dict(inactivity_timeout=120)):
            with self.subTest(changes=changes):
                self.assertEqual(arc_lib.validate_arc_config(config(**changes), CLUSTER), [])

    def test_invalid_cases(self):
        cases = (
            (dict(arc_release_url=""), CLUSTER),
            (dict(arc_release_url="https://x/arc-1.8.1-deb-linux.tar.gz"), CLUSTER),
            (dict(arc_release_url="https://x/arc-1.8.1.bin"), CLUSTER),
            (dict(arc_version="1.9.0"), CLUSTER),
            (dict(arc_version="1.8.1-RC.2"), CLUSTER),
            (dict(controller_nodes=["missing"]), CLUSTER),
            (dict(rack_config=""), CLUSTER),
            (dict(rack_config="/rack/$bad"), CLUSTER),
            (dict(spec_mode="bad"), CLUSTER),
            (dict(ARC_INSTALL_PATH="//"), CLUSTER),
            (dict(run_timeout=-1), CLUSTER),
            (dict(run_timeout="abc"), CLUSTER),
            (dict(run_timeout=True), CLUSTER),
            (dict(inactivity_timeout=60), CLUSTER),
            (dict(inactivity_timeout="x"), CLUSTER),
            ({}, {"username": "a b", "node_dict": {"rack-a": {}}}),
        )
        for changes, cluster in cases:
            with self.subTest(changes=changes):
                self.assertTrue(arc_lib.validate_arc_config(config(**changes), cluster))

    def test_unsafe_install_path_does_not_hide_url_error(self):
        problems = arc_lib.validate_arc_config(
            config(arc_release_url="https://x/arc-1.8.1.bin", ARC_INSTALL_PATH="/tmp/bad$path"), CLUSTER
        )
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("direct .deb/.rpm/.tar.gz" in problem or "flavour" in problem for problem in problems))
        self.assertEqual(sum("ARC_INSTALL_PATH" in problem for problem in problems), 1)


class TestResolveArcPaths(unittest.TestCase):
    def test_relocation_default_packages_and_home(self):
        cfg = config(ARC_INSTALL_PATH="/tmp/arc-prefix")
        self.assertEqual(arc_lib.resolve_arc_paths(cfg, "tar").arc_bin, "/tmp/arc-prefix/arc/arc.py")
        self.assertEqual(arc_lib.resolve_arc_paths(config(ARC_INSTALL_PATH=""), "tar").prefix, "/opt/amdtools")
        for pkg in ("deb", "rpm"):
            self.assertEqual(arc_lib.resolve_arc_paths(cfg, pkg).arc_bin, "/opt/amdtools/arc/arc.py")
        with patch("os.path.expanduser", return_value="/home/runner/arc"):
            self.assertEqual(
                arc_lib.resolve_arc_paths(config(ARC_INSTALL_PATH="~/arc"), "tar").prefix, "/home/runner/arc"
            )


class TestParseArcVersion(unittest.TestCase):
    def test_labelled_versions_only(self):
        for output, expected in (
            ("arc 1.8.1", "1.8.1"),
            ("arc.py 1.8.1", "1.8.1"),
            ("ARC 1.8.1-rc.2", "1.8.1-rc.2"),
            ("/x/arc-1.9.0/arc.py: No such file", None),
            ("arc 1.8.1-RC.2", None),
        ):
            self.assertEqual(arc_lib.parse_arc_version(output), expected)


class TestInstallArc(unittest.TestCase):
    def _run(self, pkg, outputs, **changes):
        orch = RecordingOrch(outputs)
        with (
            patch.object(arc_lib, "print_test_output"),
            patch.object(arc_lib, "update_test_result") as update,
            patch.object(arc_lib, "fail_test") as fail,
        ):
            arc_lib.install_arc(
                orch, CLUSTER, config(arc_release_url=f"https://x/arc-tool-heliosr-1.8.1.{pkg}", **changes)
            )
        return orch, update, fail

    def test_tar_installs_only_on_controller(self):
        orch, update, fail = self._run(
            "tar.gz",
            [{"rack-a": "missing"}, {"rack-a": "ARC_INSTALL_SUCCESS"}, {"rack-a": "arc 1.8.1"}],
            ARC_INSTALL_PATH="/tmp/arc",
        )
        command = orch.calls[1][0]
        self.assertIn("ARC download progress", command)
        self.assertIn("tar -xzf arc.tar.gz -C '/tmp/arc'", command)
        self.assertIn("--version", command)
        self.assertTrue(all(kwargs["hosts"] == ["rack-a"] for _, kwargs in orch.calls))
        fail.assert_not_called()
        update.assert_called_once()

    def test_deb_and_rpm_commands(self):
        for pkg, fragment in (("deb", "dpkg -i --force-depends ./arc.deb"), ("rpm", "dnf install -y ./arc.rpm")):
            orch, _, _ = self._run(
                pkg, [{"rack-a": "missing"}, {"rack-a": "ARC_INSTALL_SUCCESS"}, {"rack-a": "arc 1.8.1"}]
            )
            self.assertIn(fragment, orch.calls[1][0])

    def test_precheck_skip_and_postcheck_failure(self):
        orch, update, _ = self._run("tar.gz", [{"rack-a": "arc 1.8.1"}])
        self.assertEqual(len(orch.calls), 1)
        update.assert_called_once()
        _, _, fail = self._run(
            "tar.gz", [{"rack-a": "missing"}, {"rack-a": "ARC_INSTALL_SUCCESS"}, {"rack-a": "arc 1.7.0"}]
        )
        self.assertIn("rack-a", fail.call_args.args[0])

    def test_unreachable_and_legacy(self):
        with patch.object(arc_lib.amdtools_lib, "fail_test") as shared_fail:
            self._run("tar.gz", [{}, {}, {}])
        self.assertTrue(any("rack-a" in call.args[0] for call in shared_fail.call_args_list))
        orch = RecordingOrch([])
        with patch.object(arc_lib, "fail_test") as fail, patch.object(arc_lib, "update_test_result"):
            arc_lib.install_arc(orch, CLUSTER, config(arc_release_url="https://x/arc-1.8.1-deb-linux.tar.gz"))
        self.assertEqual(orch.calls, [])
        self.assertIn("direct", fail.call_args.args[0])


class TestBuildArcInvocation(unittest.TestCase):
    def test_flags_and_name_validation(self):
        command = arc_lib.build_arc_invocation("/arc/arc.py", ["PingTest"], "/rack.json", "partial", True, True)
        self.assertIn("--run --rack '/rack.json' --spec-mode partial --no-progress", command)
        self.assertTrue(command.startswith("MOCK=1 "))
        self.assertIn("--continue-on-failure", command)
        self.assertNotIn("--mock", command)
        other = arc_lib.build_arc_invocation("/arc/arc.py", ["PingTest"], "/rack.json", "full", False, False)
        self.assertFalse(other.startswith("MOCK=1 "))
        self.assertNotIn("--continue-on-failure", other)
        with self.assertRaises(ValueError):
            arc_lib.build_arc_invocation("/arc", ["bad;name"], "/rack", "full", False, False)


class TestBuildArcRunCommand(unittest.TestCase):
    def build_command(self, run_timeout=7200, stdout_file=None):
        return arc_lib.build_arc_run_command(
            "python3 '/a/arc.py' PingTest --run", "$HOME/arc_logs/t/1", "test_PingTest", run_timeout, stdout_file
        )

    def test_print_all_streams_with_unconditional_heartbeat(self):
        command = self.build_command()
        self.assertIn("setsid env PYTHONUNBUFFERED=1 python3 '/a/arc.py'", command)
        self.assertIn('-o "$base" & arc_pid=$!', command)
        self.assertIn('while kill -0 "$arc_pid"', command)
        self.assertIn('ARC test_PingTest running: $((now - start))s elapsed"', command)
        self.assertIn('next=$((start + 60))', command)
        for fragment in ('> "', 'grep -E', 'bytes of output', '-gt "$last"'):
            self.assertNotIn(fragment, command)
        self.assertLess(command.index('wait "$arc_pid"; rc=$?'), command.index('CVS_ARC_EXIT_CODE=$rc'))
        self.assertLess(command.index('arc_pid=$!'), command.index("trap '' PIPE"))
        self.assertLess(command.index('trap - PIPE'), command.index('wait "$arc_pid"'))

    def test_quiet_redirects_and_heartbeats_unconditionally(self):
        command = self.build_command(stdout_file="/tmp/runner/arc_run_$$.out")
        self.assertIn('> "/tmp/runner/arc_run_$$.out" 2>&1', command)
        self.assertIn('mkdir -p "$base" "/tmp/runner"', command)
        self.assertIn('bytes of output', command)
        self.assertIn('grep -E', command)
        self.assertIn('rm -f', command)
        self.assertNotIn('-gt "$last"', command)

    def test_watchdog_kills_process_group_and_marks_timeout(self):
        command = self.build_command()
        for fragment in (
            '-ge 7200',
            'kill -s TERM -- "-$arc_pid"',
            'kill -s KILL -- "-$arc_pid"',
            '-lt 30',
            'timed out after 7200s (arc.run_timeout)',
            'CVS_ARC_TIMED_OUT=7200',
            'rc=124',
            '|| { cvs_arc_stop; break; }',
        ):
            self.assertIn(fragment, command)

    def test_zero_run_timeout_disables_watchdog(self):
        command = self.build_command(run_timeout=0)
        self.assertNotIn('CVS_ARC_TIMED_OUT', command)
        self.assertNotIn('-ge 0', command)
        self.assertIn('cvs_arc_stop', command)
        self.assertIn('ARC test_PingTest running:', command)

    def test_mock_prefix_composes_with_env(self):
        invocation = arc_lib.build_arc_invocation("/a/arc.py", ["PingTest"], "/rack.json", "full", True, False)
        command = arc_lib.build_arc_run_command(invocation, "$HOME/arc_logs/t/1", "test_PingTest", 7200)
        self.assertIn('setsid env PYTHONUNBUFFERED=1 MOCK=1 python3', command)


class TestRunArcSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.layout = patch("cvs.core.run_layout.RunLayout.get", return_value=SimpleNamespace(run_dir=self.temp.name))
        self.layout.start()
        self.addCleanup(self.layout.stop)

    def _run(self, output, summary=None, **changes):
        orch = RecordingOrch([{"rack-a": output}])

        def pull(_orch, _host, _user, _remote, dest, _tool):
            root = os.path.join(dest, "session", "ARC")
            os.makedirs(root, exist_ok=True)
            if summary is not None:
                with open(os.path.join(root, "summary.json"), "w") as fh:
                    json.dump(summary, fh)
            return os.path.dirname(root), None

        with (
            patch.object(arc_lib, "pull_remote_dir", side_effect=pull) as pull_mock,
            patch.object(arc_lib, "print_test_output"),
            patch.object(arc_lib, "update_test_result") as update,
            patch.object(arc_lib, "fail_test") as fail,
        ):
            arc_lib.run_arc_selection(orch, CLUSTER, config(**changes), ["example_pass"], "test_example_pass")
        self.pull_calls = pull_mock.call_count
        return orch, update, fail

    @staticmethod
    def summary(status="SUCCESS", numeric=0, failed=0):
        return {
            "overall_summary": {
                "status_code": status,
                "status_code_numeric": numeric,
                "total_selected_items": 6,
                "total_passed": 3,
                "total_completed": 1,
                "total_failed": failed,
                "total_skipped": 0,
                "total_not_run": 1 if failed else 0,
            }
        }

    def test_pass_success_and_warning(self):
        for status in ("SUCCESS", "WARNING"):
            _, update, fail = self._run("CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=0", self.summary(status))
            fail.assert_not_called()
            update.assert_called_once()

    def test_failure_statuses_and_example_failure(self):
        cases = (
            (8, self.summary("HW_CONN", 8), ("HW_CONN", "exit code 8")),
            (0, self.summary("HW_CONN", 8), ("HW_CONN", "exit code 0")),
            (6, self.summary("SUCCESS"), ("disagrees", "exit code 6")),
        )
        for rc, summary, phrases in cases:
            _, _, fail = self._run(f"CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE={rc}", summary)
            self.assertTrue(fail.called)
            for phrase in phrases:
                self.assertIn(phrase, fail.call_args.args[0])
        example = self.summary("NOT_RUN", 13, failed=1)
        example["results"] = {
            "Mock_FailingTest": {
                "name": "Mock_FailingTest",
                "status": "FAILED",
                "rc": "AFID 15038",
                "summary": "ERROR",
            },
            "Mock_PassingTest": {"name": "Mock_PassingTest", "status": "NOT_RUN", "summary": "Parent skipped: Group_4"},
        }
        _, _, fail = self._run("CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=1", example)
        for phrase in ("NOT_RUN", "exit code 1", "1 failed"):
            self.assertIn(phrase, fail.call_args.args[0])

    def test_missing_exit_summary_not_found_and_unreachable(self):
        for output, summary, phrase in (
            ("CVS_ARC_LOG_DIR=/logs/session", self.summary(), "no exit code"),
            ("CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=1", None, "without writing summary.json"),
            ("Test flow or test item 'bad' not found", None, "not available on the remote system"),
        ):
            _, _, fail = self._run(output, summary)
            self.assertIn(phrase, fail.call_args.args[0])
            if phrase == "no exit code":
                self.assertIn("inactivity_timeout", fail.call_args.args[0])
        orch = RecordingOrch([{"node-only": "ignored"}])
        with patch.object(arc_lib, "update_test_result"), patch.object(arc_lib, "fail_test") as fail:
            arc_lib.run_arc_selection(orch, CLUSTER, config(), ["PingTest"], "test_PingTest")
        self.assertIn("unreachable", fail.call_args.args[0])

    def test_command_scoping_timeout_and_quiet_mode(self):
        orch, _, _ = self._run(
            "CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=0",
            self.summary(),
            inactivity_timeout=77,
            print_all_to_console="False",
        )
        cmd, kwargs = orch.calls[0]
        self.assertEqual(kwargs, {"hosts": ["rack-a"], "inactivity_timeout": 77})
        self.assertIn("/tmp/runner/arc_run_$$.out", cmd)
        self.assertIn("ARC test_example_pass running:", cmd)
        self.assertIn("setsid", cmd)
        self.assertIn("CVS_ARC_TIMED_OUT=14400", cmd)
        self.assertIn('echo; logdir=', cmd)
        orch, _, _ = self._run("CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=0", self.summary())
        self.assertIn("ARC test_example_pass running:", orch.calls[0][0])
        self.assertIn("setsid", orch.calls[0][0])
        self.assertIn("CVS_ARC_LOG_DIR", orch.calls[0][0])
        self.assertIn("CVS_ARC_EXIT_CODE", orch.calls[0][0])
        self.assertIn('echo; logdir=', orch.calls[0][0])

    def test_run_timeout_config_plumbing(self):
        output = "CVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=0"
        orch, _, _ = self._run(output, self.summary(), run_timeout="600")
        self.assertIn("CVS_ARC_TIMED_OUT=600", orch.calls[0][0])
        self.assertEqual(orch.calls[0][1]["inactivity_timeout"], 900)
        orch, _, _ = self._run(output, self.summary(), run_timeout=0)
        self.assertNotIn("CVS_ARC_TIMED_OUT", orch.calls[0][0])

    def test_timeout_verdict_reason(self):
        output = (
            "ARC test_example_pass timed out after 600s (arc.run_timeout); stopping ARC process group 42\n"
            "CVS_ARC_TIMED_OUT=600\nCVS_ARC_LOG_DIR=/logs/session\nCVS_ARC_EXIT_CODE=124"
        )
        for summary in (self.summary(), None):
            with self.subTest(summary=summary):
                _, _, fail = self._run(output, summary)
                reason = fail.call_args.args[0]
                self.assertIn("timed out after 600 s", reason)
                self.assertIn("arc.run_timeout", reason)
                self.assertNotIn("disagrees", reason)
                self.assertNotIn("without writing summary.json", reason)
                self.assertEqual(self.pull_calls, 1)


class TestSampleConfigDefaults(unittest.TestCase):
    def test_timeouts_match_library_defaults(self):
        path = Path(arc_lib.__file__).resolve().parents[1] / "input" / "config_file" / "arc" / "arc_config.json"
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)["arc"]
        self.assertEqual(cfg["run_timeout"], arc_lib.DEFAULT_ARC_RUN_TIMEOUT)
        self.assertEqual(cfg["inactivity_timeout"], arc_lib.DEFAULT_ARC_INACTIVITY_TIMEOUT)
        self.assertGreaterEqual(cfg["inactivity_timeout"], arc_lib.MIN_ARC_INACTIVITY_TIMEOUT)


class TestReadinessAndValidation(unittest.TestCase):
    def test_cached_ready_install_and_missing_rack(self):
        previous = (arc_lib._ARC_READY, arc_lib._ARC_INSTALL_PATHS)
        self.addCleanup(lambda: setattr(arc_lib, "_ARC_READY", previous[0]))
        self.addCleanup(lambda: setattr(arc_lib, "_ARC_INSTALL_PATHS", previous[1]))
        arc_lib._ARC_READY = False
        with (
            patch.object(arc_lib, "_arc_installed", return_value=False),
            patch.object(arc_lib, "install_arc") as install,
            patch.object(arc_lib, "resolve_arc_install_location", return_value=Mock()),
            patch.object(arc_lib, "check_rack_config_present", return_value=True),
        ):
            arc_lib.ensure_arc_ready(Mock(), CLUSTER, config())
            arc_lib.ensure_arc_ready(Mock(), CLUSTER, config())
        install.assert_called_once()
        self.assertTrue(arc_lib._ARC_READY)
        arc_lib._ARC_READY = False
        with (
            patch.object(arc_lib, "_arc_installed", return_value=True),
            patch.object(arc_lib, "resolve_arc_install_location", return_value=Mock()),
            patch.object(arc_lib, "check_rack_config_present", return_value=False),
            patch.object(arc_lib, "update_test_result"),
        ):
            arc_lib.ensure_arc_ready(Mock(), CLUSTER, config())
        self.assertFalse(arc_lib._ARC_READY)

    def test_validate_rack_command_and_failure(self):
        orch = RecordingOrch([{"rack-a": "CVS_ARC_EXIT_CODE=3"}])
        with (
            patch.object(arc_lib, "ensure_arc_ready"),
            patch.object(arc_lib, "print_test_output"),
            patch.object(arc_lib, "update_test_result"),
            patch.object(arc_lib, "fail_test") as fail,
        ):
            arc_lib.validate_rack_config(orch, CLUSTER, config())
        cmd, kwargs = orch.calls[0]
        self.assertEqual(kwargs["hosts"], ["rack-a"])
        self.assertIn("--validate --spec-mode partial", cmd)
        self.assertNotIn("--run", cmd)
        self.assertNotIn("--no-progress", cmd)
        self.assertIn('echo; echo "CVS_ARC_EXIT_CODE=$rc"', cmd)
        self.assertIn("exit code 3", fail.call_args.args[0])

    def test_validate_rack_reports_unreachable_controller(self):
        orch = RecordingOrch([{}])
        with (
            patch.object(arc_lib, "ensure_arc_ready"),
            patch.object(arc_lib, "print_test_output"),
            patch.object(arc_lib, "update_test_result"),
            patch.object(arc_lib.amdtools_lib, "fail_test") as fail,
        ):
            arc_lib.validate_rack_config(orch, CLUSTER, config())
        fail.assert_called_once()
        self.assertIn("unreachable", fail.call_args.args[0])


class TestDisruptiveSkipReason(unittest.TestCase):
    def test_gate_and_mock(self):
        self.assertIn("changes hardware state", arc_lib.disruptive_skip_reason(config(), "reboot_all"))
        self.assertIsNone(arc_lib.disruptive_skip_reason(config(allow_disruptive="True"), "reboot_all"))
        self.assertIsNotNone(arc_lib.disruptive_skip_reason(config(mock="True"), "reboot_all"))


class TestContentLists(unittest.TestCase):
    def test_unique_and_disruptive(self):
        names = [name for suite in arc_lib.ARC_SUITES for name in suite["names"]]
        self.assertEqual(len(names), len(set(names)))
        known = {
            "reboot_all",
            "scale_up_config",
            "scale_up_verify",
            "heliosr_mfg_l11",
            "RebootComputeTest",
            "RebootSwitchTest",
            "RebootRmcTest",
            "ComputeDCCycle",
            "PortToggle",
            "ScaleupEAMConfig",
            "ScaleupSwitchConfig",
            "ScaleUpConnectivityTest",
            "PRBSCheck",
            "PRBSCheckEAMFirst",
            "SBRTest",
        }
        disruptive = {name for suite in arc_lib.ARC_SUITES if suite["disruptive"] for name in suite["names"]}
        self.assertLessEqual(known, disruptive)
        self.assertTrue(all(suite["disruptive"] for suite in arc_lib.ARC_SUITES if "disruptive" in suite["key"]))


class TestGeneratedSuitesInSync(unittest.TestCase):
    def test_names_gates_and_profiles(self):
        root = Path(arc_lib.__file__).resolve().parents[1] / "tests" / "arc"
        for suite in arc_lib.ARC_SUITES:
            with self.subTest(suite=suite["key"]):
                path = root / f"arc_test_{suite['key']}.py"
                tree = ast.parse(path.read_text())
                funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                self.assertEqual({node.name for node in funcs}, {"test_" + name for name in suite["names"]})
                for node in funcs:
                    self.assertEqual("disruptive_skip_reason" in ast.unparse(node), suite["disruptive"])
                self.assertEqual(load_json_profile(f"arc_test_{suite['key']}")["sources"]["results"], "arc_res_dict")


class TestArcReporting(unittest.TestCase):
    def test_failure_links_json_and_tarball(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = os.path.join(tmp, "session", "ARC")
            os.makedirs(logs)
            summary = os.path.join(logs, "summary.json")
            errors = os.path.join(logs, "errors.json")
            for path in (summary, errors):
                with open(path, "w") as fh:
                    fh.write("{}")
            result = arc_lib.ControllerResult("failed", tmp, "rack-a", summary, errors, {})
            mgr = Mock(is_enabled=True)
            mgr.add_html_to_report.side_effect = lambda path, **kwargs: os.path.join(
                "report", kwargs.get("dest_name") or os.path.basename(path)
            )
            request = SimpleNamespace(
                config=SimpleNamespace(_html_report_manager=mgr), node=SimpleNamespace(name="test_x")
            )
            with (
                patch.object(arc_lib, "resolve_runner_results_base", return_value=tmp),
                patch.object(arc_lib.amdtools_lib, "stash_report_link") as link,
            ):
                json_hrefs, tar_hrefs = arc_lib._attach_arc_logs_to_html(
                    request, config(), "test_x", [result], "ts", True
                )
            self.assertEqual(json_hrefs["rack-a"], "rack-a_test_x_ts_errors.json")
            self.assertEqual(tar_hrefs["rack-a"], "rack-a_test_x_ts_arc_logs.tar.gz")
            self.assertEqual(link.call_count, 3)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "rack-a_test_x_ts_arc_logs.tar.gz")))

    def test_disruptive_suite_skips_before_ready(self):
        from cvs.tests.arc import arc_test_disruptive

        with (
            patch.object(arc_lib, "ensure_arc_ready") as ready,
            patch.object(arc_lib, "run_arc_items") as run,
            patch.object(arc_test_disruptive.pytest, "skip", side_effect=RuntimeError("skipped")),
        ):
            with self.assertRaisesRegex(RuntimeError, "skipped"):
                arc_test_disruptive.test_RebootComputeTest(Mock(), CLUSTER, config(mock="True"), Mock())
        ready.assert_not_called()
        run.assert_not_called()
