'''Artifact publishing tests without a live cluster.'''

import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.rccl_session import variant_from_config
from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck, pytest_session_failed
from cvs.lib.report.rundeck.unittests.test_rccl_payload import _graph
from cvs.lib.report.testing.fixtures import generic_variant, two_cell_inf_res
from cvs.lib.report_plugins import HtmlReportManager


class TestGenerateRundeck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.addCleanup(
            setattr, HtmlReportManager, "_current_instance", getattr(HtmlReportManager, "_current_instance", None)
        )
        self.patches.enter_context(
            patch("cvs.lib.report.rundeck.generate_rundeck.build_inference_report_provenance", return_value={})
        )
        self.patches.enter_context(patch("cvs.lib.report.rundeck.generate_rundeck.cvs_version", return_value="test"))
        self.patches.enter_context(patch("cvs.lib.report_plugins.sys.argv", ["cvs"]))

    def make_session(self, suite, testsfailed=0, exitstatus=pytest.ExitCode.OK):
        html_path = self.root / f"{suite}.html"
        html_path.write_text('<html><body><table id="environment"></table></body></html>', encoding="utf-8")
        config = SimpleNamespace(
            _suite_name=suite,
            _test_html_dir=f"{suite}_html",
            _suite_report_config=None,
            option=SimpleNamespace(htmlpath=str(html_path), log_file=None, self_contained_html=True),
        )
        self.assertTrue(try_auto_register_suite_report(config))
        return SimpleNamespace(config=config, testsfailed=testsfailed, exitstatus=exitstatus), HtmlReportManager(config)

    def test_pytest_session_failed(self):
        cases = (
            (SimpleNamespace(), False),
            (SimpleNamespace(testsfailed=0, exitstatus=pytest.ExitCode.OK), False),
            (SimpleNamespace(testsfailed=0, exitstatus=0), False),
            (SimpleNamespace(testsfailed=0, exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED), False),
            (SimpleNamespace(testsfailed=1, exitstatus=pytest.ExitCode.TESTS_FAILED), True),
            (SimpleNamespace(testsfailed=0, exitstatus=pytest.ExitCode.INTERRUPTED), True),
            (SimpleNamespace(testsfailed=0, exitstatus=pytest.ExitCode.INTERNAL_ERROR), True),
        )
        for session, expected in cases:
            with self.subTest(session=session):
                self.assertEqual(pytest_session_failed(session), expected)

    def test_all_rccl_stems_publish_linked_html_json_and_viewer_in_zip(self):
        for suite in ("rccl_perf", "rccl_regression", "rccl_pairwise"):
            with self.subTest(suite=suite):
                session, manager = self.make_session(suite)
                store = {"cvs_results_dict": _graph(), "variant_config": variant_from_config({}, {}, suite)}
                with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                    artifacts = generate_rundeck(session, manager)
                self.assertNotIn("summary", artifacts)
                self.assertEqual(artifacts["viewer"], manager.log_dir / "rccl_run_deck_viewer.html")
                viewer = artifacts["viewer"].read_text(encoding="utf-8")
                self.assertIn('id="embedded-report-json"', viewer)
                self.assertIn('data-viewer="series"', viewer)
                self.assertIn('<a class="to-top" href="#top">Go to top</a>', viewer)
                self.assertIn("function pageLongTables", viewer)
                self.assertIn("Bus bandwidth vs message size", viewer)
                self.assertIn("all_reduce_perf", viewer)
                self.assertIn("all_gather_perf", viewer)
                self.assertNotIn("viewer_deck_html", viewer)
                self.assertNotIn("metricLabels", viewer)
                self.assertEqual(artifacts["html"], manager.log_dir / "rccl_run_deck.html")
                payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
                self.assertEqual(payload["overall_status"], "record")
                self.assertEqual(len(payload["results_table"]["rows"]), 3)
                manager.create_zip_bundle(session)
                document = manager.htmlpath.read_text(encoding="utf-8")
                self.assertIn(f'{suite}_html/rccl_run_deck.html', document)
                self.assertNotIn(f'{suite}_html/rccl_run_deck.json', document)
                with zipfile.ZipFile(next(self.root.glob(f"{suite}_*.zip"))) as bundle:
                    self.assertIn(f"{suite}_html/rccl_run_deck.html", bundle.namelist())
                    self.assertIn(f"{suite}_html/rccl_run_deck.json", bundle.namelist())
                    self.assertIn(f"{suite}_html/rccl_run_deck_viewer.html", bundle.namelist())

    def test_rccl_deck_status_matches_pytest_outcome(self):
        passed = {"check": "bus_bw", "status": "pass"}
        dip_passed = {"check": "bw_dip", "status": "pass"}
        failed = {"check": "bus_bw", "status": "fail", "message": "low bandwidth"}
        cases = (
            ("rccl_perf", [passed], 1, pytest.ExitCode.TESTS_FAILED, "fail"),
            ("rccl_perf", [passed, dip_passed], 0, pytest.ExitCode.OK, "pass"),
            ("rccl_perf", [], 0, pytest.ExitCode.OK, "record"),
            ("rccl_perf", [], 1, pytest.ExitCode.TESTS_FAILED, "fail"),
            ("rccl_perf", [failed], 1, pytest.ExitCode.TESTS_FAILED, "fail"),
            ("rccl_perf", [passed], 0, pytest.ExitCode.INTERRUPTED, "fail"),
            ("rccl_regression", [passed], 0, pytest.ExitCode.OK, "pass"),
            ("rccl_pairwise", None, 1, pytest.ExitCode.TESTS_FAILED, "fail"),
            ("rccl_pairwise", None, 0, pytest.ExitCode.OK, "record"),
        )
        for suite, verdicts, testsfailed, exitstatus, expected in cases:
            with self.subTest(suite=suite, verdicts=verdicts, testsfailed=testsfailed, exitstatus=exitstatus):
                session, manager = self.make_session(suite, testsfailed=testsfailed, exitstatus=exitstatus)
                store = {
                    "cvs_results_dict": _graph(),
                    "variant_config": variant_from_config({}, {}, suite, verdicts=verdicts),
                }
                with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                    artifacts = generate_rundeck(session, manager)
                payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
                self.assertEqual(payload["overall_status"], expected)
                self.assertNotIn("pytest_failed", store)

    def test_inference_profiles_keep_model_titles_viewers_and_ci_summaries(self):
        for suite in ("vllm", "sglang", "atom"):
            with self.subTest(suite=suite):
                session, manager = self.make_session(suite)
                variant = generic_variant()
                store = {"cvs_results_dict": two_cell_inf_res(), "variant_config": variant}
                with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                    artifacts = generate_rundeck(session, manager)
                self.assertTrue(artifacts["summary"].is_file())
                self.assertTrue(artifacts["viewer"].is_file())
                viewer = artifacts["viewer"].read_text(encoding="utf-8")
                self.assertIn('id="comparison-grid"', viewer)
                self.assertNotIn("renderRunCard", viewer)
                title = load_json_profile(suite)["title"]
                self.assertIn(
                    f"<title>{title} &mdash; org/example-model</title>", artifacts["html"].read_text(encoding="utf-8")
                )
                reports = manager.generate_reports_section()
                self.assertIn(f"{suite}_run_deck.html", reports)
                for suffix in (".json", "_summary.html", "_viewer.html"):
                    self.assertNotIn(f"{suite}_run_deck{suffix}", reports)

    def test_no_html_or_no_results_skips_publication(self):
        session, manager = self.make_session("rccl_perf")
        with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value={}):
            self.assertIsNone(generate_rundeck(session, manager))
        session.config.option.htmlpath = None
        with patch(
            "cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value={"cvs_results_dict": _graph()}
        ):
            self.assertIsNone(generate_rundeck(session, manager))
        self.assertFalse(manager.log_dir.exists())

    def test_renderer_failure_does_not_interrupt_session_finish(self):
        session, manager = self.make_session("rccl_perf")
        store = {"cvs_results_dict": _graph(), "variant_config": variant_from_config({}, {})}
        with (
            patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store),
            patch("cvs.lib.report.rundeck.generate_rundeck.render_rundeck_html", side_effect=ValueError("bad chart")),
            patch("cvs.lib.report.rundeck.generate_rundeck.log.error") as error,
        ):
            self.assertIsNone(generate_rundeck(session, manager))
        error.assert_called_once()
        manager.create_zip_bundle(session)
        self.assertEqual(len(list(self.root.glob("rccl_perf_*.zip"))), 1)


if __name__ == "__main__":
    unittest.main()
