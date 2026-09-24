'''Artifact publishing tests without a live cluster.'''

import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.rccl_session import variant_from_config
from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck
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

    def make_session(self, suite):
        html_path = self.root / f"{suite}.html"
        html_path.write_text('<html><body><table id="environment"></table></body></html>', encoding="utf-8")
        config = SimpleNamespace(
            _suite_name=suite,
            _test_html_dir=f"{suite}_html",
            _suite_report_config=None,
            option=SimpleNamespace(htmlpath=str(html_path), log_file=None, self_contained_html=True),
        )
        self.assertTrue(try_auto_register_suite_report(config))
        return SimpleNamespace(config=config), HtmlReportManager(config)

    def test_all_rccl_stems_publish_linked_html_and_json_in_zip(self):
        for suite in ("rccl_perf", "rccl_regression", "rccl_pairwise"):
            with self.subTest(suite=suite):
                session, manager = self.make_session(suite)
                store = {"cvs_results_dict": _graph(), "variant_config": variant_from_config({}, {}, suite)}
                with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                    artifacts = generate_rundeck(session, manager)
                self.assertIsNone(artifacts["summary"])
                self.assertNotIn("viewer", artifacts)
                self.assertEqual(artifacts["html"], manager.log_dir / "rccl_run_deck.html")
                payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
                self.assertEqual(len(payload["results_table"]["rows"]), 3)
                manager.create_zip_bundle(session)
                document = manager.htmlpath.read_text(encoding="utf-8")
                for suffix in ("html", "json"):
                    self.assertIn(f'{suite}_html/rccl_run_deck.{suffix}', document)
                with zipfile.ZipFile(next(self.root.glob(f"{suite}_*.zip"))) as bundle:
                    self.assertIn(f"{suite}_html/rccl_run_deck.html", bundle.namelist())
                    self.assertIn(f"{suite}_html/rccl_run_deck.json", bundle.namelist())

    def test_inference_profiles_keep_model_titles_viewers_and_ci_summaries(self):
        for suite in ("vllm", "sglang", "atom"):
            with self.subTest(suite=suite):
                session, manager = self.make_session(suite)
                variant = generic_variant()
                store = {"inf_res_dict": two_cell_inf_res(), "variant_config": variant}
                with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                    artifacts = generate_rundeck(session, manager)
                self.assertTrue(artifacts["summary"].is_file())
                self.assertTrue(artifacts["viewer"].is_file())
                title = load_json_profile(suite)["title"]
                self.assertIn(
                    f"<title>{title} &mdash; org/example-model</title>", artifacts["html"].read_text(encoding="utf-8")
                )
                self.assertIn(f"{suite}_run_deck_summary.html", manager.generate_reports_section())

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
            patch("cvs.lib.report.rundeck.generate_rundeck.log.warning") as warning,
        ):
            self.assertIsNone(generate_rundeck(session, manager))
        warning.assert_called_once()
        manager.create_zip_bundle(session)
        self.assertEqual(len(list(self.root.glob("rccl_perf_*.zip"))), 1)


if __name__ == "__main__":
    unittest.main()
