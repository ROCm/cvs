'''End-to-end tests for building a consolidated report, without a cluster.'''

import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.consolidate import report
from cvs.lib.report.consolidate.report import (
    ConsolidationError,
    consolidate,
    resolve_log_dirs,
    variant_id,
)
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, found, status_matrix, write_run
from cvs.lib.report.profiles.hooks.rccl_session import variant_from_config
from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck
from cvs.lib.report.rundeck.unittests.test_rccl_payload import _graph
from cvs.lib.report_plugins import HtmlReportManager


class ReportCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "workspace"
        self.root.mkdir()
        self.out = Path(tmp.name) / "out"

    def consolidate(self, *args, **kwargs):
        with self.assertLogs(level="INFO") as logs:
            result = consolidate(*args, **kwargs)
        self.log_output = "\n".join(logs.output)
        return result

    def read_zip(self, zip_path):
        with zipfile.ZipFile(zip_path) as out:
            names = out.namelist()
            top = names[0].split("/")[0]
            return top, {name[len(top) + 1 :]: out.read(name) for name in names}


class TestConsolidate(ReportCase):
    def write_campaign(self):
        failing = status_matrix({"10.0.0.1_node-01": {"gst": "fail"}, "10.0.0.2_node-02": {"gst": "pass"}})
        write_run(self.root, "rvs_cvs", deck("rvs_cvs", "pass", generated_at="2026-11-09 08:00 UTC"), run_id="local-1")
        write_run(
            self.root,
            "rvs_cvs",
            deck("rvs_cvs", "fail", dataset=failing, generated_at="2026-11-10 08:00 UTC"),
            run_id="local-2",
            bundle="2026-11-10T080000",
        )
        write_run(
            self.root / "network",
            "rccl_perf",
            deck("rccl", "record", "series"),
            folder=False,
            bundle="2026-11-10T090000",
        )
        write_run(self.root, "agfhc_cvs", deck("agfhc_cvs", schema_version=2), run_id="local-3")

    def test_campaign_report_zip(self):
        self.write_campaign()
        result = self.consolidate(str(self.root))
        self.assertEqual(result.zip_path.parent, self.root.resolve())
        self.assertEqual((len(result.found), len(result.duplicates)), (5, 1))
        top, files = self.read_zip(result.zip_path)
        self.assertEqual(top, result.zip_path.stem)
        self.assertIn("index.html", files)

        summary = json.loads(files["summary.json"])
        self.assertEqual(summary["schema_version"], 1)
        self.assertFalse(summary["sample"])
        self.assertEqual(summary["verdict"], "needs_attention")
        suites = {suite["id"]: suite for suite in summary["suites"]}
        self.assertEqual(sorted(suites), ["agfhc_cvs", "rccl_perf", "rvs_cvs"])
        self.assertFalse(any(key.startswith("_") for suite in summary["suites"] for key in suite))

        rvs = suites["rvs_cvs"]
        self.assertEqual(
            (rvs["status"], rvs["category"], rvs["failedNodes"]), ("fail", "Burn-in / Diag", ["10.0.0.1_node-01"])
        )
        self.assertEqual([run["status"] for run in rvs["history"]], ["pass"])
        self.assertIn(rvs["reportUrl"], files)
        self.assertIn(rvs["history"][0]["reportUrl"], files)
        self.assertIn(b"cvs-consolidated-report-link", files[rvs["reportUrl"]])

        self.assertEqual((suites["rccl_perf"]["status"], suites["rccl_perf"]["category"]), ("record", "Network"))
        self.assertFalse(suites["agfhc_cvs"]["available"])
        self.assertEqual([warning["code"] for warning in summary["warnings"]], ["invalid_deck"])

        for suite_id in suites:
            detail = json.loads(files["data/details/%s.json" % suite_id])
            self.assertEqual(detail["schema_version"], 1)
            self.assertEqual(detail["summary"]["id"], suite_id)
        self.assertIn("Collection verdict: needs_attention", self.log_output)

    def test_originals_are_unchanged_and_no_temporary_folder_is_left(self):
        run = write_run(self.root, "rvs_cvs", deck("rvs_cvs"), bundle="2026-11-10T080000")
        original = (run / "rvs_cvs_html" / "rvs_cvs_run_deck.html").read_bytes()
        result = self.consolidate(str(self.root), output_dir=str(self.out))
        self.assertEqual((run / "rvs_cvs_html" / "rvs_cvs_run_deck.html").read_bytes(), original)
        self.assertEqual([path.name for path in self.out.iterdir()], [result.zip_path.name])

    def test_two_suite_variants_sharing_a_suite_id_stay_separate(self):
        write_run(self.root, "rccl_perf", deck("rccl", "record", "series"), run_id="a")
        pairwise = deck("rccl", "record", "series", generated_at="2026-11-10 21:00 UTC")
        write_run(self.root, "rccl_pairwise", pairwise, run_id="b")
        summary = self.consolidate(str(self.root)).summary
        self.assertEqual([suite["name"] for suite in summary["suites"]], ["rccl \u00b7 pairwise", "rccl \u00b7 perf"])
        self.assertEqual(summary["verdict"], "evaluation_incomplete")

    def test_several_directories_go_to_the_current_directory(self):
        write_run(self.root / "health", "rvs_cvs", deck("rvs_cvs"))
        write_run(self.root / "network", "rccl_perf", deck("rccl", "record", "series"))
        self.out.mkdir()
        with patch("pathlib.Path.cwd", return_value=self.out):
            result = self.consolidate("%s,%s" % (self.root / "health", self.root / "network"))
        self.assertEqual(result.zip_path.parent, self.out.resolve())
        self.assertEqual(len(result.summary["suites"]), 2)

    def test_suite_left_out_with_its_directory(self):
        write_run(self.root / "health", "rvs_cvs", deck("rvs_cvs"))
        write_run(self.root / "network", "rccl_perf", deck("rccl", "record", "series"))
        result = self.consolidate(str(self.root / "health"))
        self.assertEqual([suite["id"] for suite in result.summary["suites"]], ["rvs_cvs"])
        self.assertEqual(result.summary["verdict"], "all_passed")

    def test_unknown_suite_is_uncategorized_with_a_warning(self):
        write_run(self.root, "brand_new", deck("brand_new"))
        summary = self.consolidate(str(self.root)).summary
        self.assertEqual(summary["suites"][0]["category"], "Uncategorized")
        self.assertEqual([warning["code"] for warning in summary["warnings"]], ["unknown_suite_id"])

    def test_deck_without_a_timestamp_still_counts(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs", generated_at=""))
        summary = self.consolidate(str(self.root)).summary
        self.assertEqual(summary["suites"][0]["status"], "pass")
        self.assertEqual(summary["generated"], "Unavailable")
        self.assertEqual([warning["code"] for warning in summary["warnings"]], ["missing_field"])


class TestFailures(ReportCase):
    def test_missing_directory(self):
        with self.assertRaisesRegex(ConsolidationError, "Directory not found: .*missing"):
            consolidate(str(self.root / "missing"))

    def test_no_decks(self):
        with self.assertLogs(level="INFO"):
            with self.assertRaisesRegex(ConsolidationError, "No Run Decks found under"):
                consolidate(str(self.root))

    def test_output_that_cannot_be_written(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"))
        with patch.object(report.tempfile, "mkdtemp", side_effect=PermissionError("read-only")):
            with self.assertLogs(level="INFO"):
                with self.assertRaisesRegex(ConsolidationError, "Cannot write to"):
                    consolidate(str(self.root))

    def test_failure_while_writing_leaves_nothing_behind(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"))
        with patch.object(report.writer, "write_zip", side_effect=OSError("No space left on device")):
            with self.assertLogs(level="INFO"):
                with self.assertRaisesRegex(ConsolidationError, "No space left on device"):
                    consolidate(str(self.root), output_dir=str(self.out))
        self.assertEqual(list(self.out.iterdir()), [])


class TestHelpers(ReportCase):
    def test_resolve_log_dirs(self):
        (self.root / "a").mkdir()
        (self.root / "b").mkdir()
        dirs = resolve_log_dirs(" %s , %s,%s, " % (self.root / "a", self.root / "b", self.root / "a"))
        self.assertEqual(dirs, [(self.root / "a").resolve(), (self.root / "b").resolve()])
        self.assertEqual(resolve_log_dirs([self.root / "a"]), [(self.root / "a").resolve()])
        with self.assertRaisesRegex(ConsolidationError, "No directory given"):
            resolve_log_dirs(" , ")

    def test_variant_id_is_safe_for_file_names(self):
        self.assertEqual(variant_id(found(self.root, deck("rvs_cvs", suite_stem="../evil name"))), "evil_name")
        self.assertEqual(variant_id(found(self.root, None, "deck.json")), "unknown")


class TestPublishedDecks(ReportCase):
    '''Decks written by the Run Deck publisher, read from their run folders and their bundles.'''

    def setUp(self):
        super().setUp()
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.addCleanup(
            setattr, HtmlReportManager, "_current_instance", getattr(HtmlReportManager, "_current_instance", None)
        )
        patches.enter_context(
            patch("cvs.lib.report.rundeck.generate_rundeck.build_inference_report_provenance", return_value={})
        )
        patches.enter_context(patch("cvs.lib.report.rundeck.generate_rundeck.cvs_version", return_value="test"))
        patches.enter_context(patch("cvs.lib.report_plugins.sys.argv", ["cvs"]))

    def publish(self, run_dir, suite, store):
        run_dir.mkdir(parents=True)
        html_path = run_dir / ("%s.html" % suite)
        html_path.write_text('<html><body><table id="environment"></table></body></html>', encoding="utf-8")
        config = SimpleNamespace(
            _suite_name=suite,
            _test_html_dir="%s_html" % suite,
            _suite_report_config=None,
            option=SimpleNamespace(htmlpath=str(html_path), log_file=None, self_contained_html=True),
        )
        self.assertTrue(try_auto_register_suite_report(config))
        session, manager = SimpleNamespace(config=config), HtmlReportManager(config)
        with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
            self.assertIsNotNone(generate_rundeck(session, manager))
        manager.create_zip_bundle(session)

    def test_published_decks_are_consolidated_once_each(self):
        nodes = {
            "10.0.0.1_node-01": {"status": "pass"},
            "10.0.0.2_node-02": {"status": "fail", "items": [{"name": "fp8", "status": "fail", "message": "fp8 low"}]},
        }
        health = {"cvs_results_dict": {"_meta": {"cluster": "helios"}, "groups": {"gst": {"nodes": nodes}}}}
        rccl = {"cvs_results_dict": _graph(), "variant_config": variant_from_config({}, {}, "rccl_perf")}
        self.publish(self.root / "run-rvs", "rvs_cvs", health)
        self.publish(self.root / "run-rccl", "rccl_perf", rccl)

        result = self.consolidate(str(self.root))
        self.assertEqual((len(result.found), len(result.duplicates)), (4, 2))
        suites = {suite["id"]: suite for suite in result.summary["suites"]}
        self.assertEqual(sorted(suites), ["rccl_perf", "rvs_cvs"])
        rvs, rccl_suite = suites["rvs_cvs"], suites["rccl_perf"]
        self.assertEqual((rvs["name"], rvs["status"], rvs["cluster"]), ("RVS", "fail", "helios"))
        self.assertEqual(rvs["failedNodes"], ["10.0.0.2_node-02"])
        self.assertEqual(
            (rccl_suite["status"], rccl_suite["unit"], rccl_suite["total"]), ("record", "recorded result rows", 3)
        )
        self.assertEqual(result.summary["verdict"], "needs_attention")
        self.assertEqual(
            result.summary["manifest"]["hosts"][1], {"host": "node-02", "suites": ["RVS"], "failed_in": ["RVS"]}
        )

        _, files = self.read_zip(result.zip_path)
        for suite in suites.values():
            for key in ("reportUrl", "viewerUrl", "jsonUrl", "pytestUrl"):
                with self.subTest(suite=suite["id"], link=key):
                    self.assertIn(suite[key], files)
        detail = json.loads(files["data/details/rvs_cvs.json"])
        self.assertEqual(detail["evidence"][0]["message"], "fp8 low")
        for key in ("reportUrl", "viewerUrl"):
            with self.subTest(page=key):
                self.assertIn(b'<a id="cvs-consolidated-report-link" class="ail-back"', files[rvs[key]])


if __name__ == "__main__":
    unittest.main()
