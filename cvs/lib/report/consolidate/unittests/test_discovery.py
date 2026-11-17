'''Tests for finding Run Decks in run folders and zip bundles.'''

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from cvs.lib.report.consolidate.discovery import (
    discover,
    drop_duplicates,
    is_candidate,
    is_own_output,
    is_valid,
    resolve_stem,
)
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, found, status_matrix, write_run
from cvs.lib.report.consolidate.warning_log import UNREADABLE_FILE, WarningLog


class DiscoveryCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.warnings = WarningLog()

    def discover(self):
        return discover([self.root], self.warnings)


class TestDiscover(DiscoveryCase):
    def test_finds_a_deck_in_a_run_folder(self):
        run = write_run(self.root, "rvs_cvs", deck("rvs_cvs"))
        decks, ignored = self.discover()
        self.assertEqual(len(decks), 1)
        self.assertFalse(decks[0].in_bundle)
        self.assertEqual(decks[0].stem, "rvs_cvs")
        self.assertEqual(decks[0].run_root, run)
        self.assertEqual(decks[0].json_rel, "rvs_cvs_html/rvs_cvs_run_deck.json")
        self.assertEqual(ignored, 0)

    def test_finds_a_deck_inside_a_bundle(self):
        write_run(self.root, "rccl_perf", deck("rccl", "record", "series"), folder=False, bundle="2026-11-10T193205")
        decks, _ = self.discover()
        self.assertEqual(len(decks), 1)
        self.assertTrue(decks[0].in_bundle)
        self.assertEqual(decks[0].stem, "rccl_perf")
        self.assertEqual(decks[0].json_rel, "rccl_perf_html/rccl_run_deck.json")
        self.assertEqual(decks[0].relative_path, "local-1/rccl_perf_2026-11-10T193205.zip/" + decks[0].json_rel)

    def test_searches_nested_trees(self):
        write_run(self.root / "health" / "cvs_runs", "rvs_cvs", deck("rvs_cvs"), run_id="local-1")
        write_run(self.root / "network" / "deep" / "cvs_runs", "rccl_perf", deck("rccl"), run_id="12345")
        decks, _ = self.discover()
        self.assertEqual(sorted(found_deck.stem for found_deck in decks), ["rccl_perf", "rvs_cvs"])

    def test_skips_its_own_output_and_temporary_folders(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"))
        previous = self.root / "consolidated_report_2026-11-01T000000.zip"
        with zipfile.ZipFile(previous, "w") as out:
            out.writestr("x/runs/agfhc_cvs/r/agfhc_cvs_html/agfhc_run_deck.json", json.dumps(deck("agfhc_cvs")))
        write_run(self.root / "consolidated_report_2026-11-01T000000" / "runs", "agfhc_cvs", deck("agfhc_cvs"))
        write_run(self.root / ".consolidated_report.tmp-abc", "agfhc_cvs", deck("agfhc_cvs"))
        decks, _ = self.discover()
        self.assertEqual([found_deck.stem for found_deck in decks], ["rvs_cvs"])

    def test_ignores_unrelated_json_and_zips_without_a_deck(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"))
        (self.root / "cluster.json").write_text('{"nodes": {}}', encoding="utf-8")
        (self.root / "notes.json").write_text("not json", encoding="utf-8")
        with zipfile.ZipFile(self.root / "logs.zip", "w") as out:
            out.writestr("a.json", "{}")
        decks, ignored = self.discover()
        self.assertEqual(len(decks), 1)
        self.assertEqual(ignored, 3)
        self.assertEqual(self.warnings.entries, [])

    def test_corrupt_zip_is_skipped_with_a_warning(self):
        (self.root / "rvs_cvs_2026-11-10T193205.zip").write_bytes(b"PK\x03\x04 truncated")
        with self.assertLogs(level="WARNING"):
            decks, _ = self.discover()
        self.assertEqual(decks, [])
        self.assertEqual([entry["code"] for entry in self.warnings.entries], [UNREADABLE_FILE])

    def test_unparsable_json_in_a_deck_folder_is_kept_as_unreadable(self):
        write_run(self.root, "rvs_cvs", '{"schema_version": 1, "suite_')
        decks, _ = self.discover()
        self.assertEqual(len(decks), 1)
        self.assertIsNone(decks[0].data)
        self.assertEqual(decks[0].stem, "rvs_cvs")


class TestDropDuplicates(DiscoveryCase):
    def test_folder_and_bundle_of_one_run_count_once_and_the_folder_is_kept(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"), bundle="2026-11-10T193205")
        decks, _ = self.discover()
        self.assertEqual(len(decks), 2)
        unique, duplicates = drop_duplicates(decks)
        self.assertEqual(len(unique), 1)
        self.assertFalse(unique[0].in_bundle)
        self.assertTrue(duplicates[0].in_bundle)

    def test_different_runs_are_kept(self):
        write_run(self.root, "rvs_cvs", deck("rvs_cvs"), run_id="local-1")
        write_run(self.root, "rvs_cvs", deck("rvs_cvs", generated_at="2026-11-11 08:00 UTC"), run_id="local-2")
        unique, duplicates = drop_duplicates(self.discover()[0])
        self.assertEqual((len(unique), len(duplicates)), (2, 0))


class TestDeckChecks(unittest.TestCase):
    def test_candidate_needs_schema_version_suite_id_and_deck_profile(self):
        self.assertTrue(is_candidate(deck("rvs_cvs")))
        self.assertFalse(is_candidate({"schema_version": 1, "suite_id": "rvs_cvs"}))
        self.assertFalse(is_candidate(["not", "a", "deck"]))

    def test_valid_deck(self):
        self.assertTrue(is_valid(deck("rvs_cvs", dataset=status_matrix({"n1": {"g": "pass"}}))))

    def test_invalid_decks(self):
        self.assertFalse(is_valid(deck("rvs_cvs", schema_version=2)))
        self.assertFalse(is_valid(deck("rvs_cvs", overall_status="")))
        self.assertFalse(is_valid(deck("rvs_cvs", datasets=[])))
        self.assertFalse(is_valid(deck("rvs_cvs", report={})))
        self.assertFalse(is_valid(None))

    def test_own_output_names(self):
        self.assertTrue(is_own_output("consolidated_report_2026-11-10T193205.zip"))
        self.assertTrue(is_own_output(".consolidated_report.tmp-x"))
        self.assertFalse(is_own_output("rvs_cvs_2026-11-10T193205.zip"))


class TestResolveStem(unittest.TestCase):
    def test_suite_stem_field_wins(self):
        data = deck("rccl", suite_stem="rccl_pairwise")
        self.assertEqual(resolve_stem(data, "rccl_perf_html/rccl_run_deck.json"), "rccl_pairwise")

    def test_deck_folder_before_pytest_report_name(self):
        data = deck("rccl", provenance={"pytest_html_path": "/out/custom_name.html"})
        self.assertEqual(resolve_stem(data, "rccl_perf_html/rccl_run_deck.json"), "rccl_perf")

    def test_pytest_report_name_from_provenance(self):
        data = deck("vllm", provenance={"pytest_html_path": "C:\\runs\\vllm_single.html"})
        self.assertEqual(resolve_stem(data, "vllm_run_deck.json"), "vllm_single")

    def test_bundle_name_then_suite_id(self):
        self.assertEqual(
            resolve_stem(deck("atom"), "atom_run_deck.json", Path("/r/atom_2026-11-10T193205.zip")), "atom"
        )
        self.assertEqual(resolve_stem(deck("atom"), "atom_run_deck.json", Path("/r/archive.zip")), "atom")
        self.assertEqual(
            resolve_stem({"suite_id": "x"}, "x.json", Path("/r/sglang_single_2026-11-10T193205.zip")), "sglang_single"
        )

    def test_unreadable_deck_without_a_folder_has_no_stem(self):
        self.assertIsNone(resolve_stem(None, "deck.json"))


class TestFoundDeckPaths(unittest.TestCase):
    def test_display_path_names_the_bundle_member(self):
        root = Path(tempfile.gettempdir())
        bundled = found(root, deck("rccl"), "rccl_perf_html/rccl_run_deck.json", bundle=True)
        self.assertTrue(bundled.display_path.endswith("x_2026-11-10T193205.zip/rccl_perf_html/rccl_run_deck.json"))
        folder = found(root, deck("rccl"), "rccl_perf_html/rccl_run_deck.json")
        self.assertEqual(folder.relative_path, "run/rccl_perf_html/rccl_run_deck.json")


if __name__ == "__main__":
    unittest.main()
