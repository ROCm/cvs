'''Tests for assembling the consolidated report folder and writing its zip.'''

import json
import os
import re
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from cvs.lib.report.consolidate import writer
from cvs.lib.report.consolidate.discovery import discover
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, write_run
from cvs.lib.report.consolidate.warning_log import WarningLog
from cvs.lib.report.rundeck.runtime.theme import OVERVIEW_LINK_MARKER, THEME_BOOT_SCRIPT, tokens_css


class WriterCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.inputs = self.tmp / "inputs"
        self.report = self.tmp / "report"
        self.report.mkdir()
        self.warnings = WarningLog()

    def only_deck(self):
        decks, _ = discover([self.inputs], self.warnings)
        self.assertEqual(len(decks), 1)
        return decks[0]


class TestPlaceRun(WriterCase):
    def test_folder_run_copies_deck_folder_pytest_report_and_log(self):
        write_run(self.inputs, "rvs_cvs", deck("rvs_cvs"), run_id="local-7")
        links = writer.place_run(self.only_deck(), "rvs_cvs", self.report, self.warnings)
        self.assertEqual(
            links,
            {
                "jsonUrl": "runs/rvs_cvs/local-7/rvs_cvs_html/rvs_cvs_run_deck.json",
                "reportUrl": "runs/rvs_cvs/local-7/rvs_cvs_html/rvs_cvs_run_deck.html",
                "viewerUrl": None,
                "pytestUrl": "runs/rvs_cvs/local-7/rvs_cvs.html",
                "logUrl": "runs/rvs_cvs/local-7/rvs_cvs.log",
            },
        )

    def test_bundle_run_is_extracted(self):
        data = deck("rccl", "record", "series", summary={"viewer_html": "rccl_run_deck_viewer.html"})
        run = write_run(self.inputs, "rccl_perf", data, folder=False, bundle="2026-11-10T193205")
        with zipfile.ZipFile(next(run.glob("*.zip")), "a") as bundle:
            bundle.writestr("rccl_perf_html/rccl_run_deck_viewer.html", "<html><body>viewer</body></html>")
        links = writer.place_run(self.only_deck(), "rccl_perf", self.report, self.warnings)
        base = "runs/rccl_perf/rccl_perf_2026-11-10T193205/"
        self.assertEqual(links["reportUrl"], base + "rccl_perf_html/rccl_run_deck.html")
        self.assertEqual(links["viewerUrl"], base + "rccl_perf_html/rccl_run_deck_viewer.html")
        self.assertEqual(links["pytestUrl"], base + "rccl_perf.html")
        self.assertEqual(links["logUrl"], base + "rccl_perf_html/rccl_perf.log")

    def test_runs_with_the_same_folder_name_do_not_collide(self):
        write_run(self.inputs, "rvs_cvs", deck("rvs_cvs"), run_id="local-1")
        found = self.only_deck()
        first = writer.place_run(found, "rvs_cvs", self.report, self.warnings)
        second = writer.place_run(found, "rvs_cvs", self.report, self.warnings)
        self.assertIn("/local-1/", first["jsonUrl"])
        self.assertIn("/local-1-1/", second["jsonUrl"])

    def test_deck_outside_a_deck_folder_copies_only_its_own_files(self):
        folder = self.inputs / "loose"
        folder.mkdir(parents=True)
        (folder / "atom_run_deck.json").write_text(json.dumps(deck("atom", builder="sweep")), encoding="utf-8")
        (folder / "atom_run_deck.html").write_text("<html><body></body></html>", encoding="utf-8")
        (folder / "unrelated_big_file.bin").write_bytes(b"x" * 10)
        links = writer.place_run(self.only_deck(), "atom", self.report, self.warnings)
        copied = sorted(path.name for path in (self.report / "runs" / "atom" / "loose").iterdir())
        self.assertEqual(copied, ["atom_run_deck.html", "atom_run_deck.json"])
        self.assertEqual(links["reportUrl"], "runs/atom/loose/atom_run_deck.html")

    def test_copies_do_not_keep_read_only_permissions(self):
        run = write_run(self.inputs, "rvs_cvs", deck("rvs_cvs"))
        source = run / "rvs_cvs_html" / "rvs_cvs_run_deck.html"
        os.chmod(source, stat.S_IREAD)
        self.addCleanup(os.chmod, source, stat.S_IREAD | stat.S_IWRITE)
        links = writer.place_run(self.only_deck(), "rvs_cvs", self.report, self.warnings)
        self.assertTrue(os.stat(self.report / links["reportUrl"]).st_mode & stat.S_IWRITE)

    def test_bundle_member_outside_the_folder_is_not_extracted(self):
        bundle = self.tmp / "evil.zip"
        with zipfile.ZipFile(bundle, "w") as out:
            out.writestr("../escaped.txt", "x")
            out.writestr("/abs.txt", "x")
            out.writestr("ok/inside.txt", "x")
        target = self.tmp / "target"
        target.mkdir()
        writer.extract_bundle(bundle, target)
        self.assertTrue((target / "ok" / "inside.txt").is_file())
        self.assertFalse((self.tmp / "escaped.txt").exists())
        self.assertEqual([path.name for path in target.rglob("*") if path.is_file()], ["inside.txt"])

    def test_unreadable_bundle_member_is_reported_and_the_rest_kept(self):
        write_run(self.inputs, "rccl_perf", deck("rccl", "record", "series"), folder=False, bundle="2026-11-10T193205")
        found = self.only_deck()
        with patch.object(writer, "extract_bundle", side_effect=zipfile.BadZipFile("Bad CRC-32")):
            with self.assertLogs(level="WARNING"):
                links = writer.place_run(found, "rccl_perf", self.report, self.warnings)
        self.assertEqual(self.warnings.entries[0]["code"], "unreadable_file")
        self.assertIsNone(links["reportUrl"])


class TestBackLink(WriterCase):
    def test_link_is_added_once_and_points_at_the_overview(self):
        page = self.report / "runs" / "rvs_cvs" / "local-1" / "rvs_cvs_html" / "deck.html"
        page.parent.mkdir(parents=True)
        page.write_text('<html><body class="x"><h1>RVS</h1></body></html>', encoding="utf-8")
        self.assertTrue(writer.add_back_link(page, self.report))
        self.assertFalse(writer.add_back_link(page, self.report))
        document = page.read_text(encoding="utf-8")
        self.assertEqual(document.count(writer.BACK_LINK_ID), 1)
        self.assertIn('<body class="x"><nav id="%s"' % writer.BACK_LINK_ID, document)
        self.assertIn('href="../../../../index.html"', document)

    def test_link_goes_into_the_cvs_header_when_the_deck_has_one(self):
        page = self.report / "runs" / "rvs_cvs" / "local-1" / "rvs_cvs_html" / "deck.html"
        page.parent.mkdir(parents=True)
        page.write_text(
            '<html><body><div class="cvs-header">brand%s</div><h1>RVS</h1></body></html>' % OVERVIEW_LINK_MARKER,
            encoding="utf-8",
        )
        self.assertTrue(writer.add_back_link(page, self.report))
        self.assertFalse(writer.add_back_link(page, self.report))
        document = page.read_text(encoding="utf-8")
        link = '<a id="%s" class="cvs-back" href="../../../../index.html">' % writer.BACK_LINK_ID
        self.assertIn("brand" + link, document)
        self.assertNotIn(OVERVIEW_LINK_MARKER, document)
        self.assertNotIn("<nav", document)

    def test_page_without_a_body_is_left_alone(self):
        page = self.report / "fragment.html"
        page.write_text("<div>no body</div>", encoding="utf-8")
        self.assertFalse(writer.add_back_link(page, self.report))
        self.assertEqual(page.read_text(encoding="utf-8"), "<div>no body</div>")


class TestRenderIndex(unittest.TestCase):
    def test_deck_strings_cannot_close_the_data_block(self):
        hostile = "</script><script>alert(1)</script>"
        page = writer.render_index({"summary": {"suites": [{"name": hostile}]}, "details": {}, "areas": []})
        self.assertNotIn(hostile, page)
        data = page.split('<script type="application/json" id="report-data">', 1)[1].split("</script>", 1)[0]
        self.assertNotIn("<", data)
        self.assertEqual(json.loads(data)["summary"]["suites"][0]["name"], hostile)

    def test_page_has_no_placeholders_left(self):
        page = writer.render_index({"summary": {}, "details": {}, "areas": []})
        for placeholder in ("__THEME_BOOT__", "__CSS__", "__JS__", "__DATA__"):
            self.assertNotIn(placeholder, page)

    def test_overview_shares_the_deck_theme(self):
        page = writer.render_index({"summary": {}, "details": {}, "areas": []})
        self.assertIn(THEME_BOOT_SCRIPT, page)
        self.assertIn(tokens_css(), page)
        self.assertIn('<link rel="stylesheet" href="fonts/fonts.css"/>', page)


class TestFonts(WriterCase):
    def test_fonts_ship_with_their_licenses(self):
        writer.copy_fonts(self.report)
        fonts = self.report / "fonts"
        referenced = re.findall(r"url\(([^)]+)\)", (fonts / "fonts.css").read_text(encoding="utf-8"))
        self.assertEqual(len(referenced), 9)
        for name in referenced:
            with self.subTest(font=name):
                self.assertTrue((fonts / name).is_file())
        for name in ("OFL-Inter.txt", "OFL-SpaceGrotesk.txt", "OFL-IBMPlexMono.txt"):
            with self.subTest(license=name):
                self.assertIn("SIL Open Font License", (fonts / name).read_text(encoding="utf-8"))


class TestWriteZip(WriterCase):
    def make_report(self, name="consolidated_report_2026-11-10T193205"):
        root = self.tmp / "work" / name
        (root / "data").mkdir(parents=True)
        (root / "index.html").write_text("<html></html>", encoding="utf-8")
        (root / "data" / "x.json").write_text("{}", encoding="utf-8")
        return root

    def test_zip_holds_the_report_folder(self):
        root = self.make_report()
        zip_path = writer.write_zip(root, self.tmp)
        self.assertEqual(zip_path.name, "consolidated_report_2026-11-10T193205.zip")
        with zipfile.ZipFile(zip_path) as out:
            self.assertEqual(
                sorted(out.namelist()),
                [
                    "consolidated_report_2026-11-10T193205/data/x.json",
                    "consolidated_report_2026-11-10T193205/index.html",
                ],
            )

    def test_existing_name_gets_a_numeric_suffix(self):
        root = self.make_report()
        first = writer.write_zip(root, self.tmp)
        second = writer.write_zip(root, self.tmp)
        self.assertEqual(second.name, "consolidated_report_2026-11-10T193205-1.zip")
        self.assertTrue(first.is_file())

    def test_failure_partway_leaves_no_zip_behind(self):
        root = self.make_report()
        with patch.object(zipfile.ZipFile, "write", side_effect=OSError("No space left on device")):
            with self.assertRaises(OSError):
                writer.write_zip(root, self.tmp)
        self.assertEqual([path.name for path in self.tmp.iterdir() if path.is_file()], [])

    def test_write_json(self):
        path = self.tmp / "a" / "b.json"
        writer.write_json(path, {"name": "caf\u00e9"})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"name": "caf\u00e9"})


if __name__ == "__main__":
    unittest.main()
