'''Unit tests for cvs.lib.anc_rundeck (Run Deck result capture from ANC artifacts).'''

import json
import os
import tempfile
import unittest

from cvs.lib import anc_rundeck


class TestMakeMeta(unittest.TestCase):
    def test_pulls_cluster_and_version(self):
        meta = anc_rundeck.make_meta(
            {"cluster_name": "helios"},
            {"anc": {"anc_version": "1.4.9"}},
            "anc_test_cpu",
            "2026-09-23",
        )
        self.assertEqual(meta["cluster"], "helios")
        self.assertEqual(meta["version"], "1.4.9")
        self.assertEqual(meta["suite"], "anc_test_cpu")
        self.assertEqual(meta["generated_at"], "2026-09-23")

    def test_defaults_when_missing(self):
        meta = anc_rundeck.make_meta({}, {}, "", "")
        self.assertEqual(meta["cluster"], "—")
        self.assertEqual(meta["version"], "—")


class TestParseItemsSummary(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix="_console.log")
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_returns_verbatim_summary_line(self):
        path = self._write("noise\nItems: 12 Total | 10 PASSED, 2 FAILED, 0 SKIPPED\nmore\n")
        line = anc_rundeck._parse_items_summary(path)
        self.assertEqual(line, "Items: 12 Total | 10 PASSED, 2 FAILED, 0 SKIPPED")

    def test_missing_file_returns_empty(self):
        self.assertEqual(anc_rundeck._parse_items_summary("/no/such/console.log"), "")

    def test_no_summary_line(self):
        path = self._write("just some logs\nno summary here\n")
        self.assertEqual(anc_rundeck._parse_items_summary(path), "")


class TestParseErrorItems(unittest.TestCase):
    def _write_json(self, obj):
        fd, path = tempfile.mkstemp(suffix="_errors.json")
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
        self.addCleanup(os.remove, path)
        return path

    def test_real_anc_schema_errors_dict(self):
        # Real ANC errors.json (schema 0.2): errors is a DICT keyed by item id.
        path = self._write_json(
            {
                "release_version": "1.7.0-rc.1",
                "schema_version": "0.2",
                "errors": {
                    "i002-oblex_remix2": {
                        "status": "FAILED",
                        "name": "oblex_remix2",
                        "rc_enum": "ANC_PDC_DMESG_ERROR",
                        "rc_desc": "Passive Data Collector detected dmesg errors during test execution.",
                        "summary": "PDC Failed: DmesgPlugin (see: items/i002-oblex_remix2/dmesg_plugin)",
                    }
                },
            }
        )
        items = anc_rundeck._parse_error_items(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "oblex_remix2")
        self.assertEqual(items[0]["status"], "fail")
        self.assertEqual(
            items[0]["message"],
            "ANC_PDC_DMESG_ERROR: PDC Failed: DmesgPlugin (see: items/i002-oblex_remix2/dmesg_plugin)",
        )

    def test_list_of_records_fallback(self):
        path = self._write_json({"errors": [{"name": "hbm_ecc", "summary": "bad"}]})
        items = anc_rundeck._parse_error_items(path)
        self.assertEqual(items[0]["name"], "hbm_ecc")
        self.assertEqual(items[0]["message"], "bad")

    def test_key_used_when_no_name(self):
        path = self._write_json({"errors": {"i005-foo": {"status": "FAILED", "rc_desc": "boom"}}})
        items = anc_rundeck._parse_error_items(path)
        self.assertEqual(items[0]["name"], "i005-foo")
        self.assertEqual(items[0]["message"], "boom")

    def test_missing_file(self):
        self.assertEqual(anc_rundeck._parse_error_items("/no/such.json"), [])

    def test_malformed_json(self):
        fd, path = tempfile.mkstemp(suffix="_errors.json")
        with os.fdopen(fd, "w") as fh:
            fh.write("{not json")
        self.addCleanup(os.remove, path)
        self.assertEqual(anc_rundeck._parse_error_items(path), [])


class TestBuildNodeRecord(unittest.TestCase):
    def _console(self, text):
        fd, path = tempfile.mkstemp(suffix="_console.log")
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def _errors(self, obj):
        fd, path = tempfile.mkstemp(suffix="_errors.json")
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
        self.addCleanup(os.remove, path)
        return path

    def test_pass_record_keeps_summary_but_no_item_rows(self):
        # Passing node: summary line kept for the cell label; NO fabricated rows.
        console = self._console("Items: 3 Total | 3 PASSED, 0 FAILED\n")
        rec = anc_rundeck.build_node_record(
            status="pass",
            console_path=console,
            errors_json_path=None,
            errors_json_href="",
            log_tarball_href="",
        )
        self.assertEqual(rec["status"], "pass")
        self.assertEqual(rec["items"], [])
        self.assertIn("3 PASSED", rec["items_summary"])
        self.assertNotIn("return_code", rec)

    def test_fail_record_surfaces_only_real_error_items(self):
        console = self._console("Items: 5 Total | 3 PASSED, 2 FAILED\n")
        errors = self._errors({"errors": {"i1-ecc": {"name": "ecc", "summary": "812<900"}}})
        rec = anc_rundeck.build_node_record(
            status="fail",
            console_path=console,
            errors_json_path=errors,
            errors_json_href="n_errors.json",
            log_tarball_href="n_logs.tar.gz",
        )
        # Only the real errors.json failures appear -- no synthetic passed[N] rows.
        self.assertEqual([i["name"] for i in rec["items"]], ["ecc"])
        self.assertTrue(all(i["status"] == "fail" for i in rec["items"]))
        self.assertEqual(rec["errors_json_href"], "n_errors.json")
        self.assertEqual(rec["log_tarball_href"], "n_logs.tar.gz")

    def test_fail_without_errors_json_has_no_items(self):
        # No errors.json -> no fabricated rows; the summary line still carries counts.
        console = self._console("Items: 2 Total | 0 PASSED, 2 FAILED\n")
        rec = anc_rundeck.build_node_record(
            status="fail",
            console_path=console,
            errors_json_path=None,
            errors_json_href="",
            log_tarball_href="",
        )
        self.assertEqual(rec["items"], [])
        self.assertIn("2 FAILED", rec["items_summary"])


class TestRecordGroup(unittest.TestCase):
    def test_merges_groups_and_sets_meta_once(self):
        acc = {}
        anc_rundeck.record_group(acc, "cpu_sanity", {"n1": {"status": "pass"}}, meta={"cluster": "c"})
        anc_rundeck.record_group(acc, "hbm_lvl3", {"n1": {"status": "fail"}}, meta={"cluster": "OTHER"})
        self.assertEqual(acc["_meta"]["cluster"], "c")  # first meta wins
        self.assertIn("cpu_sanity", acc["groups"])
        self.assertIn("hbm_lvl3", acc["groups"])
        self.assertEqual(acc["groups"]["cpu_sanity"]["nodes"]["n1"]["status"], "pass")

    def test_rerun_same_group_overwrites_node(self):
        acc = {}
        anc_rundeck.record_group(acc, "g", {"n1": {"status": "fail"}})
        anc_rundeck.record_group(acc, "g", {"n1": {"status": "pass"}})
        self.assertEqual(acc["groups"]["g"]["nodes"]["n1"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
