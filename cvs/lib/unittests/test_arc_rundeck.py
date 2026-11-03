'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Tests for ARC Run Deck record construction.
'''

import unittest

from cvs.lib import arc_rundeck


EXAMPLE_FAILURE = {
    "overall_summary": {
        "status_code": "NOT_RUN",
        "total_selected_items": 6,
        "total_passed": 3,
        "total_completed": 1,
        "total_failed": 1,
        "total_skipped": 0,
        "total_not_run": 1,
    },
    "results": {
        "Mock_FailingTest": {"name": "Mock_FailingTest", "status": "FAILED", "rc": "AFID 15038", "summary": "ERROR"},
        "Mock_PassingTest": {"name": "Mock_PassingTest", "status": "NOT_RUN", "summary": "Parent skipped: Group_4"},
    },
}


class TestArcRundeck(unittest.TestCase):
    def test_meta_and_counts(self):
        meta = arc_rundeck.make_meta({"name": "rack"}, {"arc": {"arc_version": "1.8.1"}}, "arc_test_examples", "ts")
        self.assertEqual(meta["version_label"], "ARC version")
        self.assertEqual(meta["version"], "1.8.1")
        self.assertEqual(
            arc_rundeck.items_summary_line(EXAMPLE_FAILURE),
            "Items: 6 Total | 3 PASSED, 1 COMPLETED, 1 FAILED, 0 SKIPPED, 1 NOT RUN",
        )
        self.assertEqual(arc_rundeck.items_summary_line(None), "")
        self.assertEqual(arc_rundeck.items_summary_line({}), "")

    def test_failed_items_only(self):
        self.assertEqual(
            arc_rundeck.failed_items(EXAMPLE_FAILURE),
            [{"name": "Mock_FailingTest", "status": "fail", "message": "AFID 15038: ERROR"}],
        )
        self.assertEqual(arc_rundeck.failed_items({"results": []}), [])
        self.assertEqual(arc_rundeck.failed_items({"results": {"pass": {"status": "PASSED"}}}), [])

    def test_node_record(self):
        fail = arc_rundeck.build_node_record(
            status="fail", summary=EXAMPLE_FAILURE, errors_json_href="errors.json", log_tarball_href="logs.tar.gz"
        )
        self.assertEqual(fail["status"], "fail")
        self.assertEqual(len(fail["items"]), 1)
        self.assertEqual(fail["errors_json_href"], "errors.json")
        passing = arc_rundeck.build_node_record(
            status="pass", summary=EXAMPLE_FAILURE, errors_json_href="", log_tarball_href=""
        )
        self.assertEqual(passing["items"], [])
