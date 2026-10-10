"""Pytest HTML and JUnit coverage for Aorta sub-tests."""

import html
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]


def _html_tests(html_text):
    match = re.search(r'data-jsonblob="([^"]*)"', html_text)
    return json.loads(html.unescape(match.group(1)))["tests"]


def _panel_rows(entry):
    rows = []
    for extra in entry.get("extras") or []:
        content = str(extra.get("content") or "")
        rows.extend(
            (result, html.unescape(label))
            for result, label in re.findall(
                r"<td class='col-result'>(\w+)</td>(?:<td class='col-node'>[^<]*</td>)?"
                r"<td class='col-testId'>([^<]+)",
                content,
            )
        )
    return rows


class TestAortaSubtestsIntegration(unittest.TestCase):
    def test_subtests_render_in_html_junit_and_fail_parents(self):
        self.assertTrue((REPO_ROOT / "cvs" / "conftest.py").is_file())
        source = textwrap.dedent(
            '''
            from types import SimpleNamespace

            import pytest

            from cvs.parsers.tracelens import TraceLensParser
            from cvs.tests.benchmark.aorta import _common

            @pytest.fixture
            def lifecycle():
                result = SimpleNamespace(
                    avg_iteration_time_ms=20.0,
                    avg_iteration_time_us=20000.0,
                    std_iteration_time_us=100.0,
                    avg_compute_ratio=0.5,
                    avg_overlap_ratio=0.1,
                )
                return SimpleNamespace(failed=False, parser=TraceLensParser(use_tracelens=False), benchmark_result=result)

            @pytest.fixture
            def aorta_job():
                config = SimpleNamespace(
                    enforce_thresholds=True,
                    thresholds={"expected_results": {
                        "max_avg_iteration_ms": 10,
                        "min_compute_ratio": 0.01,
                        "min_overlap_ratio": None,
                    }},
                )
                return SimpleNamespace(config=config)

            @pytest.fixture
            def traces_job():
                return SimpleNamespace(
                    started=True,
                    hosts=["node-a", "node-b"],
                    trace_trees={"node-a": ["t"]},
                    collection_errors={"node-b": "download failed"},
                    collect_traces=lambda: None,
                    collect_logs=lambda: None,
                    get_artifact=lambda name: "traces",
                )

            def test_validate_thresholds(aorta_job, lifecycle, request, subtests):
                _common.validate_thresholds(aorta_job, lifecycle, request, subtests)

            def test_collect_traces(traces_job, lifecycle, request, subtests):
                _common.collect_traces(traces_job, lifecycle, request, subtests)
            '''
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "test_aorta_subtests.py"
            html_path = root / "report.html"
            xml_path = root / "report.xml"
            test_path.write_text(source)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_path),
                    "-p",
                    "cvs.tests.benchmark.aorta.conftest",
                    "-p",
                    "no:cacheprovider",
                    f"--html={html_path}",
                    "--self-contained-html",
                    f"--junitxml={xml_path}",
                    "-q",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            details = f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            self.assertEqual(completed.returncode, 1, details)
            self.assertIn("SUBFAIL", completed.stdout, details)
            self.assertIn("max_avg_iteration_ms", completed.stdout, details)
            self.assertIn("node-b", completed.stdout, details)
            html_text = html_path.read_text()
            tests = _html_tests(html_text)
            xml_root = element_tree.parse(xml_path).getroot()

        self.assertEqual(len(tests), 2, details)
        entries = {nodeid.rsplit("::", 1)[-1]: rows for nodeid, rows in tests.items()}
        self.assertEqual(set(entries), {"test_validate_thresholds", "test_collect_traces"}, details)
        for rows in entries.values():
            self.assertEqual(len(rows), 1, details)
            self.assertIn("Failed", rows[0]["resultsTableRow"][0], details)
        self.assertEqual(
            _panel_rows(entries["test_validate_thresholds"][0]),
            [
                ("Failed", "[threshold] (metric=max_avg_iteration_ms)"),
                ("Passed", "[threshold] (metric=min_compute_ratio)"),
            ],
            details,
        )
        self.assertEqual(
            _panel_rows(entries["test_collect_traces"][0]),
            [
                ("Passed", "[trace collection] (host=node-a)"),
                ("Failed", "[trace collection] (host=node-b)"),
            ],
            details,
        )
        self.assertNotIn("min_overlap_ratio", str([_panel_rows(rows[0]) for rows in entries.values()]))
        self.assertIn('cvs-subtests-count">4 subtests,', html_text, details)
        testcases = {case.get("name"): case for case in xml_root.iter("testcase")}
        self.assertEqual(set(testcases), {"test_validate_thresholds", "test_collect_traces"}, details)
        for case in testcases.values():
            self.assertTrue(case.findall("failure"), details)
        properties = {
            name: [prop.get("value") for prop in case.iter("property") if prop.get("name") == "cvs_aorta_subtest"]
            for name, case in testcases.items()
        }
        self.assertEqual(
            properties["test_validate_thresholds"],
            ["[threshold] (metric=max_avg_iteration_ms): failed", "[threshold] (metric=min_compute_ratio): passed"],
            details,
        )
        self.assertEqual(
            properties["test_collect_traces"],
            ["[trace collection] (host=node-a): passed", "[trace collection] (host=node-b): failed"],
            details,
        )
