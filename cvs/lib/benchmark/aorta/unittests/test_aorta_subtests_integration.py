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


def _panel_cells(entry):
    """Return (result, label, actual, gate, note) for each row of the parent's sub-test panel."""
    rows = []
    for extra in entry.get("extras") or []:
        content = str(extra.get("content") or "")
        for row in re.findall(r"<tr class='cvs-benchmark-metric-row[^']*'>(.*?)</tr>", content):
            cells = dict(re.findall(r"<td class='col-(\w+)'>(.*?)</td>", row))
            rows.append(
                tuple(html.unescape(cells.get(key, "")) for key in ("result", "testId", "actual", "gate", "reason"))
            )
    return rows


def _panel_rows(entry):
    return [(result, label) for result, label, *_rest in _panel_cells(entry)]


def _junit_counts(xml_root):
    """Return the testsuite's (tests, failures, errors) and each testcase's <failure> messages."""
    suite = xml_root if xml_root.tag == "testsuite" else xml_root.find("testsuite")
    totals = tuple(int(suite.get(key)) for key in ("tests", "failures", "errors"))
    failures = {
        case.get("name"): [item.get("message") for item in case.findall("failure")] for case in suite.iter("testcase")
    }
    return totals, failures


_HEADER = '''
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
'''


def _run_pytest(source):
    """Run source as a test file with the Aorta conftest; return the process, HTML text and JUnit root."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_path = root / "test_aorta_subtests.py"
        html_path = root / "report.html"
        xml_path = root / "report.xml"
        test_path.write_text(_HEADER + textwrap.dedent(source))
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
        html_text = html_path.read_text() if html_path.is_file() else ""
        xml_root = element_tree.parse(xml_path).getroot() if xml_path.is_file() else None
    return completed, details, html_text, xml_root


class TestAortaSubtestsIntegration(unittest.TestCase):
    def test_subtests_render_in_html_junit_and_fail_parents(self):
        self.assertTrue((REPO_ROOT / "cvs" / "conftest.py").is_file())
        source = textwrap.dedent(
            '''
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
        completed, details, html_text, xml_root = _run_pytest(source)
        self.assertEqual(completed.returncode, 1, details)
        self.assertIn("SUBFAIL", completed.stdout, details)
        self.assertIn("max_avg_iteration_ms", completed.stdout, details)
        self.assertIn("node-b", completed.stdout, details)
        tests = _html_tests(html_text)

        self.assertEqual(len(tests), 2, details)
        entries = {nodeid.rsplit("::", 1)[-1]: rows for nodeid, rows in tests.items()}
        self.assertEqual(set(entries), {"test_validate_thresholds", "test_collect_traces"}, details)
        for rows in entries.values():
            self.assertEqual(len(rows), 1, details)
            self.assertIn("Failed", rows[0]["resultsTableRow"][0], details)
        self.assertEqual(
            _panel_cells(entries["test_validate_thresholds"][0]),
            [
                (
                    "Failed",
                    "[threshold] (metric=max_avg_iteration_ms)",
                    "20.00 ms",
                    "max 10.00",
                    "Average iteration time 20.00ms exceeds threshold 10ms",
                ),
                ("Passed", "[threshold] (metric=min_compute_ratio)", "0.5000", "min 0.0100", ""),
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
        # JUnit counts every sub-test report: 2 parents + 4 sub-tests. One failed
        # threshold and one failed node each give a sub-test <failure> and a parent
        # <failure>, both under the parent's single <testcase>.
        totals, failures = _junit_counts(xml_root)
        self.assertEqual(totals, (6, 4, 0), details)
        self.assertEqual(
            failures,
            {
                "test_validate_thresholds": [
                    "Failed: Average iteration time 20.00ms exceeds threshold 10ms",
                    "Failed: Following FAILURES seen - ['Average iteration time 20.00ms exceeds threshold 10ms']",
                ],
                "test_collect_traces": [
                    "Failed: Trace collection on node-b: download failed",
                    "Failed: Following FAILURES seen - ['Trace collection on node-b: download failed']",
                ],
            },
            details,
        )
        properties = {
            name: [prop.get("value") for prop in case.iter("property") if prop.get("name") == "cvs_aorta_subtest"]
            for name, case in testcases.items()
        }
        self.assertEqual(
            properties["test_validate_thresholds"],
            [
                "[threshold] (metric=max_avg_iteration_ms): failed; actual 20.00 ms; gate max 10.00 ms; "
                "Average iteration time 20.00ms exceeds threshold 10ms",
                "[threshold] (metric=min_compute_ratio): passed; actual 0.5000; gate min 0.0100",
            ],
            details,
        )
        self.assertEqual(
            properties["test_collect_traces"],
            [
                "[trace collection] (host=node-a): passed",
                "[trace collection] (host=node-b): failed; Trace collection on node-b: download failed",
            ],
            details,
        )

    def test_missing_trace_artifact_message_survives_html_patch(self):
        source = """
            @pytest.fixture
            def traces_job():
                return SimpleNamespace(
                    started=True,
                    hosts=["node-a"],
                    trace_trees={"node-a": ["t"]},
                    collection_errors={},
                    collect_traces=lambda: None,
                    collect_logs=lambda: None,
                    get_artifact=lambda name: None,
                )

            def test_collect_traces(traces_job, lifecycle, request, subtests):
                _common.collect_traces(traces_job, lifecycle, request, subtests)
            """
        completed, details, html_text, xml_root = _run_pytest(source)
        self.assertEqual(completed.returncode, 1, details)
        entries = {nodeid.rsplit("::", 1)[-1]: rows for nodeid, rows in _html_tests(html_text).items()}
        self.assertEqual(list(entries), ["test_collect_traces"], details)
        (entry,) = entries["test_collect_traces"]
        self.assertIn("Failed", entry["resultsTableRow"][0], details)
        # The panel replaces the row's log, so the failure must be one of its rows.
        self.assertEqual(
            [(result, label, note) for result, label, _actual, _gate, note in _panel_cells(entry)],
            [
                ("Passed", "[trace collection] (host=node-a)", ""),
                (
                    "Failed",
                    "[trace collection] (artifact=torch_traces)",
                    "No fresh torch_traces artifact was collected",
                ),
            ],
            details,
        )
        totals, failures = _junit_counts(xml_root)
        self.assertEqual(totals, (3, 2, 0), details)
        self.assertEqual(
            failures["test_collect_traces"],
            [
                "Failed: No fresh torch_traces artifact was collected",
                "Failed: Following FAILURES seen - ['No fresh torch_traces artifact was collected']",
            ],
            details,
        )
        (case,) = xml_root.iter("testcase")
        self.assertEqual(
            [prop.get("value") for prop in case.iter("property") if prop.get("name") == "cvs_aorta_subtest"],
            [
                "[trace collection] (host=node-a): passed",
                "[trace collection] (artifact=torch_traces): failed; No fresh torch_traces artifact was collected",
            ],
            details,
        )

    def test_passing_subtests_add_tests_but_no_failures(self):
        source = """
            @pytest.fixture
            def aorta_job():
                config = SimpleNamespace(
                    enforce_thresholds=True,
                    thresholds={"expected_results": {"max_avg_iteration_ms": 100, "min_compute_ratio": 0.01}},
                )
                return SimpleNamespace(config=config)

            def test_validate_thresholds(aorta_job, lifecycle, request, subtests):
                _common.validate_thresholds(aorta_job, lifecycle, request, subtests)
            """
        completed, details, html_text, xml_root = _run_pytest(source)
        self.assertEqual(completed.returncode, 0, details)
        totals, failures = _junit_counts(xml_root)
        self.assertEqual(totals, (3, 0, 0), details)
        self.assertEqual(failures, {"test_validate_thresholds": []}, details)
        (entry,) = next(iter(_html_tests(html_text).values()))
        self.assertIn("Passed", entry["resultsTableRow"][0], details)
        self.assertEqual(
            _panel_cells(entry),
            [
                ("Passed", "[threshold] (metric=max_avg_iteration_ms)", "20.00 ms", "max 100.00", ""),
                ("Passed", "[threshold] (metric=min_compute_ratio)", "0.5000", "min 0.0100", ""),
            ],
            details,
        )
