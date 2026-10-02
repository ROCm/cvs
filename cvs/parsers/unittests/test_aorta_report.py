"""
Unit tests for cvs.parsers.aorta_report.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.schemas import ParseStatus
from cvs.runners._base_runner import RunResult, RunStatus


def _write_report(individual_dir: Path, rank: int) -> None:
    individual_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "type": ["total_time", "computation_time", "exposed_comm_time", "total_comm_time"],
            "time ms": [10.0, 6.0, 2.0, 3.0],
        }
    )
    df.to_excel(individual_dir / f"perf_rank{rank}.xlsx", sheet_name="gpu_timeline", index=False)


class TestParseGatingOnRunStatus(unittest.TestCase):
    def setUp(self):
        self.parser = AortaReportParser()

    def test_failed_run_with_reports_on_disk_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "tracelens_analysis"
            _write_report(analysis_dir / "individual_reports", 0)
            run_result = RunResult(
                status=RunStatus.FAILED,
                start_time=0,
                end_time=1,
                error_message="node b timed out",
                artifacts={"tracelens_analysis": analysis_dir},
            )
            result = self.parser.parse(run_result)
            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertIn("Run did not succeed: node b timed out", result.warnings)

    def test_failed_run_with_no_artifact_still_fails(self):
        run_result = RunResult(status=RunStatus.FAILED, start_time=0, end_time=1, error_message="all nodes died")
        result = self.parser.parse(run_result)
        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertIn("Run did not succeed: all nodes died", result.warnings)
        self.assertIn(
            "No tracelens_analysis artifact found. Ensure analysis.enable_tracelens is set in config.",
            result.errors,
        )

    def test_completed_run_parses_without_run_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp) / "tracelens_analysis"
            _write_report(analysis_dir / "individual_reports", 0)
            run_result = RunResult(
                status=RunStatus.COMPLETED, start_time=0, end_time=1, artifacts={"tracelens_analysis": analysis_dir}
            )
            result = self.parser.parse(run_result)
            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
