"""
Unit tests for cvs.parsers.tracelens.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import tempfile
import unittest
from pathlib import Path

from cvs.parsers.schemas import ParseStatus
from cvs.parsers.tracelens import TraceLensParser
from cvs.runners._base_runner import RunResult, RunStatus


def _write_trace(trace_dir: Path, rank: int) -> None:
    rank_dir = trace_dir / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "traceEvents": [
            {"name": "aten::matmul", "cat": "kernel", "dur": 100},
        ]
    }
    (rank_dir / "trace.json").write_text(json.dumps(trace))


class TestParseGatingOnRunStatus(unittest.TestCase):
    def setUp(self):
        self.parser = TraceLensParser(use_tracelens=False)

    def test_failed_run_with_traces_on_disk_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "torch_traces"
            _write_trace(trace_dir, 0)
            run_result = RunResult(
                status=RunStatus.FAILED,
                start_time=0,
                end_time=1,
                error_message="node b timed out",
                artifacts={"torch_traces": trace_dir},
            )
            result = self.parser.parse(run_result)
            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertIn("Run did not succeed: node b timed out", result.warnings)

    def test_failed_run_with_no_artifact_still_fails(self):
        run_result = RunResult(status=RunStatus.FAILED, start_time=0, end_time=1, error_message="all nodes died")
        result = self.parser.parse(run_result)
        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertIn("Run did not succeed: all nodes died", result.warnings)
        self.assertIn("No torch_traces artifact found in run result", result.errors)

    def test_completed_run_parses_without_run_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "torch_traces"
            _write_trace(trace_dir, 0)
            run_result = RunResult(
                status=RunStatus.COMPLETED, start_time=0, end_time=1, artifacts={"torch_traces": trace_dir}
            )
            result = self.parser.parse(run_result)
            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
