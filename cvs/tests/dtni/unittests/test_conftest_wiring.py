"""
B5 wiring -- the DTNI conftest's workload_run fixture MUST construct
``Job(adapter, ctx, scanner=FailurePatternScanner())``. Without the scanner
threaded in, captured dmesg/container.log streams are never scanned and the
tier-1 ``test_dmesg_clean`` style assertions are vacuous -- the run "passes"
on a real OOM.

This test reaches into the conftest helper (``_build_job``) directly so the
assertion does not require a live cluster / executor / full pytest session.
"""

from __future__ import annotations

import unittest

from cvs.lib.failure_pattern_scanner import FailurePatternScanner
from cvs.lib.job import Job


class _StubAdapter:
    def prepare(self, ctx): ...
    def launch(self, ctx): ...
    def await_completion(self, ctx): ...
    def progress_predicate(self, ctx):
        return True

    def parse(self, ctx): ...
    def verify(self, ctx): ...
    def teardown(self, ctx): ...


class _StubCtx:
    """Bare-minimum RunContext stand-in -- _build_job must not touch it."""


class TestB5ScannerWiring(unittest.TestCase):
    def test_build_job_threads_scanner(self) -> None:
        """_build_job(adapter, ctx) -> Job whose scanner is a FailurePatternScanner.

        This is the contract the Integration Milestone's tier-1 dmesg tests
        rely on. If anyone refactors the conftest to drop scanner=, this
        test must fail BEFORE the next adapter run silently misses an OOM.
        """
        from cvs.tests.dtni.conftest import _build_job

        job = _build_job(_StubAdapter(), _StubCtx())
        self.assertIsInstance(job, Job)
        self.assertIsNotNone(
            job.scanner,
            "Job.scanner is None -- B5 contract violated; dmesg/container.log "
            "scanning is dead and tier-1 assertions are vacuous",
        )
        self.assertIsInstance(
            job.scanner,
            FailurePatternScanner,
            f"Expected FailurePatternScanner, got {type(job.scanner).__name__}",
        )

    def test_scanner_has_loaded_catalog(self) -> None:
        """Sanity: the wired scanner has its seed catalog loaded (G6b smoke)."""
        from cvs.tests.dtni.conftest import _build_job

        job = _build_job(_StubAdapter(), _StubCtx())
        self.assertTrue(
            hasattr(job.scanner, "patterns") and len(job.scanner.patterns) > 0,
            "Wired scanner has no loaded patterns -- catalog wiring is broken",
        )


if __name__ == "__main__":
    unittest.main()
