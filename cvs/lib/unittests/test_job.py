"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cvs.lib.adapter_protocol import Progress
from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.config import SweepCell, parse_config
from cvs.lib.config.thresholds import ResultView
from cvs.lib.failure_pattern_scanner import FailurePattern, PatternHit
from cvs.lib.failure_taxonomy import FailureCategory
from cvs.lib.job import Job
from cvs.lib.manifest.events import EventWriter
from cvs.lib.manifest.layout import RunLayout
from cvs.lib.run_context import RunContext

BASE = {
    "framework": "vllm",
    "model": "m",
    "topology": {"nnodes": 1},
    "params": {"server_script": "s.sh"},
    "container": {"image": "rocm/vllm-dev:nightly", "env": {"HF_TOKEN": "secret-xyz", "DEBUG": "1"}},
}


class _Adapter(BaseWorkloadAdapter):
    framework = "fake"
    completion_timeout_s = 0.0
    poll_interval_s = 0.0

    def __init__(self, *, fail_prepare=False, samples=None, record_commands=None):
        self.fail_prepare = fail_prepare
        self.samples = samples or []
        self.teardown_called = False
        self.record_commands = record_commands or []

    def prepare(self, ctx):
        if self.fail_prepare:
            raise RuntimeError("boom")

    def launch(self, ctx):
        if self.record_commands:
            ctx.scratch.setdefault("commands", []).extend(self.record_commands)

    def progress_predicate(self, ctx):
        return Progress.DONE

    def parse(self, ctx):
        ctx.result = ResultView(samples=self.samples)

    def teardown(self, ctx):
        self.teardown_called = True


def _ctx(cfg, tmp, rid):
    # Single-cell shape: VllmConfig no longer carries a `sweep` block (the
    # YAML scalars under params are the cell). Build a stub SweepCell so
    # RunContext + Job's manifest path see the same shape PR-Z will lower.
    cell = SweepCell(id="single", params={})
    layout = RunLayout(tmp, "t", cell.id, "h", rid)
    return RunContext(cfg, cell, {"server": ["node-a"]}, layout, EventWriter(layout.events_path), rid)


def _fatal_scanner(severity="fatal"):
    """Minimal scanner double exposing the contract Job._scan_patterns uses."""

    class _S:
        patterns = [
            FailurePattern(
                id="hbm_ecc",
                source="dmesg",
                pattern="hbm",
                category=FailureCategory.FAILURE_PATTERN_MATCHED,
                severity=severity,
                hint="HBM ECC",
            )
        ]

        def scan(self, text, source=None):
            if not text or (source is not None and source != "dmesg"):
                return []
            return [
                PatternHit(
                    pattern_id="hbm_ecc",
                    category=FailureCategory.FAILURE_PATTERN_MATCHED,
                    source="dmesg",
                    line_no=1,
                    line="hbm ecc",
                )
            ]

    return _S()


class TestJobLifecycle(unittest.TestCase):
    """Brief minimum (one failure class + teardown-in-finally) plus the B6/B7
    bake-ins the addendum mandates."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = parse_config(BASE)

    def test_setup_failure_classified_at_boundary_b4(self):
        # Brief: "one failure-class wired through Job.run (mocked adapter)".
        m = Job(_Adapter(fail_prepare=True), _ctx(self.cfg, self.tmp, "s")).run()
        self.assertEqual(m.verdicts.overall_status, "failed")
        self.assertEqual(m.verdicts.failure_category, FailureCategory.SETUP_FAILURE.value)

    def test_teardown_runs_in_finally_on_failure(self):
        # Brief: "the teardown-in-finally fires even when the adapter raises".
        adapter = _Adapter(fail_prepare=True)
        ctx = _ctx(self.cfg, self.tmp, "t2")
        Job(adapter, ctx).run()
        self.assertTrue(adapter.teardown_called)

    def test_job_owns_phase_boundary_events_b6(self):
        # Addendum B6: Job owns prepare.*, parse.done, teardown.*.
        ctx = _ctx(self.cfg, self.tmp, "ev")
        Job(_Adapter(), ctx).run()
        text = ctx.layout.events_path.read_text()
        names = set()
        for line in text.splitlines():
            if '"event":' in line:
                names.add(line.split('"event": "', 1)[1].split('"', 1)[0])
        for required in ("prepare.start", "prepare.done", "parse.done", "teardown.start", "teardown.done"):
            self.assertIn(required, names, "Job did not emit phase-boundary event " + required)

    def test_manifest_env_and_commands_populated_b7_half(self):
        # Addendum B7 (G3+G5 half): ConfigInputs.env from container.env,
        # ConfigInputs.commands from ctx.scratch, both AS-IS (no redaction).
        adapter = _Adapter(record_commands=["docker run img:latest", "curl localhost:8000"])
        ctx = _ctx(self.cfg, self.tmp, "b7")
        m = Job(adapter, ctx).run()
        self.assertEqual(m.config.env, {"HF_TOKEN": "secret-xyz", "DEBUG": "1"})
        self.assertEqual(m.config.commands, ["docker run img:latest", "curl localhost:8000"])
        # Pin the no-redaction contract: a future "let's redact" patch must
        # delete this assertion (forcing the conversation).
        self.assertIn("secret-", m.config.env["HF_TOKEN"])


class TestFatalPatternOverride(unittest.TestCase):
    """B4 override is an upgrade, not a relabel: it beats complete/error but
    must NOT clobber a classified WorkloadFailure. Pins adversarial-review
    BLOCKER #1."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = parse_config(BASE)

    def test_classified_verification_failure_survives_fatal_pattern(self):
        cfg = parse_config(
            {**BASE, "thresholds": [{"type": "percentile", "metric": "ttft_ms", "op": "<=", "value": 1}]}
        )
        adapter = _Adapter(samples=[{"ttft_ms": 999}])
        ctx = _ctx(cfg, self.tmp, "vf_keep")
        ctx.logs["node-a.dmesg.txt"] = "hbm something"
        m = Job(adapter, ctx, scanner=_fatal_scanner()).run()
        self.assertEqual(m.verdicts.failure_category, FailureCategory.VERIFICATION_FAILURE.value)

    def test_successful_run_with_fatal_pattern_promotes_to_failed(self):
        ctx = _ctx(self.cfg, self.tmp, "ok_fatal")
        ctx.logs["node-a.dmesg.txt"] = "hbm something"
        m = Job(_Adapter(), ctx, scanner=_fatal_scanner()).run()
        self.assertEqual(m.verdicts.overall_status, "failed")
        self.assertEqual(m.verdicts.failure_category, FailureCategory.FAILURE_PATTERN_MATCHED.value)


class TestScannerContractFailsClosed(unittest.TestCase):
    """Pins adversarial-review BLOCKER #2: a scanner without `.patterns`
    cannot have its hits' severity resolved, so the B4 fatal override would
    silently never fire. Must raise instead."""

    def test_scanner_missing_patterns_attribute_raises(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = parse_config(BASE)

        class _BadScanner:
            def scan(self, text, source=None):
                return []

        ctx = _ctx(cfg, tmp, "bad")
        ctx.logs["node-a.dmesg.txt"] = "anything"
        with self.assertRaises(TypeError):
            Job(_Adapter(), ctx, scanner=_BadScanner()).run()


if __name__ == "__main__":
    unittest.main()
