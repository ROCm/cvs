"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import logging
import unittest
from typing import Optional

from cvs.core.lifecycle.base import PhaseError, Severity
from cvs.core.lifecycle.pipeline import Pipeline


class _RecordedPhase:
    """A configurable phase. Records every applies_to/run/undo call so the
    test can assert ordering and what got rolled back."""

    def __init__(
        self,
        name: str,
        severity: Severity = Severity.HARD_FAIL,
        applies: bool = True,
        raise_on_run: Optional[Exception] = None,
        raise_on_undo: Optional[Exception] = None,
        log: Optional[list] = None,
    ):
        self.name = name
        self.severity = severity
        self._applies = applies
        self._raise_on_run = raise_on_run
        self._raise_on_undo = raise_on_undo
        self._log = log if log is not None else []

    def applies_to(self, orch) -> bool:
        self._log.append(("applies", self.name))
        return self._applies

    def run(self, orch, artifact: dict) -> None:
        self._log.append(("run", self.name))
        if self._raise_on_run:
            raise self._raise_on_run
        artifact["did"] = "yes"

    def undo(self, orch, artifact: dict) -> None:
        self._log.append(("undo", self.name))
        if self._raise_on_undo:
            raise self._raise_on_undo


class _FakeOrch:
    """Minimal orchestrator stub: just needs .log."""

    def __init__(self):
        self.log = logging.getLogger("test_pipeline")


class TestPipelineHappyPath(unittest.TestCase):
    def test_phases_run_in_order_and_artifacts_are_recorded(self):
        log = []
        a = _RecordedPhase("a", log=log)
        b = _RecordedPhase("b", log=log)
        pipeline = Pipeline("prepare", [a, b])

        artifacts = pipeline.run(_FakeOrch())

        self.assertEqual(
            [step for step in log if step[0] == "run"],
            [("run", "a"), ("run", "b")],
        )
        self.assertEqual(artifacts["a"]["status"], "ok")
        self.assertEqual(artifacts["b"]["status"], "ok")
        self.assertEqual(artifacts["a"]["did"], "yes")


class TestPipelineAppliesTo(unittest.TestCase):
    def test_phase_that_does_not_apply_records_skipped_and_does_not_run(self):
        log = []
        a = _RecordedPhase("a", applies=True, log=log)
        b = _RecordedPhase("b", applies=False, log=log)
        c = _RecordedPhase("c", applies=True, log=log)
        artifacts = Pipeline("prepare", [a, b, c]).run(_FakeOrch())

        self.assertEqual(artifacts["b"], {"status": "skipped"})
        # b's run was never called
        ran = [step for step in log if step[0] == "run"]
        self.assertEqual(ran, [("run", "a"), ("run", "c")])


class TestPipelineHardFailAbortsAndRollsBack(unittest.TestCase):
    def test_hard_fail_undoes_prior_ok_phases_in_reverse_order(self):
        log = []
        a = _RecordedPhase("a", log=log)
        b = _RecordedPhase("b", log=log)
        c = _RecordedPhase(
            "c", severity=Severity.HARD_FAIL, raise_on_run=PhaseError("c blew up"),
            log=log,
        )
        d = _RecordedPhase("d", log=log)  # should never run
        pipeline = Pipeline("prepare", [a, b, c, d])

        with self.assertRaises(PhaseError) as ctx:
            pipeline.run(_FakeOrch())
        self.assertIn("c blew up", str(ctx.exception))

        # d.run should never have happened
        self.assertNotIn(("run", "d"), log)
        # undo order: b then a (c was never ok, d was never run)
        undos = [step for step in log if step[0] == "undo"]
        self.assertEqual(undos, [("undo", "b"), ("undo", "a")])


class TestPipelineSoftFailContinues(unittest.TestCase):
    def test_soft_fail_records_error_and_pipeline_keeps_going(self):
        log = []
        a = _RecordedPhase("a", log=log)
        b = _RecordedPhase(
            "b", severity=Severity.SOFT_FAIL, raise_on_run=PhaseError("just a warning"),
            log=log,
        )
        c = _RecordedPhase("c", log=log)
        artifacts = Pipeline("prepare", [a, b, c]).run(_FakeOrch())

        self.assertEqual(artifacts["a"]["status"], "ok")
        self.assertEqual(artifacts["b"]["status"], "failed")
        self.assertIn("just a warning", artifacts["b"]["error"])
        self.assertEqual(artifacts["c"]["status"], "ok")
        # c.run still happened despite b's soft fail
        self.assertIn(("run", "c"), log)


class TestPipelineRollbackPublic(unittest.TestCase):
    """The public rollback() walks artifacts (not internal state), so the
    Orchestrator can call rollback even after a successful run."""

    def test_rollback_walks_artifacts_in_reverse_for_ok_phases_only(self):
        log = []
        a = _RecordedPhase("a", log=log)
        b = _RecordedPhase("b", applies=False, log=log)  # skipped
        c = _RecordedPhase("c", log=log)
        pipeline = Pipeline("prepare", [a, b, c])
        artifacts = pipeline.run(_FakeOrch())

        # Now externally trigger rollback (mirrors Orchestrator.cleanup).
        pipeline.rollback(_FakeOrch(), artifacts)

        undos = [step for step in log if step[0] == "undo"]
        # c first (reverse), then a; b is skipped so no undo
        self.assertEqual(undos, [("undo", "c"), ("undo", "a")])


class TestPipelineUndoErrorsDoNotPropagate(unittest.TestCase):
    """A bug in undo() must not break rollback for the rest of the phases."""

    def test_undo_exception_logged_but_swallowed(self):
        log = []
        a = _RecordedPhase("a", log=log)
        b = _RecordedPhase(
            "b", raise_on_undo=RuntimeError("undo borked"), log=log,
        )
        c = _RecordedPhase(
            "c", severity=Severity.HARD_FAIL, raise_on_run=PhaseError("c boom"),
            log=log,
        )
        pipeline = Pipeline("prepare", [a, b, c])

        with self.assertRaises(PhaseError):
            pipeline.run(_FakeOrch())

        # Both undos should have been attempted even though b's raised.
        undos = [step for step in log if step[0] == "undo"]
        self.assertEqual(undos, [("undo", "b"), ("undo", "a")])


if __name__ == "__main__":
    unittest.main()
