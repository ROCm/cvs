'''Unit tests for health Run Deck stage timings.'''

import unittest

import pytest

from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage


class TestHealthLifecycle(unittest.TestCase):
    def test_record_appends_seconds_for_the_timeline(self):
        lifecycle = HealthLifecycle()
        lifecycle.record("gpu_enumeration", 1.25)
        lifecycle.record("level_config", 40)
        self.assertEqual(
            lifecycle.report,
            {"health": [("gpu_enumeration", 1.25, "s"), ("level_config", 40.0, "s")]},
        )

    def test_record_ignores_non_numeric_and_negative(self):
        lifecycle = HealthLifecycle()
        lifecycle.record("gpu_enumeration", "nope")
        lifecycle.record("level_config", -1)
        self.assertEqual(lifecycle.report, {})

    def test_timed_stage_records_elapsed_and_reraises(self):
        lifecycle = HealthLifecycle()
        with timed_stage(lifecycle, "a2a"):
            pass
        self.assertEqual(lifecycle.report["health"][0][0], "a2a")
        self.assertEqual(lifecycle.report["health"][0][2], "s")
        self.assertGreaterEqual(lifecycle.report["health"][0][1], 0)

        with self.assertRaises(RuntimeError):
            with timed_stage(lifecycle, "p2p"):
                raise RuntimeError("preset failed")
        self.assertEqual(lifecycle.report["health"][1][0], "p2p")

    def test_timed_stage_without_lifecycle_is_a_no_op(self):
        with timed_stage(None, "a2a"):
            pass

    def test_timed_stage_propagates_pytest_fail_without_lifecycle(self):
        with self.assertRaises(pytest.fail.Exception):
            with timed_stage(None, "a2a"):
                pytest.fail("boom")

    def test_timed_stage_propagates_pytest_skip_without_lifecycle(self):
        with self.assertRaises(pytest.skip.Exception):
            with timed_stage(None, "a2a"):
                pytest.skip("off")

    def test_timed_stage_propagates_outcomes_when_record_raises(self):
        class RecordRaises:
            def __init__(self, error):
                self.error = error

            def record(self, label, seconds):
                raise self.error

        lifecycles = [object(), *(RecordRaises(error()) for error in (AttributeError, TypeError, ValueError))]
        for lifecycle in lifecycles:
            for outcome, expected in ((pytest.fail, pytest.fail.Exception), (pytest.skip, pytest.skip.Exception)):
                with self.subTest(lifecycle=lifecycle, outcome=outcome):
                    with self.assertRaises(expected):
                        with timed_stage(lifecycle, "a2a"):
                            outcome("verdict")

    def test_timed_stage_swallows_record_failure_on_success(self):
        class RecordRaises:
            def record(self, label, seconds):
                raise ValueError("reporting failed")

        with timed_stage(RecordRaises(), "a2a"):
            pass

    def test_timed_stage_records_elapsed_when_block_fails(self):
        lifecycle = HealthLifecycle()
        with self.assertRaises(pytest.fail.Exception):
            with timed_stage(lifecycle, "a2a"):
                pytest.fail("boom")

        self.assertEqual(lifecycle.report["health"][0][0], "a2a")
        self.assertGreaterEqual(lifecycle.report["health"][0][1], 0)

    def test_timed_stage_preserves_block_return_value(self):
        def returns_from_block():
            with timed_stage(None, "a2a"):
                return 42

        def raises_from_block():
            with timed_stage(None, "a2a"):
                raise RuntimeError("boom")

        self.assertEqual(returns_from_block(), 42)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            raises_from_block()
