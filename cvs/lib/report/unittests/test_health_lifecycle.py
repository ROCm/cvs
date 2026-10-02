'''Unit tests for health Run Deck stage timings.'''

import unittest

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
