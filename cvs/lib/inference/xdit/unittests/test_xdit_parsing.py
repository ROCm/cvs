import unittest

from cvs.lib.inference.xdit.xdit_parsing import (
    XDIT_METRIC_UNITS,
    XDIT_RESULTS_COLUMNS,
    tier_metric_specs,
)


class TestXditReportMetrics(unittest.TestCase):
    def test_latency_tier_selects_available_timing_gate(self):
        thresholds = {
            "avg_pipe_time_s": {"kind": "max", "value": 3.0},
            "sample_count": {"kind": "min", "value": 1},
        }

        self.assertEqual(
            tier_metric_specs(thresholds, "latency"),
            {"avg_pipe_time_s": {"kind": "max", "value": 3.0}},
        )

    def test_report_contract_exposes_both_xdit_timing_metrics(self):
        metric_keys = {key for _, key in XDIT_RESULTS_COLUMNS if key}

        self.assertEqual(XDIT_METRIC_UNITS["avg_pipe_time_s"], "s")
        self.assertEqual(XDIT_METRIC_UNITS["clip_score"], "score")
        self.assertEqual(XDIT_METRIC_UNITS["mean_frame_clip_score"], "score")
        self.assertIn("avg_pipe_time_s", metric_keys)
        self.assertIn("avg_total_time_s", metric_keys)
        self.assertIn("clip_score", metric_keys)
        self.assertIn("mean_frame_clip_score", metric_keys)

    def test_quality_tier_selects_clip_score(self):
        thresholds = {
            "avg_pipe_time_s": {"kind": "max", "value": 3.0},
            "clip_score": {"kind": "min", "value": 20.0},
        }

        self.assertEqual(
            tier_metric_specs(thresholds, "quality"),
            {"clip_score": {"kind": "min", "value": 20.0}},
        )

    def test_quality_tier_selects_wan_frame_score(self):
        thresholds = {
            "avg_total_time_s": {"kind": "max", "value": 200.0},
            "mean_frame_clip_score": {"kind": "min", "value": 20.0},
        }

        self.assertEqual(
            tier_metric_specs(thresholds, "quality"),
            {"mean_frame_clip_score": {"kind": "min", "value": 20.0}},
        )


if __name__ == "__main__":
    unittest.main()
