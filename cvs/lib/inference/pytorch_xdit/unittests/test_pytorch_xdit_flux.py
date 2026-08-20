import unittest
from unittest.mock import patch

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux import log_results_summary


class TestLogResultsSummary(unittest.TestCase):
    @patch("cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux.log")
    def test_skips_single_entry(self, mock_log):
        log_results_summary([{"label": "node0", "avg_pipe_time_s": 1.0, "passed": True}])
        mock_log.info.assert_not_called()

    @patch("cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux.log")
    def test_logs_multi_node_summary(self, mock_log):
        results_summary = [
            {"label": "tus1-p3-g40", "avg_pipe_time_s": 1.18, "passed": True},
            {"label": "tus1-p3-g41", "avg_pipe_time_s": 1.14, "passed": True},
        ]

        log_results_summary(results_summary, metric_key="avg_pipe_time_s")

        rendered = []
        for call in mock_log.info.call_args_list:
            args = call.args
            if len(args) == 1:
                rendered.append(str(args[0]))
            elif len(args) >= 2:
                rendered.append(str(args[0]) % args[1:])

        joined = "\n".join(rendered)
        self.assertIn("Multi-node results summary:", joined)
        self.assertIn("tus1-p3-g40: 1.18s [PASS]", joined)
        self.assertIn("tus1-p3-g41: 1.14s [PASS]", joined)
        self.assertIn("Overall average: 1.16s", joined)
        self.assertIn("Nodes passed: 2/2", joined)

    @patch("cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux.log")
    def test_custom_metric_key_and_title(self, mock_log):
        results_summary = [
            {"label": "node-a", "avg_total_time_s": 2.0, "passed": False},
            {"label": "node-b", "avg_total_time_s": 4.0, "passed": True},
        ]

        log_results_summary(
            results_summary,
            metric_key="avg_total_time_s",
            title="Distributed results summary",
        )

        rendered = []
        for call in mock_log.info.call_args_list:
            args = call.args
            if len(args) == 1:
                rendered.append(str(args[0]))
            elif len(args) >= 2:
                rendered.append(str(args[0]) % args[1:])

        joined = "\n".join(rendered)
        self.assertIn("Distributed results summary:", joined)
        self.assertIn("Nodes passed: 1/2", joined)


if __name__ == "__main__":
    unittest.main()
