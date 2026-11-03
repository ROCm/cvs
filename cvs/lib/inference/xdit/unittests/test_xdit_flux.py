import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.lib.inference.xdit.xdit_flux import FluxOutputParser, log_results_summary


class TestLogResultsSummary(unittest.TestCase):
    @patch("cvs.lib.inference.xdit.xdit_flux.log")
    def test_skips_single_entry(self, mock_log):
        log_results_summary([{"label": "node0", "avg_pipe_time_s": 1.0, "passed": True}])
        mock_log.info.assert_not_called()

    @patch("cvs.lib.inference.xdit.xdit_flux.log")
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

    @patch("cvs.lib.inference.xdit.xdit_flux.log")
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


def _write_flux_output(root, pipe_times, with_image=True):
    results = Path(root) / "results"
    results.mkdir()
    (results / "timing.json").write_text(
        json.dumps([{"pipe_time": value} for value in pipe_times]),
        encoding="utf-8",
    )
    if with_image:
        (results / "flux_0.png").write_bytes(b"png")


class TestFluxOutputParserRequirements(unittest.TestCase):
    def test_requires_configured_repetitions_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_flux_output(tmp, [1.0] * 25)
            result, errors = FluxOutputParser(tmp, expected_repetitions=25).parse()
        self.assertEqual(errors, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.repetition_count, 25)

    def test_repetition_mismatch_is_not_a_threshold_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_flux_output(tmp, [0.5])
            result, errors = FluxOutputParser(tmp, expected_repetitions=25).parse()
        self.assertIsNone(result)
        self.assertTrue(any("has 1 repetitions, expected 25" in err for err in errors))
        self.assertFalse(any("threshold" in err.lower() for err in errors))

    def test_missing_image_fails_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_flux_output(tmp, [1.0, 1.2], with_image=False)
            result, errors = FluxOutputParser(tmp, expected_repetitions=2).parse()
        self.assertIsNone(result)
        self.assertTrue(any("No images matching" in err for err in errors))

    def test_unset_repetition_count_still_requires_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_flux_output(tmp, [1.0], with_image=False)
            result, errors = FluxOutputParser(tmp).parse()
        self.assertIsNone(result)
        self.assertTrue(any("No images matching" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
