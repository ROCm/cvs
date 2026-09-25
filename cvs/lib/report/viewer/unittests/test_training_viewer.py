'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/report/viewer/training_viewer.py.
'''

import os
import tempfile
import unittest

from cvs.lib.report.viewer.training_viewer import training_viewer_basename_for, write_training_viewer


def _payload():
    return {
        "datasets": {
            "training_sweep": {
                "training_series": {"NN4_BF16": {"learning/loss": [[0, 12.0], [5, 10.0]]}},
                "viewer_settings": {"default_series": ["learning/loss"], "skip_initial_steps": 5},
                "metric_bars": [],
            }
        }
    }


class WriteTrainingViewerTests(unittest.TestCase):
    def test_basename(self):
        self.assertEqual(training_viewer_basename_for("jaxmaxtext_run_deck"), "jaxmaxtext_run_deck_viewer.html")

    def test_writes_self_contained_viewer(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "deck_viewer.html")
            write_training_viewer(
                out, title="JAX MaxText", subtitle="single-node", deck_basename="deck", embed_payload=_payload()
            )
            html = open(out, encoding="utf-8").read()
        # placeholders substituted
        self.assertIn("JAX MaxText", html)
        self.assertNotIn("__TITLE__", html)
        self.assertIn('href="deck.html"', html)  # back-to-deck link
        # dynamic charting deps + embedded data (no PNGs)
        self.assertIn("chart.js", html)
        self.assertIn("chartjs-plugin-zoom", html)
        self.assertIn("embedded-report-json", html)
        self.assertIn("learning/loss", html)
        self.assertIn("skip_initial_steps", html)

    def test_missing_payload_still_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "deck_viewer.html")
            write_training_viewer(out, title="T", deck_basename="deck")
            self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()
