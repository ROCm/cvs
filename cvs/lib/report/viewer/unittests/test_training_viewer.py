'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/report/viewer/training_viewer.py.
'''

import tempfile
import unittest
from pathlib import Path

from cvs.lib.report.viewer.training_viewer import training_viewer_basename_for, write_training_viewer

_SWEEP = "BS=3,PRECISION=BF16,SL=8192"


def _payload():
    return {
        "datasets": {
            "training_sweep": {
                "training_series": {_SWEEP: {"learning/loss": [[0, 12.0], [1, 11.0]]}},
                "metric_bars": [
                    {
                        "metric": "training.tokens_per_sec_per_gpu",
                        "label": "tokens_per_sec_per_gpu",
                        "unit": "tok/s/GPU",
                        "values": {_SWEEP: 200.0},
                    }
                ],
            }
        }
    }


class WriteTrainingViewerTests(unittest.TestCase):
    def test_writes_self_contained_viewer(self):
        with tempfile.TemporaryDirectory() as d:
            out = write_training_viewer(
                Path(d) / "jaxmaxtext_run_deck_viewer.html",
                title="JAX MaxText Run Deck",
                subtitle="viewer",
                deck_basename="jaxmaxtext_run_deck",
                embed_payload=_payload(),
            )
            html = out.read_text(encoding="utf-8")
            self.assertIn('id="embedded-report-json"', html)  # embedded JSON for offline use
            self.assertIn("learning/loss", html)  # raw series present in the payload
            self.assertIn("tokens_per_sec_per_gpu", html)  # metric bars present
            self.assertIn("chart.js", html)  # dynamic Chart.js rendering
            self.assertIn("jaxmaxtext_run_deck.html", html)  # back-link to the deck

    def test_no_embed_still_writes_template(self):
        with tempfile.TemporaryDirectory() as d:
            out = write_training_viewer(
                Path(d) / "v.html", title="T", subtitle="", deck_basename="deck", embed_payload=None
            )
            self.assertTrue(out.is_file())

    def test_basename_helper(self):
        self.assertEqual(training_viewer_basename_for("jaxmaxtext_run_deck"), "jaxmaxtext_run_deck_viewer.html")


if __name__ == "__main__":
    unittest.main()
