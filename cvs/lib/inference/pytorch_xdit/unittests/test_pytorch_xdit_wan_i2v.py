"""Unit tests for pytorch_xdit_wan_i2v parser."""

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_wan_i2v import WanI2vOutputParser


class TestWanI2vOutputParser(unittest.TestCase):
    def test_parse_timing_and_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()
            timing = results_dir / "timing.json"
            timing.write_text(json.dumps([{"pipe_time": 120.0}, {"pipe_time": 130.0}]), encoding="utf-8")
            (results_dir / "video_i2v.mp4").write_bytes(b"fake")

            parser = WanI2vOutputParser(tmpdir)
            result, errors = parser.parse()

            self.assertEqual(errors, [])
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.avg_pipe_time_s, 125.0)
            self.assertEqual(result.repetition_count, 2)
            self.assertTrue(result.video_path.endswith("video_i2v.mp4"))

    def test_validate_threshold_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()
            (results_dir / "timing.json").write_text(json.dumps([{"pipe_time": 10.0}]), encoding="utf-8")
            (results_dir / "video_i2v.mp4").write_bytes(b"fake")

            parser = WanI2vOutputParser(tmpdir)
            result, _ = parser.parse()
            passed, message = parser.validate_threshold(
                result,
                {"auto": {"max_avg_pipe_time_s": 20.0}},
                gpu_type="auto",
            )
            self.assertTrue(passed)
            self.assertIn("PASS", message)

    def test_missing_timing_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = WanI2vOutputParser(tmpdir)
            result, errors = parser.parse()
            self.assertIsNone(result)
            self.assertTrue(any("timing.json" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
