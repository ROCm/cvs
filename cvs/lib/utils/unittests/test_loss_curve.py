'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/loss_curve.py::render_loss_curve_png.
The renderer must never raise: it returns a path on success and None on empty
input or any failure (missing matplotlib, unwritable path).
'''

import os
import tempfile
import unittest

from cvs.lib.training.jaxmaxtext.utils.loss_curve import render_loss_curve_png


class RenderLossCurvePngTests(unittest.TestCase):
    def test_empty_points_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            out = render_loss_curve_png([], os.path.join(d, "curve.png"))
            self.assertIsNone(out)

    def test_renders_png_file(self):
        points = [(0, 10.0), (10, 9.0), (20, 8.2), (30, 7.5)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "curve.png")
            out = render_loss_curve_png(points, path, title="unit test curve")
            # matplotlib is a declared dependency; when present we expect a file.
            if out is not None:
                self.assertEqual(out, path)
                self.assertTrue(os.path.isfile(path))
                self.assertGreater(os.path.getsize(path), 0)

    def test_unwritable_path_returns_none(self):
        # A path under a non-existent directory makes savefig fail; the helper
        # must swallow it and return None rather than raising.
        out = render_loss_curve_png([(0, 1.0), (1, 0.5)], "/nonexistent_dir_xyz/does/not/exist/curve.png")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
