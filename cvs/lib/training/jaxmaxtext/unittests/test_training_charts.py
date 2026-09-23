'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/training_charts.py.
'''

import os
import tempfile
import unittest

from cvs.lib.training.jaxmaxtext.utils import training_charts as tc


def _scalars():
    steps = list(range(10))
    return {
        "learning/loss": [(s, 12.0 - s * 0.3) for s in steps],
        "learning/lm_loss": [(s, 12.0 - s * 0.28) for s in steps],
        "learning/moe_lb_loss": [(s, 0.001) for s in steps],
        "learning/grad_norm": [(s, 1.5 - s * 0.05) for s in steps],
        "learning/raw_grad_norm": [(s, 1.6 - s * 0.05) for s in steps],
        "learning/param_norm": [(s, 100.0 + s) for s in steps],
        "learning/current_learning_rate": [(s, 0.001 * (s + 1) / 10) for s in steps],
        "perf/step_time_seconds": [(0, 110.0), (1, 90.0)] + [(s, 9.8 + (s % 3) * 0.1) for s in steps[2:]],
        "perf/per_device_tflops_per_sec": [(s, 650.0) for s in steps],
    }


class ChartRenderTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.d = self._dir.name

    def tearDown(self):
        self._dir.cleanup()

    def _p(self, name):
        return os.path.join(self.d, name)

    def test_scalar_charts_one_per_tag(self):
        out = tc.render_scalar_charts(_scalars(), self.d)
        # one (tag, path) per non-empty tag; files exist; sorted by tag
        tags = [t for t, _p in out]
        self.assertEqual(tags, sorted(tags))
        self.assertIn("learning/loss", tags)
        self.assertIn("perf/step_time_seconds", tags)
        self.assertEqual(len(out), len(_scalars()))
        for _tag, path in out:
            self.assertTrue(os.path.isfile(path))

    def test_scalar_charts_empty_returns_empty_list(self):
        self.assertEqual(tc.render_scalar_charts({}, self.d), [])

    def test_step_time(self):
        out = tc.render_step_time_png(_scalars(), self._p("st.png"))
        self.assertTrue(out and os.path.isfile(out))

    def test_step_time_falls_back_to_step_metrics(self):
        step_metrics = [{"step": i, "step_time_seconds": 9.8} for i in range(5)]
        out = tc.render_step_time_png({}, self._p("st2.png"), step_metrics=step_metrics)
        self.assertTrue(out and os.path.isfile(out))

    def test_mfu_with_peak(self):
        out = tc.render_mfu_png(_scalars(), self._p("mfu.png"), 1307.4)
        self.assertTrue(out and os.path.isfile(out))

    def test_mfu_without_peak_is_none(self):
        self.assertIsNone(tc.render_mfu_png(_scalars(), self._p("mfu2.png"), None))

    def test_empty_scalars_return_none(self):
        self.assertIsNone(tc.render_step_time_png({}, self._p("x.png")))
        self.assertIsNone(tc.render_mfu_png({}, self._p("x.png"), 1307.4))

    def test_cross_sweep_loss_needs_two_sweeps(self):
        one = {"BF16": [(0, 12.0), (1, 11.0)]}
        two = {"BF16": [(0, 12.0), (1, 11.0)], "FP8": [(0, 12.1), (1, 11.2)]}
        self.assertIsNone(tc.render_cross_sweep_loss_png(one, self._p("c1.png")))
        out = tc.render_cross_sweep_loss_png(two, self._p("c2.png"))
        self.assertTrue(out and os.path.isfile(out))

    def test_cross_sweep_bar_needs_two_sweeps(self):
        self.assertIsNone(tc.render_cross_sweep_bar_png({"BF16": 200.0}, self._p("b1.png"), "tok/s/GPU"))
        out = tc.render_cross_sweep_bar_png({"BF16": 200.0, "FP8": 260.0}, self._p("b2.png"), "tok/s/GPU")
        self.assertTrue(out and os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()
