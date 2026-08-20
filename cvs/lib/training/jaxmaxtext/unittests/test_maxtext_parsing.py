'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the pure parsers in cvs/lib/training/jaxmaxtext/utils/maxtext_parsing.py:
step/eval extraction, aggregate metrics, convergence (row 33), validation loss
(row 34), and the loss-curve sampling + slope verdict (row 32).
'''

import unittest

from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import (
    compute_convergence,
    evaluate_loss_decreasing,
    extract_checkpoint_timings,
    extract_eval_metrics,
    parse_training_log,
    sample_loss_curve,
)


def _step_line(step, seconds, loss):
    return (
        f"I0804 08:14:00.000000 1 metric_logger.py:196] completed step: {step}, "
        f"seconds: {seconds}, TFLOP/s/device: 200.0, Tokens/s/device: 25000.0, "
        f"total_weights: 393216, loss: {loss}, lm_loss: {loss}, perplexity: 10.0"
    )


class ExtractCheckpointTimingsTests(unittest.TestCase):
    def test_empty_when_no_timing_lines(self):
        log = "\n".join(_step_line(i, 0.5, 9.0 - i) for i in range(3))
        self.assertEqual(extract_checkpoint_timings(log), {"save_seconds": None, "load_seconds": None})

    def test_parses_save_and_load_durations(self):
        log = (
            "Saved a checkpoint at step 5 in 12.5 seconds\n"
            "Restored checkpoint from step 5 took 3.2s\n"
        )
        out = extract_checkpoint_timings(log)
        self.assertAlmostEqual(out["save_seconds"], 12.5)
        self.assertAlmostEqual(out["load_seconds"], 3.2)

    def test_save_only(self):
        out = extract_checkpoint_timings("saving checkpoint duration: 7 s\n")
        self.assertAlmostEqual(out["save_seconds"], 7.0)
        self.assertIsNone(out["load_seconds"])

    def test_never_raises_on_garbage(self):
        self.assertEqual(
            extract_checkpoint_timings(None), {"save_seconds": None, "load_seconds": None}
        )

    def test_orbax_save_uses_last_finished_save(self):
        # Real orbax wording: event_tracking "[sync] Finished save in <X> seconds".
        # Two saves (cold step 0, warm step 5) -> report the LAST (steady-state).
        log = (
            "event_tracking.py:138] [process=0] [sync] Finished save in 11.09 seconds "
            "@ /LOGS/CKPT/checkpoints/0\n"
            "event_tracking.py:125] [process=0] [sync] Finished blocking save in 3.87 seconds. "
            "Continuing save @ /LOGS/CKPT/checkpoints/5.\n"
            "event_tracking.py:138] [process=0] [sync] Finished save in 3.89 seconds "
            "@ /LOGS/CKPT/checkpoints/5\n"
        )
        out = extract_checkpoint_timings(log)
        self.assertAlmostEqual(out["save_seconds"], 3.89)

    def test_orbax_load_uses_max_read_elapsed(self):
        # Real orbax wording on restore: "/jax/orbax/read ... (time elapsed: <X> s)".
        # Per-host lines -> report the MAX (slowest host bounds the restore).
        log = (
            "checkpointer.py:307] Restoring checkpoint from /LOGS/CKPT/checkpoints/5.\n"
            "jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested "
            "throughput: 7.257 GiB/s (total gbytes: 44.9 GiB) (time elapsed: 6.183607 s) (per-host)\n"
            "jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested "
            "throughput: 7.320 GiB/s (total gbytes: 44.9 GiB) (time elapsed: 6.130086 s) (per-host)\n"
        )
        out = extract_checkpoint_timings(log)
        self.assertAlmostEqual(out["load_seconds"], 6.183607)


class ExtractEvalMetricsTests(unittest.TestCase):
    def test_no_eval_lines_returns_empty(self):
        log = "\n".join(_step_line(i, 0.5, 9.0 - i) for i in range(3))
        self.assertEqual(extract_eval_metrics(log), [])

    def test_config_dump_lines_are_ignored(self):
        # Config-dump lines mention eval + loss but are not eval results.
        log = "\n".join(
            [
                "I0804 08:13:33.767627 1 pyconfig.py:465] Config param target_eval_loss: 0.0",
                "I0804 08:13:33.764653 1 pyconfig.py:465] Config param eval_interval: -1",
            ]
        )
        self.assertEqual(extract_eval_metrics(log), [])

    def test_parses_eval_loss_and_step(self):
        log = "\n".join(
            [
                _step_line(100, 0.5, 3.0),
                "I0804 08:20:00.0 1 metric_logger.py:210] eval metrics after step: 100, eval_loss: 2.5",
            ]
        )
        evals = extract_eval_metrics(log)
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["step"], 100)
        self.assertAlmostEqual(evals[0]["eval_loss"], 2.5)

    def test_parses_bare_loss_on_eval_line(self):
        log = "eval summary after step: 50, loss: 4.2"
        evals = extract_eval_metrics(log)
        self.assertEqual(len(evals), 1)
        self.assertAlmostEqual(evals[0]["eval_loss"], 4.2)


class ComputeConvergenceTests(unittest.TestCase):
    def setUp(self):
        # loss falls 10, 8, 6, 4, 2 over 5 steps of 1.0s each.
        self.steps = [{"step": i, "seconds": 1.0, "loss": 10.0 - 2 * i} for i in range(5)]
        self.evals = [
            {"step": 2, "eval_loss": 6.5},
            {"step": 4, "eval_loss": 3.5},
        ]

    def test_disabled_when_target_non_positive(self):
        self.assertEqual(compute_convergence(self.steps, self.evals, "auto", 0.0), (None, None))
        self.assertEqual(compute_convergence(self.steps, self.evals, "train_loss", -1.0), (None, None))

    def test_train_loss_target(self):
        # First step with loss <= 5.0 is step 3 (loss 4.0); cumulative time = 4.0s.
        steps_to_target, time_to_target = compute_convergence(self.steps, self.evals, "train_loss", 5.0)
        self.assertEqual(steps_to_target, 3)
        self.assertAlmostEqual(time_to_target, 4.0)

    def test_eval_loss_target(self):
        # First eval point with eval_loss <= 4.0 is step 4; cumulative time = 5.0s.
        steps_to_target, time_to_target = compute_convergence(self.steps, self.evals, "eval_loss", 4.0)
        self.assertEqual(steps_to_target, 4)
        self.assertAlmostEqual(time_to_target, 5.0)

    def test_auto_prefers_eval_when_present(self):
        steps_to_target, _ = compute_convergence(self.steps, self.evals, "auto", 4.0)
        self.assertEqual(steps_to_target, 4)  # eval step, not the train-loss step

    def test_auto_falls_back_to_train_loss_without_eval(self):
        steps_to_target, _ = compute_convergence(self.steps, [], "auto", 5.0)
        self.assertEqual(steps_to_target, 3)

    def test_target_never_reached(self):
        self.assertEqual(compute_convergence(self.steps, self.evals, "train_loss", 0.5), (None, None))

    def test_never_raises_on_empty(self):
        self.assertEqual(compute_convergence([], [], "auto", 1.0), (None, None))


class ParseTrainingLogEvalLossTests(unittest.TestCase):
    def test_eval_loss_none_without_eval(self):
        log = "\n".join(_step_line(i, 0.5, 9.0 - i) for i in range(3))
        res = parse_training_log(log, num_gpus=8)
        self.assertIn("training.eval_loss", res)
        self.assertIsNone(res["training.eval_loss"])

    def test_eval_loss_reports_last_eval_point(self):
        log = "\n".join(
            [
                _step_line(0, 0.5, 9.0),
                "eval metrics after step: 0, eval_loss: 8.0",
                _step_line(1, 0.5, 8.5),
                "eval metrics after step: 1, eval_loss: 7.0",
            ]
        )
        res = parse_training_log(log, num_gpus=8)
        self.assertAlmostEqual(res["training.eval_loss"], 7.0)

    def test_empty_log_has_eval_loss_key(self):
        res = parse_training_log("", num_gpus=8)
        self.assertIn("training.eval_loss", res)
        self.assertIsNone(res["training.eval_loss"])


class SampleLossCurveTests(unittest.TestCase):
    def _steps(self, n):
        return [{"step": i, "seconds": 1.0, "loss": 10.0 - i * 0.1} for i in range(n)]

    def test_samples_every_n_plus_first_and_last(self):
        pts = sample_loss_curve(self._steps(25), sample_every=10, milestone_steps=[])
        steps = [s for s, _ in pts]
        # multiples of 10 (0,10,20) plus first (0) and last (24)
        self.assertEqual(steps, [0, 10, 20, 24])

    def test_includes_milestones(self):
        steps_data = [{"step": i, "seconds": 1.0, "loss": 5.0} for i in (0, 3, 7, 12, 50)]
        pts = sample_loss_curve(steps_data, sample_every=1000, milestone_steps=[7, 12])
        steps = [s for s, _ in pts]
        # first(0), last(50) always; milestones 7 and 12 included; 3 excluded
        self.assertEqual(steps, [0, 7, 12, 50])

    def test_deduped_and_ordered(self):
        pts = sample_loss_curve(self._steps(11), sample_every=5, milestone_steps=[0, 10])
        steps = [s for s, _ in pts]
        self.assertEqual(steps, sorted(set(steps)))
        self.assertEqual(steps, [0, 5, 10])

    def test_ignores_steps_without_loss(self):
        data = [{"step": 0, "seconds": 1.0}, {"step": 1, "seconds": 1.0, "loss": 3.0}]
        pts = sample_loss_curve(data, sample_every=1, milestone_steps=[])
        self.assertEqual(pts, [(1, 3.0)])

    def test_empty_input(self):
        self.assertEqual(sample_loss_curve([], 10, [100]), [])


class EvaluateLossDecreasingTests(unittest.TestCase):
    def test_decreasing(self):
        pts = [(i, 10.0 - i) for i in range(6)]
        decreasing, slope, _detail = evaluate_loss_decreasing(pts, max_slope=0.0)
        self.assertTrue(decreasing)
        self.assertAlmostEqual(slope, -1.0)

    def test_increasing(self):
        pts = [(i, 1.0 + i) for i in range(6)]
        decreasing, slope, _detail = evaluate_loss_decreasing(pts, max_slope=0.0)
        self.assertFalse(decreasing)
        self.assertGreater(slope, 0.0)

    def test_flat_is_not_decreasing_at_zero_tolerance(self):
        pts = [(i, 5.0) for i in range(6)]
        decreasing, slope, _detail = evaluate_loss_decreasing(pts, max_slope=0.0)
        self.assertFalse(decreasing)
        self.assertAlmostEqual(slope, 0.0)

    def test_too_few_points_returns_none(self):
        self.assertIsNone(evaluate_loss_decreasing([(0, 5.0)], 0.0))
        self.assertIsNone(evaluate_loss_decreasing([], 0.0))

    def test_degenerate_x_spread_returns_none(self):
        # all steps identical -> zero denominator -> None (no crash)
        self.assertIsNone(evaluate_loss_decreasing([(3, 5.0), (3, 4.0)], 0.0))

    def test_noisy_but_downward(self):
        pts = [(0, 10.0), (10, 9.5), (20, 9.8), (30, 8.0), (40, 7.9), (50, 6.5)]
        decreasing, slope, _detail = evaluate_loss_decreasing(pts, max_slope=0.0)
        self.assertTrue(decreasing)
        self.assertLess(slope, 0.0)


if __name__ == "__main__":
    unittest.main()
