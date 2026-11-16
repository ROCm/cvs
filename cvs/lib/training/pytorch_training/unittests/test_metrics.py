import math
import unittest

from cvs.lib.training.pytorch_training.utils import metrics


def _artifact(**overrides):
    values = {name: 1.0 for name, _unit in metrics.ARTIFACT_METRICS}
    values.update(overrides)
    return {"metrics": values}


class TestMetricContract(unittest.TestCase):
    def test_names_are_unique(self):
        names = [name for name, _unit in metrics.METRICS]
        self.assertEqual(len(names), len(set(names)))

    def test_gated_metrics_are_artifact_metrics(self):
        artifact = {name for name, _unit in metrics.ARTIFACT_METRICS}
        self.assertLessEqual(metrics.GATED_METRICS, artifact)

    def test_benchmark_metrics_are_declared(self):
        declared = {name for name, _unit in metrics.METRICS}
        self.assertLessEqual(set(metrics.BENCHMARK_TASKS), declared)

    def test_to_training_metrics_namespaces_values(self):
        out = metrics.to_training_metrics(_artifact(eval_loss=2.5))
        self.assertEqual(out["training.tokens_per_sec"], 1.0)
        self.assertEqual(out["training.eval_loss"], 2.5)

    def test_to_training_metrics_rejects_missing(self):
        raw = _artifact()
        del raw["metrics"]["tokens_per_sec"]
        with self.assertRaisesRegex(ValueError, "missing metrics"):
            metrics.to_training_metrics(raw)

    def test_to_training_metrics_rejects_non_finite(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            metrics.to_training_metrics(_artifact(loss_final=math.nan))

    def test_to_training_metrics_rejects_incomplete_eval(self):
        with self.assertRaisesRegex(ValueError, "did not complete"):
            metrics.to_training_metrics(_artifact(eval_completed=0.0))

    def test_tier_specs_drop_info(self):
        cell = {
            "training.tokens_per_sec": {"kind": "min", "value": 10},
            "training.mfu_pct": {"kind": "info"},
        }
        self.assertEqual(list(metrics.tier_metric_specs(cell, "throughput")), ["training.tokens_per_sec"])


class TestDerivedMetrics(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(metrics.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(metrics.percentile(range(1, 101), 95), 95.05)

    def test_ga_overhead_sign(self):
        self.assertAlmostEqual(metrics.gradient_accumulation_overhead_pct(100.0, 80.0), 25.0)
        self.assertLess(metrics.gradient_accumulation_overhead_pct(100.0, 110.0), 0)

    def test_ga_overhead_rejects_zero(self):
        with self.assertRaises(ValueError):
            metrics.gradient_accumulation_overhead_pct(0.0, 1.0)

    def test_scaling_efficiency(self):
        self.assertAlmostEqual(metrics.compute_scaling_efficiency(360.0, 4, 100.0, 1), 90.0)
        self.assertIsNone(metrics.compute_scaling_efficiency(360.0, 4, 0.0, 1))

    def test_perplexity(self):
        self.assertAlmostEqual(metrics.perplexity(math.log(20.0)), 20.0)
        self.assertTrue(math.isfinite(metrics.perplexity(5000.0)))

    def test_loss_curve_decreased(self):
        self.assertTrue(metrics.loss_curve_decreased([3.0, 2.5, 2.6, 2.0]))
        self.assertFalse(metrics.loss_curve_decreased([2.0, 2.5]))
        self.assertFalse(metrics.loss_curve_decreased([2.0]))

    def test_convergence_point_takes_later_target(self):
        series = [{"step": s, "loss": 10.0 - s, "time_seconds": float(s)} for s in range(1, 9)]
        evals = [{"step": 6, "eval_loss": 3.0, "time_seconds": 6.5}]
        self.assertEqual(metrics.convergence_point(series, target_train_loss=7.0), (3, 3.0))
        self.assertEqual(
            metrics.convergence_point(series, target_train_loss=7.0, evaluations=evals, target_eval_loss=3.5),
            (6, 6.5),
        )
        self.assertIsNone(metrics.convergence_point(series, target_train_loss=0.5))

    def test_milestone_losses(self):
        series = [{"step": 100, "loss": 4.0}, {"step": 200, "loss": 3.0}]
        self.assertEqual(metrics.milestone_losses(series, [100, 500]), {"loss_step_100": 4.0})

    def test_benchmark_scores_convert_to_percent(self):
        results = {"leaderboard_mmlu_pro": {"acc,none": 0.25}, "gsm8k": {"exact_match,strict-match": 0.5}}
        self.assertEqual(metrics.benchmark_scores(results), {"mmlu_pro_pct": 25.0, "gsm8k_pct": 50.0})

    def test_benchmark_scores_reject_out_of_range(self):
        with self.assertRaises(ValueError):
            metrics.benchmark_scores({"hellaswag": {"acc_norm,none": 1.5}})

    def test_codecarbon_requires_active_tracker(self):
        payload = {"version": "3.2.4", "tracker": "amdsmi", "gpu_count": 8, "active": True, "energy_kwh": 0.5}
        self.assertEqual(metrics.parse_codecarbon_metrics(payload, 8)["training.energy_kwh"], 0.5)
        with self.assertRaises(ValueError):
            metrics.parse_codecarbon_metrics({**payload, "active": False}, 8)

    def test_required_metrics_by_mode(self):
        perf = metrics.required_metrics_for_run("perf")
        self.assertNotIn("eval_loss", perf)
        train = metrics.required_metrics_for_run("train_5k", codecarbon_enabled=True, benchmarks=["gsm8k_pct"])
        self.assertTrue({"eval_loss", "loss_step_5000", "tokens_per_kwh", "gsm8k_pct"} <= train)


if __name__ == "__main__":
    unittest.main()
