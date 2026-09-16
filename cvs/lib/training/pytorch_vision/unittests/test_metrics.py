import unittest

from cvs.lib.training.pytorch_vision.utils.metrics import (
    ARTIFACT_METRICS,
    accuracy_from_counts,
    codecarbon_tracking_active,
    convergence_point,
    gradient_accumulation_overhead_pct,
    loss_curve_decreased,
    parse_codecarbon_metrics,
    required_metrics_for_run,
    throughput_overhead_pct,
    to_training_metrics,
)


class TestTrainingMetrics(unittest.TestCase):
    def setUp(self):
        self.raw = {"metrics": {name: index + 1.5 for index, (name, _unit) in enumerate(ARTIFACT_METRICS)}}

    def test_namespaces_all_metrics(self):
        actual = to_training_metrics(self.raw)
        self.assertEqual(len(actual), len(ARTIFACT_METRICS))
        self.assertEqual(actual["training.images_per_sec"], 1.5)

    def test_rejects_missing_metric(self):
        self.raw["metrics"].pop("loss_final")
        with self.assertRaisesRegex(ValueError, "loss_final"):
            to_training_metrics(self.raw)

    def test_rejects_non_numeric_metric(self):
        self.raw["metrics"]["images_per_sec"] = "fast"
        with self.assertRaisesRegex(ValueError, "not numeric"):
            to_training_metrics(self.raw)

    def test_rejects_non_finite_metric(self):
        self.raw["metrics"]["images_per_sec"] = float("nan")
        with self.assertRaisesRegex(ValueError, "not finite"):
            to_training_metrics(self.raw)

    def test_rejects_missing_metrics_object(self):
        with self.assertRaisesRegex(ValueError, "metrics object"):
            to_training_metrics({})

    def test_rejects_incomplete_evaluation_evidence(self):
        self.raw["metrics"]["eval_completed"] = 0
        self.raw["metrics"]["eval_sample_count"] = 0
        with self.assertRaisesRegex(ValueError, "did not complete"):
            to_training_metrics(self.raw)

    def test_gradient_accumulation_overhead(self):
        self.assertAlmostEqual(gradient_accumulation_overhead_pct(80.0, 88.0), 10.0)

    def test_gradient_accumulation_overhead_rejects_bad_baseline(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            gradient_accumulation_overhead_pct(0.0, 88.0)

    def test_throughput_overhead_uses_time_per_image(self):
        self.assertAlmostEqual(throughput_overhead_pct(1000.0, 800.0), 25.0)
        self.assertAlmostEqual(throughput_overhead_pct(1000.0, 1250.0), -20.0)

    def test_throughput_overhead_rejects_bad_candidate(self):
        with self.assertRaisesRegex(ValueError, "candidate"):
            throughput_overhead_pct(1000.0, 0.0)

    def test_distributed_accuracy_uses_summed_counts(self):
        self.assertEqual(accuracy_from_counts(7, 9, 10), (70.0, 90.0))

    def test_distributed_accuracy_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "0 <= top1"):
            accuracy_from_counts(9, 8, 10)

    def test_loss_curve_and_convergence(self):
        self.assertTrue(loss_curve_decreased([5.0, 4.5, 4.0]))
        point = convergence_point(
            [
                {"step": 100, "time_seconds": 10, "top1_accuracy_pct": 60, "eval_loss": 2.0},
                {"step": 200, "time_seconds": 20, "top1_accuracy_pct": 70, "eval_loss": 1.5},
            ],
            target_top1_pct=65,
            target_eval_loss=1.6,
        )
        self.assertEqual(point, (200, 20.0))

    def test_required_metrics_are_mode_aware(self):
        perf = required_metrics_for_run("perf", "synthetic")
        training = required_metrics_for_run("train_5k", "rocal", codecarbon_enabled=True)
        self.assertNotIn("top1_accuracy_pct", perf)
        self.assertIn("top1_accuracy_pct", training)
        self.assertIn("data_loader_images_per_sec", training)
        self.assertIn("codecarbon_tracking_active", training)

    def test_codecarbon_detection_and_parsing(self):
        payload = {
            "version": "3.2.4",
            "tracker": "amdsmi",
            "gpu_count": 8,
            "active": True,
            "emissions_kg_co2eq": 0.25,
            "energy_kwh": 1.5,
        }
        self.assertTrue(codecarbon_tracking_active(payload, 8))
        self.assertEqual(
            parse_codecarbon_metrics(payload, 8)["training.codecarbon_tracking_active"],
            1.0,
        )
        self.assertEqual(parse_codecarbon_metrics(payload, 8)["training.energy_kwh"], 1.5)

    def test_codecarbon_never_maps_unavailable_to_zero(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            parse_codecarbon_metrics({"active": False, "reason": "no AMDSMI"}, 8)


if __name__ == "__main__":
    unittest.main()
