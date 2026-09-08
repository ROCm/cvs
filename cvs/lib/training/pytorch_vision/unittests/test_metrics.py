import unittest

from cvs.lib.training.pytorch_vision.utils.metrics import METRICS, to_training_metrics


class TestTrainingMetrics(unittest.TestCase):
    def setUp(self):
        self.raw = {"metrics": {name: index + 1.5 for index, (name, _unit) in enumerate(METRICS)}}

    def test_namespaces_all_metrics(self):
        actual = to_training_metrics(self.raw)
        self.assertEqual(len(actual), len(METRICS))
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


if __name__ == "__main__":
    unittest.main()
