'''Unit tests for the PyTorch Vision Run Deck adapter.'''

import unittest

from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.training_cells import build_training_cells, hide_training_scaling_efficiency
from cvs.lib.training.pytorch_vision.utils.rundeck_adapter import (
    flat_train_res_from_vision,
    vision_cell_dimensions,
)


SWEEP = "NNODES=1,MODEL=resnet50,DATA=rocAL-GPU,PRECISION=BF16,MBS=128,GA=1,RES=224"


def _results():
    return {
        SWEEP: {
            "rank0": {
                "training.images_per_sec_per_gpu": 1100.5,
                "training.loss_final": 0.8,
                "training.top1_accuracy_pct": 76.2,
                "training.eval_sample_count": None,
            }
        }
    }


class TestVisionCellDimensions(unittest.TestCase):
    def test_parses_sweep_name_tokens(self):
        dims = vision_cell_dimensions(None, SWEEP)
        self.assertEqual(
            dims,
            {
                "model": "resnet50",
                "precision": "BF16",
                "mbs": "128",
                "ga": "1",
                "res": "224",
                "data": "rocAL-GPU",
            },
        )

    def test_missing_tokens_are_empty(self):
        dims = vision_cell_dimensions(None, "default")
        self.assertEqual(dims["model"], "")
        self.assertEqual(dims["precision"], "")


class TestFlatTrainRes(unittest.TestCase):
    def test_first_host_metrics_and_loss_curve(self):
        flat = flat_train_res_from_vision(_results(), {SWEEP: [(1, 2.0), (2, float("nan")), (3, 1.5)]})
        combo = flat[SWEEP]
        self.assertEqual(combo["images_per_sec_per_gpu"], ["1100.5"])
        self.assertEqual(combo["loss_final"], ["0.8"])
        self.assertEqual(combo["top1_accuracy_pct"], ["76.2"])
        self.assertNotIn("eval_sample_count", combo)
        self.assertEqual(combo["_loss_curve"], [[1, 2.0], [3, 1.5]])

    def test_cells_use_vision_dimensions(self):
        profile = load_json_profile("pytorch_vision_single")
        config = resolve_report_config(profile)
        cells = build_training_cells(config, None, flat_train_res_from_vision(_results()), {})
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["model"], "resnet50")
        self.assertEqual(cells[0]["mbs"], "128")
        self.assertEqual(cells[0]["data"], "rocAL-GPU")
        self.assertIn("training.images_per_sec_per_gpu", cells[0]["actuals"])


class TestScalingVisibility(unittest.TestCase):
    def test_single_node_hides_scaling_efficiency(self):
        self.assertTrue(hide_training_scaling_efficiency(None, suite_stem="pytorch_vision_single"))
        self.assertFalse(hide_training_scaling_efficiency(None, suite_stem="pytorch_vision_distributed"))
