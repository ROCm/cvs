import unittest
from types import SimpleNamespace

from cvs.lib.report.inference_html import render_report_html
from cvs.lib.report.inference_payload import build_inference_report_payload
from cvs.lib.report.presets.pytorch_vision_training import (
    PYTORCH_VISION_TRAINING_REPORT_CONFIG,
)
from cvs.lib.training.pytorch_vision.utils.metrics import METRICS


class TestPyTorchVisionRunDeck(unittest.TestCase):
    def setUp(self):
        self.cell_id = "NNODES=1,STEPS=50,MODEL=resnet50,PRECISION=BF16,MBS=256,GA=1,GBS=2048,RES=224"
        sweep = SimpleNamespace(
            name=self.cell_id,
            label="W1-BF16-R224-MBS256-GA1",
            model="resnet50",
            precision="BF16",
            image_size=224,
            batch_size=256,
            gradient_accumulation_steps=1,
            training_flops_per_image=24600000000,
            data_mode="synthetic",
        )
        thresholds = {f"training.{name}": {"kind": "info", "value": 0} for name, _unit in METRICS}
        thresholds["training.images_per_sec"] = {"kind": "min", "value": 24500}
        thresholds["training.images_per_sec_per_gpu"] = {"kind": "min", "value": 3062.5}
        thresholds["training.step_time_ms_p95"] = {"kind": "max_ms", "value": 88}
        thresholds["training.peak_memory_allocated_mb"] = {"kind": "max", "value": 13000}
        thresholds["training.peak_memory_reserved_mb"] = {"kind": "max", "value": 15000}
        self.variant = SimpleNamespace(
            sweep=lambda _ref: sweep,
            training=SimpleNamespace(
                enabled_sweeps=lambda: [sweep],
                gpus_per_node=8,
                peak_tflops_per_gpu=1307.4,
            ),
            gpu_arch="MI325X",
            container=SimpleNamespace(image="rocm/pytorch:test@sha256:abc"),
            enforce_thresholds=True,
            thresholds={self.cell_id: thresholds},
            cell_key=lambda combo_key, image_size, batch_size: self.cell_id,
        )
        actuals = {f"training.{name}": index + 1.0 for index, (name, _unit) in enumerate(METRICS)}
        actuals.update(
            {
                "training.images_per_sec": 25800.0,
                "training.images_per_sec_per_gpu": 3225.0,
                "training.step_time_ms_p95": 80.0,
                "training.peak_memory_allocated_mb": 12019.0,
                "training.peak_memory_reserved_mb": 14112.0,
            }
        )
        key = (
            "resnet50",
            "MI325X",
            "W1-BF16-R224",
            224,
            "GA1",
            256,
        )
        self.payload = build_inference_report_payload(
            config=PYTORCH_VISION_TRAINING_REPORT_CONFIG,
            variant_config=self.variant,
            inf_res_dict={key: {"node0": actuals}},
            lifecycle_report={
                "test_training[W1-BF16-R224-MBS256-GA1]": [("training", 30.0, "s")],
                "test_training[OTHER-BF16-R224-MBS256-GA1]": [("training", 99.0, "s")],
            },
        )

    def test_payload_uses_training_labels_and_gates(self):
        self.assertEqual(self.payload["overall_status"], "pass")
        self.assertEqual(
            self.payload["report"]["shape_axis_labels"],
            ("Workload", "Resolution"),
        )
        self.assertEqual(self.payload["report"]["sweep_axis_label"], "MBS/GPU")
        self.assertEqual(self.payload["cells"][0]["cell_id"], self.cell_id)
        self.assertEqual(self.payload["cells"][0]["cell_lifecycle"]["training"], 30.0)

    def test_html_uses_training_vocabulary(self):
        document = render_report_html(self.payload)
        self.assertIn("PyTorch Vision W1 Run Deck", document)
        self.assertIn("Workload=W1-BF16-R224", document)
        self.assertIn("MBS/GPU=256", document)
        self.assertIn("images/s", document)
        self.assertNotIn("ISL=", document)
        self.assertNotIn("TTFT", document)
        self.assertNotIn("tok/s", document)


if __name__ == "__main__":
    unittest.main()
