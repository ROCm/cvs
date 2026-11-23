import unittest
from types import SimpleNamespace

from cvs.tests.training.pytorch_vision._common import (
    _refresh_pipeline_overheads,
    _refresh_precision_parity,
)


def _sweep(name, device, augmentation, precision="BF16"):
    return SimpleNamespace(
        name=name,
        model="resnet50",
        backend="torchvision",
        precision=precision,
        image_size=224,
        batch_size=128,
        gradient_accumulation_steps=1,
        data_mode="rocal",
        rocal_device=device,
        augmentation=augmentation,
    )


class TestPipelineComparisons(unittest.TestCase):
    def test_derives_cpu_and_augmentation_time_per_image_overhead(self):
        gpu = _sweep("gpu-standard", "gpu", "standard")
        cpu = _sweep("cpu-standard", "cpu", "standard")
        heavy = _sweep("gpu-heavy", "gpu", "heavy")
        variant = SimpleNamespace(training=SimpleNamespace(enabled_sweeps=lambda: [gpu, cpu, heavy]))
        results = {
            gpu.name: {"host": {"training.data_loader_images_per_sec": 1000.0}},
            cpu.name: {"host": {"training.data_loader_images_per_sec": 500.0}},
            heavy.name: {"host": {"training.data_loader_images_per_sec": 800.0}},
        }

        _refresh_pipeline_overheads(variant, results)

        self.assertEqual(
            results[gpu.name]["host"]["training.augmentation_overhead_pct"],
            0.0,
        )
        self.assertEqual(
            results[gpu.name]["host"]["training.rocal_cpu_overhead_pct"],
            0.0,
        )
        self.assertEqual(
            results[cpu.name]["host"]["training.rocal_cpu_overhead_pct"],
            100.0,
        )
        self.assertEqual(
            results[heavy.name]["host"]["training.augmentation_overhead_pct"],
            25.0,
        )


class TestPrecisionParity(unittest.TestCase):
    def test_compares_reduced_precision_against_matching_fp32(self):
        fp32 = _sweep("fp32", "gpu", "standard", precision="FP32")
        bf16 = _sweep("bf16", "gpu", "standard")
        other = _sweep("bf16-cpu", "cpu", "standard")
        variant = SimpleNamespace(training=SimpleNamespace(enabled_sweeps=lambda: [fp32, bf16, other]))
        results = {name: {"host": {}} for name in ("fp32", "bf16", "bf16-cpu")}
        losses = {"fp32": [(1, 4.0), (2, 2.0)], "bf16": [(1, 4.4), (2, 2.0)], "bf16-cpu": [(1, 1.0)]}
        norms = {"fp32": [(1, 1.0), (2, 2.0)], "bf16": [(1, 1.2), (2, 2.0)]}

        _refresh_precision_parity(variant, results, losses, norms)

        self.assertAlmostEqual(results["bf16"]["host"]["training.precision_parity_loss_delta_pct"], 5.0)
        self.assertAlmostEqual(results["bf16"]["host"]["training.precision_parity_grad_norm_delta_pct"], 10.0)
        self.assertEqual(results["fp32"]["host"], {})
        self.assertEqual(results["bf16-cpu"]["host"], {})

    def test_waits_for_the_reference_run(self):
        fp32 = _sweep("fp32", "gpu", "standard", precision="FP32")
        bf16 = _sweep("bf16", "gpu", "standard")
        variant = SimpleNamespace(training=SimpleNamespace(enabled_sweeps=lambda: [fp32, bf16]))
        results = {"bf16": {"host": {}}}

        _refresh_precision_parity(variant, results, {"bf16": [(1, 1.0)]}, {})

        self.assertEqual(results["bf16"]["host"], {})


if __name__ == "__main__":
    unittest.main()
