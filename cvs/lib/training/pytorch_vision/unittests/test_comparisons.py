import unittest
from types import SimpleNamespace

from cvs.tests.training.pytorch_vision.pytorch_vision_training import (
    _refresh_pipeline_overheads,
)


def _sweep(name, device, augmentation):
    return SimpleNamespace(
        name=name,
        model="resnet50",
        backend="torchvision",
        precision="BF16",
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


if __name__ == "__main__":
    unittest.main()
