'''Unit tests for Megatron gpus_per_node config.'''

import unittest

from pydantic import ValidationError

from cvs.lib.training.megatron.utils.training_config_loader import MegatronVariantConfig
from cvs.unittests.support import TMP


def _config(**overrides):
    data = {
        "gpu_name": "MI300X",
        "enforce_thresholds": False,
        "paths": {
            "hf_token_file": f"{TMP}/token",
            "log_dir": f"{TMP}/logs",
            "scripts_dir": f"{TMP}/scripts",
            "data_cache_dir": f"{TMP}/cache",
        },
        "container": {
            "name": "megatron",
            "image": "rocm/megatron-lm:test",
            "runtime": {"name": "docker"},
        },
        "train_params": {
            "model_name": "llama",
            "micro_batch_size": "1",
            "global_batch_size": "8",
            "precision": "BF16",
        },
        "thresholds": {"default": {}},
    }
    data.update(overrides)
    return MegatronVariantConfig.model_validate(data)


class TestGpusPerNode(unittest.TestCase):
    def test_defaults_to_eight(self):
        self.assertEqual(_config().gpus_per_node, 8)

    def test_reads_config_value(self):
        self.assertEqual(_config(gpus_per_node=4).gpus_per_node, 4)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValidationError):
            _config(gpus_per_node=0)

    def test_derives_from_container_env_gpus_per_node(self):
        cfg = _config(
            container={
                "name": "megatron",
                "image": "rocm/megatron-lm:test",
                "runtime": {"name": "docker"},
                "env": {"GPUS_PER_NODE": "4"},
            }
        )
        self.assertEqual(cfg.gpus_per_node, 4)

    def test_explicit_gpus_per_node_wins_and_overwrites_container_env(self):
        cfg = _config(
            gpus_per_node=8,
            container={
                "name": "megatron",
                "image": "rocm/megatron-lm:test",
                "runtime": {"name": "docker"},
                "env": {"GPUS_PER_NODE": "4"},
            },
        )
        self.assertEqual(cfg.gpus_per_node, 8)
        self.assertEqual(cfg.container.env["GPUS_PER_NODE"], "8")

    def test_explicit_gpus_per_node_adds_container_env_when_absent(self):
        cfg = _config(gpus_per_node=4)
        self.assertEqual(cfg.gpus_per_node, 4)
        self.assertEqual(cfg.container.env["GPUS_PER_NODE"], "4")


class TestGpuName(unittest.TestCase):
    def test_accepts_mi300x(self):
        self.assertEqual(_config(gpu_name="MI300X").gpu_name, "MI300X")

    def test_accepts_mi325x(self):
        self.assertEqual(_config(gpu_name="MI325X").gpu_name, "MI325X")

    def test_accepts_mi355x(self):
        self.assertEqual(_config(gpu_name="MI355X").gpu_name, "MI355X")

    def test_accepts_mi400x(self):
        self.assertEqual(_config(gpu_name="MI400X").gpu_name, "MI400X")

    def test_accepts_mi455x(self):
        self.assertEqual(_config(gpu_name="MI455X").gpu_name, "MI455X")

    def test_uppercases_lowercase_input(self):
        self.assertEqual(_config(gpu_name="mi400x").gpu_name, "MI400X")

    def test_rejects_unsupported_gpu_name(self):
        with self.assertRaises(ValidationError):
            _config(gpu_name="MI250X")


if __name__ == "__main__":
    unittest.main()
