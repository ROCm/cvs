'''Unit tests for Megatron gpus_per_node config.'''

import unittest

from pydantic import ValidationError

from cvs.lib.training.megatron.utils.training_config_loader import MegatronVariantConfig


def _config(**overrides):
    data = {
        "gpu_name": "MI300X",
        "enforce_thresholds": False,
        "paths": {
            "hf_token_file": "/tmp/token",
            "log_dir": "/tmp/logs",
            "scripts_dir": "/tmp/scripts",
            "data_cache_dir": "/tmp/cache",
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


if __name__ == "__main__":
    unittest.main()
