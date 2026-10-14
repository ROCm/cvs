'''Unit tests for TorchTitan gpus_per_node config.'''

import unittest

from pydantic import ValidationError

from cvs.lib.training.torchtitan.utils.training_config_loader import TorchTitanVariantConfig
from cvs.unittests.support import TMP


def _config(**overrides):
    data = {
        "gpu_name": "MI355X",
        "enforce_thresholds": False,
        "paths": {
            "hf_token_file": f"{TMP}/token",
            "log_dir": f"{TMP}/logs",
            "scripts_dir": f"{TMP}/scripts",
            "data_cache_dir": f"{TMP}/cache",
        },
        "container": {
            "name": "torchtitan",
            "image": "rocm/torchtitan:test",
            "runtime": {"name": "docker"},
        },
        "train_params": {
            "model_name": "llama3_1_8b",
            "micro_batch_size": "1",
            "global_batch_size": "8",
            "precision": "BF16",
        },
        "sweep": {
            "combinations": {"default": {"name": "default"}},
            "runs": ["default"],
        },
        "thresholds": {"MBS=1,GBS=16,PRECISION=BF16": {}},
    }
    data.update(overrides)
    return TorchTitanVariantConfig.model_validate(data)


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
