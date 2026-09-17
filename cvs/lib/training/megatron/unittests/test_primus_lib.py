'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for Primus training command construction.
'''

import unittest
from unittest.mock import MagicMock

from cvs.lib.training.megatron.primus_lib import PrimusTrainingJob


class _FakeVariant:
    gpu_name = "MI325X"

    def __init__(self, train_params):
        self.train_params = train_params

    def job_config_dict(self):
        return {}


def _job(train_params, **kwargs):
    orch = MagicMock()
    orch.hosts = ["n0"]
    orch.all.exec.return_value = {}
    return PrimusTrainingJob(
        orch,
        _FakeVariant(train_params),
        hf_token="token",
        micro_batch_size=train_params.get("micro_batch_size", "2"),
        global_batch_size=train_params.get("global_batch_size", "16"),
        precision=train_params.get("precision", "FP8"),
        **kwargs,
    )


class TestPrimusBuildTrainingJobCmd(unittest.TestCase):
    def test_emits_tp_and_pp_cli_flags(self):
        job = _job(
            {
                "model_name": "llama3.1_8B",
                "tokenizer_model": "meta-llama/Llama-3.1-8B",
                "micro_batch_size": "2",
                "global_batch_size": "16",
                "precision": "FP8",
                "training_iterations": "20",
                "tensor_parallelism": "8",
                "pipeline_parallelism": "2",
            }
        )
        job.build_training_job_cmd()
        self.assertIn("--tensor_model_parallel_size 8", job.job_cmd)
        self.assertIn("--pipeline_model_parallel_size 2", job.job_cmd)

    def test_defaults_tp_pp_to_one(self):
        job = _job(
            {
                "model_name": "llama3.1_8B",
                "tokenizer_model": "meta-llama/Llama-3.1-8B",
                "micro_batch_size": "2",
                "global_batch_size": "16",
                "precision": "BF16",
                "training_iterations": "10",
            }
        )
        job.build_training_job_cmd()
        self.assertIn("--tensor_model_parallel_size 1", job.job_cmd)
        self.assertIn("--pipeline_model_parallel_size 1", job.job_cmd)

    def test_sweep_overrides_tp_pp(self):
        job = _job(
            {
                "model_name": "llama3.1_8B",
                "tokenizer_model": "meta-llama/Llama-3.1-8B",
                "micro_batch_size": "2",
                "global_batch_size": "16",
                "precision": "BF16",
                "training_iterations": "10",
                "tensor_parallelism": "1",
                "pipeline_parallelism": "1",
            },
            sweep_overrides={"tensor_parallelism": "4", "pipeline_parallelism": "2"},
        )
        job.build_training_job_cmd()
        self.assertIn("--tensor_model_parallel_size 4", job.job_cmd)
        self.assertIn("--pipeline_model_parallel_size 2", job.job_cmd)


if __name__ == "__main__":
    unittest.main()
