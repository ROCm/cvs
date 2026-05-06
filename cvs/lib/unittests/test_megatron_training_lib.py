'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/lib/megatron_training_lib.py: helper builds a fully-mocked
# MegatronLlamaTrainingJob so subsequent tests can target individual methods
# without SSH or real time.sleep waits.

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import megatron_training_lib  # noqa: F401  (referenced by name in subsequent PRs)
from cvs.lib.megatron_training_lib import MegatronLlamaTrainingJob


def _make_megatron_job(
    *,
    training_overrides=None,
    distributed_training=True,
    model_name='llama3_1_8b',
    gpu_type='mi300',
    phdl_exec_returns=None,
):
    """Construct a MegatronLlamaTrainingJob with all heavyweight init paths mocked.

    Args:
        training_overrides: dict shallow-merged onto a minimal default training_dict.
        distributed_training: passed straight to the constructor.
        model_name: key in model_params_dict[topology].
        gpu_type: key in model_params_dict[topology][model_name].
        phdl_exec_returns: string returned by every phdl.exec(...) call (mapped
            per-host via host_list). Default '' which forces detect_rocm_path's
            legacy /opt/rocm fallback.

    Notes:
        - time.sleep is patched out so the constructor's two 2-second sleeps
          and detect_rocm_path are zero-cost.
        - phdl.exec is a MagicMock returning {'n0': phdl_exec_returns}, so
          ibv_devinfo / cat training.log / etc. all see the same fixture.
    """
    phdl = MagicMock()
    phdl.host_list = ['n0']
    phdl.user = 'tester'
    phdl.exec = MagicMock(return_value={'n0': phdl_exec_returns or ''})
    phdl.exec_cmd_list = MagicMock(return_value=None)

    training_dict = {'training_iterations': 2, 'nnodes': 1, 'nic_type': 'thor2'}
    if training_overrides:
        training_dict.update(training_overrides)

    topology = 'multi_node' if distributed_training else 'single_node'
    model_params_dict = {
        topology: {
            model_name: {
                gpu_type: {
                    'tokenizer_model': 'NousResearch/Meta-Llama-3-8B',
                    'model_size': '8',
                    'batch_size': '128',
                    'micro_batch_size': '2',
                    'precision': 'TE_FP8',
                    'sequence_length': '8192',
                    'tensor_parallelism': '1',
                    'pipeline_parallelism': '1',
                    'recompute': '0',
                    'fsdp': '0',
                    'result_dict': {'throughput_per_gpu': '0.0', 'tokens_per_gpu': '0.0'},
                }
            }
        }
    }

    with patch('cvs.lib.megatron_training_lib.time.sleep'):
        return MegatronLlamaTrainingJob(
            phdl,
            model_name,
            training_dict,
            model_params_dict,
            hf_token='dummy',
            gpu_type=gpu_type,
            distributed_training=distributed_training,
            tune_model_params=False,
        )


class TestMegatronJobConstructor(unittest.TestCase):
    """Smoke test that _make_megatron_job actually constructs a job with default config."""

    def test_init_constructs_with_default_config(self):
        job = _make_megatron_job()
        self.assertEqual(int(job.nnodes), 1)
        self.assertEqual(job.nic_type, 'thor2')
        # tokenizer_model contains 'Llama-3', so __init__ picks the train_llama3.sh branch.
        self.assertTrue(job.training_script.endswith('train_llama3.sh'))


if __name__ == '__main__':
    unittest.main()
