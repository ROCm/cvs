'''Unit tests for cvs/lib/inference/sglang/sglang_wide_ep_lib.py.'''

import unittest
from unittest import mock

from cvs.lib.inference.sglang.sglang_wide_ep_lib import SglangWideEP


def _inference():
    return {
        'nnodes': '2',
        '_execution_hosts': ['node-a', 'node-b'],
        'log_dir': '/logs',
        'container_name': 'sglang_container',
        'log_level': 'info',
        'dist_init_port': '40001',
        'proxy_router_serv_port': '8010',
    }


def _benchmark(**overrides):
    bp = {
        'model': '/root/models/DeepSeek-R1-0528',
        'tensor_parallelism': '16',
        'pipeline_parallelism': '1',
        'expert_parallelism': '16',
        'attn_data_parallelism': '16',
        'deepep_mode': 'normal',
        'memory_fraction': '0.7',
        'inference_poll_iterations': '1',
        'inference_poll_total_timeout_sec': '1',
        'max_concurrency': '64',
    }
    bp.update(overrides)
    return bp


def _orch():
    orch = mock.Mock()
    orch.hosts = ['node-a', 'node-b']
    orch.exec_on_host.return_value = {'node-a': 'Mon Nov  9 12:00'}
    return orch


class TestSglangWideEP(unittest.TestCase):
    def test_launch_server_passes_expert_parallel_flags(self):
        obj = SglangWideEP(
            'DeepSeek-R1-0528',
            _inference(),
            _benchmark(add_flags=['--attention-backend aiter', '--moe-a2a-backend mooncake']),
            orch=_orch(),
        )
        with (
            mock.patch('cvs.lib.inference.sglang.sglang_wide_ep_lib.time.sleep'),
            mock.patch('cvs.lib.inference.sglang.sglang_wide_ep_lib.stage_and_start_launch_script') as stage,
        ):
            obj.launch_server()

        self.assertEqual(stage.call_count, 2)
        rank0 = stage.call_args_list[0]
        body = rank0.args[3]
        self.assertIn('--tp-size 16', body)
        self.assertIn('--pp-size 1', body)
        self.assertIn('--ep-size 16', body)
        self.assertIn('--dp-size 16', body)
        self.assertIn('--enable-dp-attention', body)
        self.assertNotIn('--attn-dp-size', body)
        self.assertIn('--moe-a2a-backend mooncake', body)
        self.assertIn('--deepep-mode normal', body)
        self.assertIn('--disable-cuda-graph', body)
        self.assertIn('--attention-backend aiter', body)
        self.assertEqual(rank0.args[1], 'node-a')
        self.assertIn('--node-rank 0', body)
        rank1 = stage.call_args_list[1].args[3]
        self.assertIn('--node-rank 1', rank1)
        self.assertEqual(obj._configured_gpu_count(), 16)

    def test_launch_server_omits_all_to_all_backend_unless_add_flags_sets_it(self):
        obj = SglangWideEP('DeepSeek-R1-0528', _inference(), _benchmark(), orch=_orch())
        with (
            mock.patch('cvs.lib.inference.sglang.sglang_wide_ep_lib.time.sleep'),
            mock.patch('cvs.lib.inference.sglang.sglang_wide_ep_lib.stage_and_start_launch_script') as stage,
        ):
            obj.launch_server()
        body = stage.call_args_list[0].args[3]
        self.assertNotIn('--moe-a2a-backend', body)

    def test_server_env_exports_expert_parallel_sizes(self):
        obj = SglangWideEP('DeepSeek-R1-0528', _inference(), _benchmark(), orch=_orch())
        body = obj._server_env_body()
        self.assertIn('export EP=16\n', body)
        self.assertIn('export ATTN_DP=16\n', body)
        self.assertNotIn('MOE_A2A_BACKEND', body)
        self.assertIn('export TP=16\n', body)
        self.assertIn('export PP=1\n', body)

    def test_rejects_pipeline_parallel(self):
        with self.assertRaisesRegex(ValueError, 'pipeline_parallelism 1'):
            SglangWideEP(
                'DeepSeek-R1-0528',
                _inference(),
                _benchmark(pipeline_parallelism='2'),
                orch=_orch(),
            )

    def test_rejects_expert_size_that_differs_from_tensor_parallel(self):
        with self.assertRaisesRegex(ValueError, 'expert_parallelism == tensor_parallelism'):
            SglangWideEP(
                'DeepSeek-R1-0528',
                _inference(),
                _benchmark(expert_parallelism='8'),
                orch=_orch(),
            )

    def test_rejects_attention_dp_that_differs_from_expert_parallel(self):
        with self.assertRaisesRegex(ValueError, 'attn_data_parallelism == expert_parallelism'):
            SglangWideEP(
                'DeepSeek-R1-0528',
                _inference(),
                _benchmark(attn_data_parallelism='8'),
                orch=_orch(),
            )

    def test_gpu_counts_record_expert_and_attention_dp(self):
        obj = SglangWideEP('DeepSeek-R1-0528', _inference(), _benchmark(), orch=_orch())
        obj._host_exec = mock.Mock(return_value={})
        topo = {
            'groups': {'server': {'per_node': {'node-a': 8, 'node-b': 8}, 'total': 16}},
            'total_occupied_gpus': 16,
        }
        with mock.patch(
            'cvs.lib.inference.sglang.sglang_wide_ep_lib.collect_sglang_gpu_topology',
            return_value=topo,
        ):
            result = obj.sglang_wide_ep_gpu_counts()
        self.assertEqual(result['configured_ep'], 16)
        self.assertEqual(result['configured_attn_dp'], 16)
        self.assertEqual(result['configured_pp'], 1)
        self.assertNotIn('moe_a2a_backend', result)
        self.assertEqual(result['total_occupied_gpus'], 16)


if __name__ == '__main__':
    unittest.main()
