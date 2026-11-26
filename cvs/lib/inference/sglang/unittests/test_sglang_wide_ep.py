'''Unit tests for cvs/lib/inference/sglang/sglang_wide_ep_lib.py.'''

import unittest
from unittest import mock

from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed
from cvs.lib.inference.sglang.sglang_wide_ep_lib import SglangWideEP, distributed_controller_class

_STAGE = 'cvs.lib.inference.sglang.sglang_distributed_lib.stage_and_start_launch_script'
_SLEEP = 'cvs.lib.inference.sglang.sglang_distributed_lib.time.sleep'
_OK = 'WIDE_EP_PREFLIGHT={"gpus": 8, "mori": "0.1", "model_found": true}'


def _inference(**overrides):
    inference = {
        'nnodes': '2',
        '_execution_hosts': ['node-a', 'node-b'],
        'log_dir': '/logs',
        'container_name': 'sglang_container',
        'log_level': 'info',
        'dist_init_port': '40001',
        'proxy_router_serv_port': '8010',
        'sglang_cli': {
            'split_cuda_graph_flags': True,
            'router_binary': 'sgl-router',
            'benchmark_module': 'sglang.benchmark.serving',
        },
    }
    inference.update(overrides)
    return inference


def _benchmark(**overrides):
    bp = {
        'model': '/root/models/DeepSeek-V4.1-Flash',
        'tensor_parallelism': '16',
        'pipeline_parallelism': '1',
        'expert_parallelism': '16',
        'attn_data_parallelism': '16',
        'memory_fraction': '0.7',
        'inference_poll_iterations': '1',
        'inference_poll_total_timeout_sec': '1',
        'max_concurrency': '64',
        'add_flags': ['--attention-backend aiter', '--moe-a2a-backend mori'],
    }
    bp.update(overrides)
    return bp


def _orch(hosts=('node-a', 'node-b'), dp_form='attn-dp-size'):
    orch = mock.Mock()
    orch.hosts = list(hosts)
    orch.exec_on_host.return_value = {'node-a': 'Mon Nov  9 12:00'}
    orch.exec.return_value = {'node-a': f'WIDE_EP_DP_ATTENTION={dp_form}\n'}
    return orch


def _launch_bodies(obj):
    with mock.patch(_SLEEP), mock.patch(_STAGE) as stage:
        obj.launch_server()
    return [call.args[3] for call in stage.call_args_list]


class TestDistributedControllerClass(unittest.TestCase):
    def test_missing_or_false_flag_selects_plain_distributed(self):
        self.assertIs(distributed_controller_class({}), SglangDistributed)
        self.assertIs(distributed_controller_class({'enable_wideEP': 'false'}), SglangDistributed)
        self.assertIs(distributed_controller_class({'enable_wideEP': False}), SglangDistributed)

    def test_true_flag_selects_wide_ep(self):
        self.assertIs(distributed_controller_class({'enable_wideEP': 'true'}), SglangWideEP)
        self.assertIs(distributed_controller_class({'enable_wideEP': True}), SglangWideEP)

    def test_invalid_flag_raises(self):
        with self.assertRaisesRegex(ValueError, 'enable_wideEP must be true or false'):
            distributed_controller_class({'enable_wideEP': 'yes'})


class TestLaunchFlags(unittest.TestCase):
    def test_wide_ep_launch_passes_expert_parallel_flags_on_every_rank(self):
        obj = SglangWideEP('m', _inference(), _benchmark(), orch=_orch())

        rank0, rank1 = _launch_bodies(obj)

        for body in (rank0, rank1):
            self.assertIn('--tp-size 16', body)
            self.assertIn('--pp-size 1', body)
            self.assertIn('--ep-size 16', body)
            self.assertIn('--attn-dp-size 16', body)
            self.assertNotIn('--dp-size', body)
            self.assertNotIn('--enable-dp-attention', body)
            self.assertIn('--deepep-mode normal', body)
            self.assertIn('--moe-a2a-backend mori', body)
            self.assertIn('--attention-backend aiter', body)
        self.assertIn('--node-rank 0', rank0)
        self.assertIn('--node-rank 1', rank1)

    def test_image_without_attn_dp_size_gets_dp_size_with_enable_dp_attention(self):
        obj = SglangWideEP('m', _inference(), _benchmark(), orch=_orch(dp_form='enable-dp-attention'))

        for body in _launch_bodies(obj):
            self.assertIn('--dp-size 16', body)
            self.assertIn('--enable-dp-attention', body)
            self.assertNotIn('--attn-dp-size', body)

    def test_dp_attention_form_is_probed_once_per_run(self):
        obj = SglangWideEP('m', _inference(), _benchmark(), orch=_orch())

        _launch_bodies(obj)
        _launch_bodies(obj)

        probes = [c for c in obj.orch.exec.call_args_list if 'launch_server --help' in c.args[0]]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].kwargs['hosts'], ['node-a'])

    def test_launch_fails_when_the_image_help_names_neither_dp_flag(self):
        orch = _orch()
        orch.exec.return_value = {'node-a': 'WIDE_EP_DP_ATTENTION=unknown\n'}
        obj = SglangWideEP('m', _inference(), _benchmark(), orch=orch)

        with self.assertRaisesRegex(RuntimeError, 'could not find --attn-dp-size or --enable-dp-attention'):
            _launch_bodies(obj)

    def test_plain_distributed_launch_has_no_expert_parallel_flags(self):
        obj = SglangDistributed('m', _inference(), _benchmark(), orch=_orch())

        for body in _launch_bodies(obj):
            self.assertIn('--tp-size 16', body)
            for flag in ('--ep-size', '--dp-size', '--enable-dp-attention', '--deepep-mode'):
                self.assertNotIn(flag, body)

    def test_deepep_mode_from_config_reaches_the_launch(self):
        obj = SglangWideEP('m', _inference(), _benchmark(deepep_mode='low_latency'), orch=_orch())
        self.assertIn('--deepep-mode low_latency', _launch_bodies(obj)[0])

    def test_server_env_exports_expert_parallel_sizes(self):
        body = SglangWideEP('m', _inference(), _benchmark(), orch=_orch())._server_env_body()
        self.assertIn('export EP=16\n', body)
        self.assertIn('export ATTN_DP=16\n', body)
        self.assertIn('export TP=16\n', body)

    def test_plain_distributed_env_has_no_expert_parallel_exports(self):
        body = SglangDistributed('m', _inference(), _benchmark(), orch=_orch())._server_env_body()
        self.assertNotIn('export EP=', body)
        self.assertNotIn('export ATTN_DP=', body)


class TestLayoutRejection(unittest.TestCase):
    def test_rejects_pipeline_parallel(self):
        with self.assertRaisesRegex(ValueError, 'pipeline_parallelism 1'):
            SglangWideEP('m', _inference(), _benchmark(pipeline_parallelism='2'), orch=_orch())

    def test_rejects_expert_size_that_differs_from_tensor_parallel(self):
        with self.assertRaisesRegex(ValueError, 'expert_parallelism == tensor_parallelism'):
            SglangWideEP('m', _inference(), _benchmark(expert_parallelism='8'), orch=_orch())

    def test_rejects_attention_dp_that_differs_from_expert_parallel(self):
        with self.assertRaisesRegex(ValueError, 'attn_data_parallelism == expert_parallelism'):
            SglangWideEP('m', _inference(), _benchmark(attn_data_parallelism='8'), orch=_orch())

    def test_rejects_tensor_parallel_that_does_not_split_across_nnodes(self):
        inference = _inference(nnodes='3', _execution_hosts=['node-a', 'node-b', 'node-c'])
        with self.assertRaisesRegex(ValueError, 'split evenly across nnodes'):
            SglangWideEP('m', inference, _benchmark(), orch=_orch(['node-a', 'node-b', 'node-c']))

    def test_rejects_add_flags_that_override_the_layout(self):
        bp = _benchmark(add_flags=['--moe-a2a-backend mori', '--ep-size 8'])
        with self.assertRaisesRegex(ValueError, 'add_flags may not set --ep-size'):
            SglangWideEP('m', _inference(), bp, orch=_orch())


class TestPreflight(unittest.TestCase):
    def _obj(self, out_dict, **bp):
        obj = SglangWideEP('m', _inference(), _benchmark(**bp), orch=_orch())
        obj.orch.exec = mock.Mock(return_value=out_dict)
        return obj

    def test_runs_in_the_container_on_every_server_node(self):
        obj = self._obj({'node-a': _OK, 'node-b': _OK})

        self.assertEqual(obj.wide_ep_preflight_failures(), [])

        cmd = obj.orch.exec.call_args.args[0]
        self.assertIn('/root/models/DeepSeek-V4.1-Flash', cmd)
        self.assertTrue(cmd.split('2>&1')[0].rstrip().endswith('1'))
        self.assertEqual(obj.orch.exec.call_args.kwargs['hosts'], ['node-a', 'node-b'])

    def test_reports_mori_missing_on_one_node(self):
        bad = 'WIDE_EP_PREFLIGHT={"gpus": 8, "mori_error": "ImportError: libmori.so", "model_found": true}'
        failures = self._obj({'node-a': _OK, 'node-b': bad}).wide_ep_preflight_failures()
        self.assertEqual(len(failures), 1)
        self.assertIn('node-b: MoRI is not importable', failures[0])

    def test_reports_a_node_the_orchestrator_returned_nothing_for(self):
        failures = self._obj({'node-a': _OK}).wide_ep_preflight_failures()
        self.assertEqual(failures, ['node-b: no preflight output from the orchestrator'])

    def test_skips_mori_when_another_a2a_backend_is_selected(self):
        probe = 'WIDE_EP_PREFLIGHT={"gpus": 8, "model_found": true}'
        obj = self._obj({'node-a': probe, 'node-b': probe}, add_flags=['--moe-a2a-backend mooncake'])

        self.assertEqual(obj.wide_ep_preflight_failures(), [])
        self.assertTrue(obj.orch.exec.call_args.args[0].split('2>&1')[0].rstrip().endswith('0'))


class TestGpuCounts(unittest.TestCase):
    def test_records_expert_and_attention_dp(self):
        obj = SglangWideEP('m', _inference(), _benchmark(), orch=_orch())
        topo = {
            'groups': {'server': {'per_node': {'node-a': 8, 'node-b': 8}, 'total': 16}},
            'total_occupied_gpus': 16,
        }
        with mock.patch(
            'cvs.lib.inference.sglang.sglang_distributed_lib.collect_sglang_gpu_topology',
            return_value=topo,
        ):
            result = obj.sglang_distributed_gpu_counts()

        self.assertEqual(result['configured_tp'], 16)
        self.assertEqual(result['configured_pp'], 1)
        self.assertEqual(result['configured_ep'], 16)
        self.assertEqual(result['configured_attn_dp'], 16)
        self.assertEqual(result['total_occupied_gpus'], 16)


if __name__ == '__main__':
    unittest.main()
