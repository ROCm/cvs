'''Unit tests for cvs/lib/inference/sglang/sglang_config_loader.py.'''

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from cvs.lib.inference.sglang import sglang_config_loader as loader
from cvs.unittests.support import FAKE_CRED

FAKE_CRED_FILE = '/home/user/.hf_token'

_THRESHOLDS = {
    'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64': {
        'output_throughput': {'kind': 'min', 'value': 900},
        'mean_ttft_ms': {'kind': 'max', 'value': 60000},
        'mfu': {'kind': 'min', 'value': 0.02},
    },
    'ISL=8192,OSL=1024,TP=8,PP=1,CONC=64': {
        'output_throughput': {'kind': 'min', 'value': 450},
    },
    'ACC_ISL=131072,OSL=1024': {'pass_rate': {'kind': 'min', 'value': 0.95}},
    'BENCH=lm_eval_gsm8k': {'exact_match,flexible-extract': {'kind': 'min', 'value': 0.95}},
}


class TestFlatExpectedFromSpecs(unittest.TestCase):
    def test_unwraps_spec_dicts_and_bare_values(self):
        flat = loader.flat_expected_from_specs(
            {
                'mean_ttft_ms': {'kind': 'max', 'value': 60000},
                'mfu': 0.02,
            }
        )
        self.assertEqual(flat, {'mean_ttft_ms': 60000.0, 'mfu': 0.02})


class TestPerfCellKey(unittest.TestCase):
    def test_builds_key_from_benchmark_params(self):
        bp = {
            'tensor_parallelism': '8',
            'pipeline_parallelism': '1',
            'max_concurrency': '64',
            'inference_tests': {'bench_serv_random': {'input_length': '1024', 'output_length': '1024'}},
        }
        self.assertEqual(loader.perf_cell_key(bp), 'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64')


class TestPerfCellsFromThresholds(unittest.TestCase):
    def test_selects_and_sorts_perf_cells_only(self):
        cells = loader.perf_cells_from_thresholds(_THRESHOLDS)
        self.assertEqual(
            [(c['isl'], c['osl'], c['conc']) for c in cells],
            [('1024', '1024', '64'), ('8192', '1024', '64')],
        )


class TestPerfCellsForVariant(unittest.TestCase):
    def test_selects_runs_and_attaches_overrides(self):
        variant = mock.Mock(
            thresholds=_THRESHOLDS,
            benchmark_params={'tensor_parallelism': '8', 'pipeline_parallelism': '1'},
            sweeps={
                'ISL=8192,OSL=1024,TP=8,PP=1,CONC=64': {
                    'num_prompts': '50',
                }
            },
            sweep={
                'runs': [
                    {
                        'combo': 'ISL=8192,OSL=1024,TP=8,PP=1,CONC=64',
                    }
                ]
            },
        )

        cells = loader.perf_cells_for_variant(variant)

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]['isl'], '8192')
        self.assertEqual(cells[0]['benchmark_overrides'], {'num_prompts': '50'})
        self.assertEqual(cells[0]['benchmark_params']['num_prompts'], '50')

    def test_empty_runs_uses_every_threshold_cell(self):
        variant = mock.Mock(
            thresholds=_THRESHOLDS,
            benchmark_params={'tensor_parallelism': '8', 'pipeline_parallelism': '1'},
            sweeps={},
            sweep={'runs': []},
        )

        cells = loader.perf_cells_for_variant(variant)

        self.assertEqual(
            [cell['cell_key'] for cell in cells],
            ['ISL=1024,OSL=1024,TP=8,PP=1,CONC=64', 'ISL=8192,OSL=1024,TP=8,PP=1,CONC=64'],
        )

    def test_selected_sweep_combo_keys_empty_runs(self):
        threshold_cells = loader.perf_cells_from_thresholds(_THRESHOLDS)
        self.assertEqual(
            loader.selected_sweep_combo_keys({'runs': []}, threshold_cells),
            [cell['cell_key'] for cell in threshold_cells],
        )

    def test_selected_sweep_combo_keys_listed_runs(self):
        threshold_cells = loader.perf_cells_from_thresholds(_THRESHOLDS)
        self.assertEqual(
            loader.selected_sweep_combo_keys(
                {
                    'runs': [
                        {'combo': 'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64'},
                        {'combo': 'ISL=1024,OSL=1024,TP=8,PP=1,CONC=128'},
                    ]
                },
                threshold_cells,
            ),
            [
                'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64',
                'ISL=1024,OSL=1024,TP=8,PP=1,CONC=128',
            ],
        )

    def test_rejects_combo_without_matching_threshold_shape(self):
        variant = mock.Mock(
            thresholds=_THRESHOLDS,
            benchmark_params={'tensor_parallelism': '8', 'pipeline_parallelism': '1'},
            sweeps={},
            sweep={
                'runs': [
                    {
                        'combo': 'ISL=4096,OSL=1024,TP=8,PP=1,CONC=64',
                    }
                ]
            },
        )

        with self.assertRaisesRegex(ValueError, 'no matching threshold cell'):
            loader.perf_cells_for_variant(variant)


class TestPerfSpecsForCell(unittest.TestCase):
    def test_returns_flattened_gates(self):
        self.assertEqual(
            loader.perf_specs_for_cell(_THRESHOLDS, '1024', '1024', '64'),
            {'output_throughput': 900.0, 'mean_ttft_ms': 60000.0, 'mfu': 0.02},
        )

    def test_accepts_int_cell_coordinates(self):
        self.assertEqual(loader.perf_specs_for_cell(_THRESHOLDS, 8192, 1024, 64), {'output_throughput': 450.0})

    def test_ignores_tp_and_pp_in_cell_key(self):
        thresholds = {'ISL=1024,OSL=1024,TP=16,PP=2,CONC=8': {'mfu': {'kind': 'min', 'value': 0.05}}}
        self.assertEqual(loader.perf_specs_for_cell(thresholds, '1024', '1024', '8'), {'mfu': 0.05})

    def test_unknown_cell_returns_empty(self):
        self.assertEqual(loader.perf_specs_for_cell(_THRESHOLDS, '1024', '1024', '256'), {})

    def test_skips_null_specs(self):
        thresholds = {'ISL=1024,OSL=1024,TP=8,PP=1,CONC=4': {'mfu': None, 'goodput': {'kind': 'min', 'value': 0.99}}}
        self.assertEqual(loader.perf_specs_for_cell(thresholds, '1024', '1024', '4'), {'goodput': 0.99})


class TestResolveBenchmarkVariantKey(unittest.TestCase):
    def setUp(self):
        # The resolver honours SGLANG_BENCHMARK_KEY; pin it empty so the result
        # does not depend on the developer's or CI runner's environment.
        patcher = mock.patch.dict(os.environ, {'SGLANG_BENCHMARK_KEY': ''})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_variant_is_selected(self):
        root = {'benchmark_params': {'llama-70b': {}}}
        self.assertEqual(loader.resolve_benchmark_variant_key(root, 'cfg.json'), 'llama-70b')

    def test_multiple_variants_require_explicit_selection(self):
        root = {'benchmark_params': {'a': {}, 'b': {}}}
        with self.assertRaises(ValueError):
            loader.resolve_benchmark_variant_key(root, 'cfg.json')

    def test_missing_benchmark_params_raises(self):
        with self.assertRaises(ValueError):
            loader.resolve_benchmark_variant_key({}, 'cfg.json')


CELL = 'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64'


def _new_config(**overrides):
    config = {
        'enforce_thresholds': False,
        'threshold_json': 'threshold.json',
        'paths': {'shared_fs': '/fs', 'models_dir': '/models', 'log_dir': '/logs', 'hf_token_file': FAKE_CRED_FILE},
        'container': {
            'name': 'sglang',
            'image': 'image',
            'runtime': {
                'name': 'docker',
                'args': {'shm_size': '128G', 'volumes': ['/host:/container', '/host-ro:/container-ro:ro']},
            },
            'env': {'NCCL_IB_HCA': 'rdma0', 'SGLANG_USE_AITER': '1'},
        },
        'server_params': {
            'model': '/models/model',
            'tp': 8,
            'port': 8000,
            'master_port': 40001,
            'engine_args': {
                'attention_backend': 'aiter',
                'disable_x': True,
                'mem_fraction_static': 0.85,
                'log_level': 'info',
            },
        },
        'benchmark_params': {'num_prompts': 25, 'random_range_ratio': 0.5},
        'sweeps': {CELL: {}},
        'runs': [CELL],
        'accuracy': {
            'lm_eval': {
                'tasks': [
                    {'id': 'lm_eval_gsm8k', 'tasks': 'gsm8k', 'num_fewshot': 5, 'limit': 100, 'num_concurrent': 4}
                ]
            }
        },
    }
    config.update(overrides)
    return config


_NEW_THRESHOLDS = {
    CELL: {'output_throughput': {'kind': 'min', 'value': 900}, 'mean_ttft_ms': {'kind': 'info', 'value': 1}},
    'accuracy': {'lm_eval': {'lm_eval_gsm8k': {'exact_match,flexible-extract': {'kind': 'min', 'value': 0.95}}}},
}


def _load(config, thresholds=None, cluster=None, topology=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        (Path(tmp) / 'threshold.json').write_text(json.dumps(thresholds or _NEW_THRESHOLDS), encoding='utf-8')
        return loader.load_variant(str(path), cluster or {'username': 'test'}, topology)


class TestRuntimeViews(unittest.TestCase):
    def test_builds_the_job_dicts_from_the_validated_config(self):
        variant = _load(_new_config())
        inference, params = variant.inference, variant.benchmark_params

        self.assertEqual(params['tensor_parallelism'], '8')
        self.assertEqual(params['pipeline_parallelism'], '1')
        self.assertEqual(params['memory_fraction'], '0.85')
        self.assertEqual(params['add_flags'], ['--attention-backend aiter', '--disable-x'])
        self.assertEqual(inference['log_level'], 'info')
        self.assertEqual(inference['proxy_router_serv_port'], '8000')
        self.assertEqual(inference['dist_init_port'], '40001')
        self.assertEqual(inference['nccl_ib_hca'], 'rdma0')
        self.assertEqual(inference['hsa_force_fine_grain_pcie'], '1')
        self.assertEqual(inference['container_config']['volume_dict']['/host-ro'], '/container-ro:ro')
        self.assertEqual(params['add_export_env'], ['SGLANG_USE_AITER=1'])
        self.assertEqual(
            params['inference_tests']['bench_serv_random'],
            {'backend': 'sglang', 'num_prompts': 25, 'random_range_ratio': 0.5, 'enforce_thresholds': False},
        )
        gsm8k = params['inference_tests']['lm_eval_gsm8k']
        self.assertEqual(gsm8k['limit'], 100)
        self.assertEqual(gsm8k['expected_results']['gsm8k'], {'exact_match,flexible-extract': 0.95})
        self.assertEqual(variant.roles.server.env['SGLANG_USE_AITER'], '1')
        self.assertEqual(variant.roles.server.env['HSA_FORCE_FINE_GRAIN_PCIE'], '1')
        self.assertEqual(variant.thresholds[CELL]['mean_ttft_ms'], {'kind': 'info', 'value': 1.0})
        self.assertNotIn('inference_poll_iterations', params)

    def test_client_s_gives_the_poll_count_and_wait_of_the_old_keys(self):
        for client_s, polls, total in (
            (1500, '16', '3600'),
            (3660, '40', '3600'),
            (100, '1', '3600'),
            (4000, '44', '4030'),
        ):
            with self.subTest(client_s=client_s):
                params = _load(_new_config(timeouts={'client_s': client_s})).benchmark_params
                self.assertEqual(params['inference_poll_iterations'], polls)
                self.assertEqual(params["inference_poll_total_timeout_sec"], total)

    def test_a_niah_eval_turns_on_the_long_context_server_flags(self):
        server = {
            **_new_config()['server_params'],
            'max_model_length': 131072,
            'long_context': {'chunked_prefill_size': 8192},
        }
        niah = {'type': 'niah', 'id': 'niah_8k', 'isl': 8192, 'osl': 32, 'num_prompts': 6}
        thresholds = {
            **_NEW_THRESHOLDS,
            'accuracy': {'custom_eval': {'niah_8k': {'pass_rate': {'kind': 'min', 'value': 0.8}}}},
        }
        variant = _load(_new_config(server_params=server, accuracy={'custom_eval': [niah]}), thresholds)

        self.assertIs(variant.benchmark_params['lng_ctx_activate'], True)
        self.assertEqual(variant.benchmark_params['context_length'], '131072')
        self.assertEqual(variant.benchmark_params['chunked_prefill_size'], '8192')
        self.assertIn('long_ctx_niah', variant.benchmark_params['inference_tests'])
        self.assertEqual(
            loader.acc_cells_for_variant(variant),
            [
                {
                    'cell_key': 'ACC_ISL=8192,OSL=32',
                    'id': 'niah_8k',
                    'isl': '8192',
                    'osl': '32',
                    'num_prompts': 6,
                    'seed': 42,
                    'specs': {'pass_rate': {'kind': 'min', 'value': 0.8, 'optional': False}},
                }
            ],
        )

    def test_an_old_config_loads_with_one_future_warning_and_a_new_one_with_none(self):
        old = _new_config()
        old['server_params'] = {'model': '/models/model', 'tensor_parallelism': '8', 'memory_fraction': '0.85'}
        old_thresholds = {CELL: {'output_throughput_per_sec': {'kind': 'min_tok_s', 'value': 900}}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            variant = _load(old, old_thresholds)
        self.assertEqual([w.category for w in caught], [FutureWarning])
        self.assertEqual(variant.benchmark_params['memory_fraction'], '0.85')
        self.assertEqual(variant.thresholds[CELL], {'output_throughput': {'kind': 'min', 'value': 900.0}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _load(_new_config())
        self.assertEqual(caught, [])

    def test_an_old_config_loaded_without_warn_raises_no_warning(self):
        old = _new_config()
        old['server_params'] = {'model': '/models/model', 'tensor_parallelism': '8'}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'
                path.write_text(json.dumps(old), encoding='utf-8')
                (Path(tmp) / 'threshold.json').write_text(json.dumps(_NEW_THRESHOLDS), encoding='utf-8')
                variant = loader.load_variant(str(path), {'username': 'test'}, warn=False)
        self.assertEqual(caught, [])
        self.assertEqual(variant.benchmark_params['tensor_parallelism'], '8')

    def test_an_unknown_server_key_fails_naming_it(self):
        server = {**_new_config()['server_params'], 'client_host': 'node1'}
        with self.assertRaisesRegex(ValueError, 'client_host'):
            _load(_new_config(server_params=server))

    def test_hf_token_is_not_copied_into_logged_env(self):
        old = _new_config()
        old['container']['runtime']['args']['env'] = {'HF_TOKEN': FAKE_CRED, 'NCCL_DEBUG': 'ERROR'}
        del old['container']['env']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            variant = _load(old)
        self.assertNotIn('HF_TOKEN', variant.roles.server.env)
        self.assertNotIn(FAKE_CRED, str(variant.inference))
        self.assertNotIn(FAKE_CRED, str(variant.benchmark_params))


class TestLegacyRootLayout(unittest.TestCase):
    def test_old_threshold_names_convert_with_one_future_warning(self):
        thresholds = {
            CELL: {'output_throughput_per_sec': {'kind': 'min_tok_s', 'value': 900}},
            'ACC_ISL=8192,OSL=32': {'pass_rate': {'kind': 'min', 'value': 0.8}},
        }
        root = {
            'config': {
                'container_name': 'sglang',
                'container_image': 'image',
                'log_dir': '/fs/logs',
                'hf_token_file': FAKE_CRED_FILE,
                'container_config': {'volume_dict': {'/models': '/models'}},
            },
            'benchmark_params': {'v': {'model': 'm', 'threshold_file': 'threshold.json'}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(root), encoding='utf-8')
            (Path(tmp) / 'threshold.json').write_text(json.dumps(thresholds), encoding='utf-8')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                variant = loader.load_variant(str(path), {'username': 'test'})
        self.assertEqual([w.category for w in caught], [FutureWarning])
        self.assertEqual(variant.thresholds[CELL], {'output_throughput': {'kind': 'min', 'value': 900}})
        self.assertEqual([cell['isl'] for cell in loader.acc_cells_for_variant(variant)], ['8192'])


class TestEngineArgsToFlags(unittest.TestCase):
    def test_true_writes_a_bare_flag_and_lists_join(self):
        self.assertEqual(
            loader.engine_args_to_flags(
                {'attention_backend': 'aiter', 'disable_x': True, 'graph_bs': [1, 2], 'n': 2.0}
            ),
            ['--attention-backend aiter', '--disable-x', '--graph-bs 1 2', '--n 2'],
        )


class TestAddExportEnvFromRuntime(unittest.TestCase):
    def test_extra_scalar_env_becomes_export_lines(self):
        self.assertEqual(
            loader._add_export_env_from_runtime(
                {
                    'NCCL_DEBUG': 'ERROR',
                    'SGLANG_USE_AITER': '1',
                    'GPU_ARCHS': 'gfx942',
                    'HSA_FORCE_FINE_GRAIN_PCIE': '0',
                }
            ),
            ['SGLANG_USE_AITER=1', 'GPU_ARCHS=gfx942'],
        )

    def test_legacy_add_export_env_list_merges_without_duplicates(self):
        self.assertEqual(
            loader._add_export_env_from_runtime(
                {
                    'ADD_EXPORT_ENV': ['SGLANG_USE_AITER=1', 'GPU_ARCHS=gfx942'],
                    'SGLANG_USE_AITER': '1',
                },
                ['SGLANG_USE_AITER=1'],
            ),
            ['SGLANG_USE_AITER=1', 'GPU_ARCHS=gfx942'],
        )


class TestTopology(unittest.TestCase):
    def _multinode(self, nnodes):
        return _new_config(server_params={**_new_config()['server_params'], 'nnodes': nnodes})

    def test_single_selects_first_host_from_cluster(self):
        cluster = {'username': 'test', 'node_dict': {'node-a': {}, 'node-b': {}}}
        variant = _load(_new_config(), cluster=cluster, topology='single')
        self.assertEqual(variant.topology, 'single')
        self.assertEqual(variant.inference['_execution_hosts'], ['node-a'])
        self.assertEqual(variant.inference['benchmark_serv_node'], 'node-a')

    def test_distributed_selects_first_nnodes_from_cluster(self):
        cluster = {'username': 'test', 'node_dict': {'node-a': {}, 'node-b': {}, 'node-c': {}}}
        variant = _load(self._multinode(2), cluster=cluster, topology='distributed')
        self.assertEqual(variant.inference['_execution_hosts'], ['node-a', 'node-b'])
        self.assertEqual(variant.inference['server_node_list'], ['node-a', 'node-b'])
        self.assertEqual(variant.inference['benchmark_serv_node'], 'node-a')

    def test_without_a_topology_nnodes_above_one_means_distributed(self):
        self.assertEqual(_load(self._multinode(2)).topology, 'distributed')
        self.assertEqual(_load(_new_config()).topology, 'single')

    def test_distributed_fails_when_cluster_is_smaller_than_nnodes(self):
        cluster = {'username': 'test', 'node_dict': {'node-a': {}}}
        with self.assertRaisesRegex(ValueError, 'requests 2 nodes'):
            _load(self._multinode(2), cluster=cluster, topology='distributed')

    def test_distributed_fails_when_nnodes_less_than_2(self):
        with self.assertRaisesRegex(ValueError, 'nnodes >= 2'):
            _load(self._multinode(1), topology='distributed')

    def test_distributed_and_disaggregated_need_nnodes(self):
        for topology in ('distributed', 'disaggregated'):
            with self.subTest(topology=topology), self.assertRaisesRegex(ValueError, 'nnodes'):
                _load(_new_config(), topology=topology)

    def test_disagg_assigns_equal_prefill_decode_from_cluster(self):
        cluster = {'username': 'test', 'node_dict': {'n0': {}, 'n1': {}, 'n2': {}, 'n3': {}}}
        variant = _load(self._multinode(4), cluster=cluster, topology='disaggregated')
        self.assertEqual(variant.topology, 'disaggregated')
        self.assertEqual(variant.inference['prefill_node_list'], ['n0', 'n2'])
        self.assertEqual(variant.inference['decode_node_list'], ['n1', 'n3'])
        self.assertEqual(variant.inference['proxy_router_node'], 'n0')
        self.assertEqual(variant.inference['benchmark_serv_node'], 'n0')
        self.assertEqual(variant.inference['prefill_coordinator_addr'], 'n0')
        self.assertEqual(variant.inference['decode_coordinator_addr'], 'n1')
        self.assertEqual(variant.inference['prefill_serv_port'], '30001')
        self.assertEqual(variant.inference['decode_coordinator_port'], '40002')

    def test_disagg_odd_nnodes_fails_during_load(self):
        with self.assertRaisesRegex(ValueError, 'even nnodes'):
            _load(self._multinode(3), topology='disaggregated')

    def test_disagg_fails_when_cluster_has_fewer_than_two_hosts(self):
        cluster = {'username': 'test', 'node_dict': {'n0': {}}}
        with self.assertRaisesRegex(ValueError, 'at least 2 cluster.json hosts'):
            _load(self._multinode(2), cluster=cluster, topology='disaggregated')

    def test_disagg_fails_when_cluster_is_smaller_than_nnodes(self):
        cluster = {'username': 'test', 'node_dict': {'n0': {}, 'n1': {}}}
        with self.assertRaisesRegex(ValueError, 'requests 4 nodes'):
            _load(self._multinode(4), cluster=cluster, topology='disaggregated')
