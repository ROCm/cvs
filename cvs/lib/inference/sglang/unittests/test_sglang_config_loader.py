'''Unit tests for cvs/lib/inference/sglang/sglang_config_loader.py.'''

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvs.lib.inference.sglang import sglang_config_loader as loader

_THRESHOLDS = {
    'ISL=1024,OSL=1024,TP=8,PP=1,CONC=64': {
        'output_throughput_per_sec': {'kind': 'min_tok_s', 'value': 900},
        'mean_ttft_ms': {'kind': 'max_ms', 'value': 60000},
        'mfu': {'kind': 'min', 'value': 0.02},
    },
    'ISL=8192,OSL=1024,TP=8,PP=1,CONC=64': {
        'output_throughput_per_sec': {'kind': 'min_tok_s', 'value': 450},
    },
    'ACC_ISL=131072,OSL=1024': {'pass_rate': {'kind': 'min', 'value': 0.95}},
    'BENCH=lm_eval_gsm8k': {'exact_match,flexible-extract': {'kind': 'min', 'value': 0.95}},
}


class TestFlatExpectedFromSpecs(unittest.TestCase):
    def test_unwraps_spec_dicts_and_bare_values(self):
        flat = loader.flat_expected_from_specs(
            {
                'mean_ttft_ms': {'kind': 'max_ms', 'value': 60000},
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


class TestPerfSpecsForCell(unittest.TestCase):
    def test_returns_flattened_gates(self):
        self.assertEqual(
            loader.perf_specs_for_cell(_THRESHOLDS, '1024', '1024', '64'),
            {'output_throughput_per_sec': 900.0, 'mean_ttft_ms': 60000.0, 'mfu': 0.02},
        )

    def test_accepts_int_cell_coordinates(self):
        self.assertEqual(loader.perf_specs_for_cell(_THRESHOLDS, 8192, 1024, 64), {'output_throughput_per_sec': 450.0})

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


class TestUnifiedRuntimeViews(unittest.TestCase):
    def setUp(self):
        self.raw = {
            'threshold_json': 'threshold.json',
            'paths': {
                'log_dir': '/logs',
                'hf_token_file': '/home/user/.hf_token',
            },
            'model': {'id': '/models/model'},
            'container': {
                'name': 'sglang',
                'image': 'image',
                'lifetime': 'per_run',
                'runtime': {
                    'name': 'docker',
                    'args': {
                        'shm_size': '128G',
                        'volumes': ['/host:/container', '/host-ro:/container-ro:ro'],
                        'devices': ['/dev/kfd'],
                        'env': {'FROM_RUNTIME': '1'},
                    },
                },
            },
            'roles': {
                'server': {
                    'nnodes': '2',
                    'benchmark_serv_node': 'node1',
                    'proxy_router_serv_port': '8000',
                }
            },
            'params': {
                'tensor_parallelism': '8',
                'pipeline_parallelism': '2',
                'add_export_env': ['SGLANG_USE_AITER=1'],
                'inference_tests': {
                    'bench_serv_random': {
                        'enforce_thresholds': False,
                    }
                },
            },
            'accuracy': {
                'tasks': [
                    {
                        'id': 'lm_eval_hellaswag',
                        'tasks': 'hellaswag',
                    }
                ]
            },
        }
        self.thresholds = {
            'ISL=1024,OSL=1024,TP=8,PP=2,CONC=4': {
                'mfu': {'kind': 'min', 'value': 0.1},
            },
            'BENCH=lm_eval_hellaswag': {
                'acc_norm,none': {'kind': 'min', 'value': 0.23},
            },
        }

    def test_builds_existing_controller_dicts(self):
        inference, params, server = loader._unified_runtime_views(self.raw, self.thresholds)

        self.assertEqual(inference['container_name'], 'sglang')
        self.assertEqual(inference['nnodes'], '2')
        self.assertEqual(inference['container_config']['volume_dict']['/host-ro'], '/container-ro:ro')
        self.assertEqual(params['model'], '/models/model')
        self.assertEqual(params['inference_tests']['lm_eval_hellaswag']['tasks'], 'hellaswag')
        self.assertEqual(
            params['inference_tests']['lm_eval_hellaswag']['expected_results']['hellaswag'],
            {'acc_norm,none': 0.23},
        )
        self.assertEqual(server['env']['SGLANG_USE_AITER'], '1')

    def test_duplicate_accuracy_task_ids_raise(self):
        self.raw['accuracy']['tasks'].append(
            {
                'id': 'lm_eval_hellaswag',
                'tasks': 'hellaswag',
            }
        )

        with self.assertRaisesRegex(ValueError, 'duplicate accuracy task id'):
            loader._unified_runtime_views(self.raw, self.thresholds)


class TestUnifiedPackagedConfigs(unittest.TestCase):
    @staticmethod
    def _replace_changeme(value):
        if isinstance(value, str):
            return value.replace('<changeme>', 'node1')
        if isinstance(value, list):
            return [TestUnifiedPackagedConfigs._replace_changeme(item) for item in value]
        if isinstance(value, dict):
            return {key: TestUnifiedPackagedConfigs._replace_changeme(item) for key, item in value.items()}
        return value

    def test_all_sglang_workload_configs_load(self):
        config_dir = Path(__file__).resolve().parents[4] / 'input' / 'config_file' / 'inference' / 'sglang'
        config_paths = sorted(config_dir.glob('mi3xx_sglang_*_single.json'))
        config_paths += sorted(config_dir.glob('mi3xx_sglang_*_distributed.json'))
        config_paths += sorted(config_dir.glob('mi3xx_sglang_*_disaggregated.json'))
        self.assertEqual(len(config_paths), 6)

        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                raw = self._replace_changeme(json.loads(config_path.read_text(encoding='utf-8')))
                raw['threshold_json'] = str((config_dir / raw['threshold_json']).resolve())
                with tempfile.TemporaryDirectory() as tmp:
                    temp_config = Path(tmp) / config_path.name
                    temp_config.write_text(json.dumps(raw), encoding='utf-8')
                    variant = loader.load_variant(str(temp_config), {'username': 'test'})

                self.assertEqual(variant.framework, 'sglang')
                self.assertEqual(len(variant.accuracy.tasks), 2)
                self.assertEqual(
                    set(variant.benchmark_params['inference_tests']),
                    {'bench_serv_random', 'lm_eval_hellaswag', 'lm_eval_gsm8k'},
                )
                self.assertEqual(set(variant.params.inference_tests), {'bench_serv_random'})
                self.assertEqual(variant.params.add_flags, ['--attention-backend aiter'])
                self.assertFalse(variant.enforce_thresholds)

                if 'llama_70b_distributed' in config_path.name:
                    self.assertEqual(
                        variant.params.inference_tests['bench_serv_random']['num_prompts'],
                        '100',
                    )
                if 'deepseek' in config_path.name:
                    self.assertIn('GPU_ARCHS=gfx942', variant.params.add_export_env)
                if variant.topology == 'disaggregated':
                    self.assertEqual(variant.params.prefill_policy, 'cache_aware')
                    self.assertEqual(variant.params.decode_policy, 'cache_aware')

    def test_missing_topology_role_fails_during_load(self):
        config_dir = Path(__file__).resolve().parents[4] / 'input' / 'config_file' / 'inference' / 'sglang'
        config_path = config_dir / 'mi3xx_sglang_llama_70b_distributed.json'
        raw = self._replace_changeme(json.loads(config_path.read_text(encoding='utf-8')))
        raw['threshold_json'] = str((config_dir / raw['threshold_json']).resolve())
        del raw['roles']['server']['server_node_list']

        with tempfile.TemporaryDirectory() as tmp:
            temp_config = Path(tmp) / config_path.name
            temp_config.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'server_node_list'):
                loader.load_variant(str(temp_config), {'username': 'test'})
