'''Unit tests for cvs/lib/inference/sglang/sglang_config_loader.py.'''

import os
import unittest
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
