'''Unit tests for Megatron-LM / Primus iteration-line parsers.'''

import math
import unittest

from cvs.lib.training.megatron.utils.iteration_metrics import (
    derive_step_time_stats,
    dialect_from_image,
    parse_iteration_metrics,
    sample_metric_curve,
    sample_training_curves,
)

MEGATRON_STEP2 = (
    '[2026-09-24 09:43:42] iteration        2/      20 | consumed samples:           32 | '
    'elapsed time per iteration (ms): 658.3 | mem usages: 0.5462 | '
    'throughput per GPU (TFLOP/s/GPU): 1441.4 | learning rate: 1.000000E-04 | '
    'global batch size:    16 | lm loss: 1.250274E+01 | loss scale: 1.0 | '
    'grad norm: 2.772 | number of skipped iterations:   0 | number of nan iterations:   0 |\n'
)

MEGATRON_MEM = (
    '[Rank 0] (after 1 iterations) memory (MB) | allocated: 70974.689453125 | '
    'max allocated: 136133.79638671875 | reserved: 137672.0 | max reserved: 137672.0\n'
)

PRIMUS_STEP2 = (
    'iteration        2/      20 | consumed samples:           32 | '
    'elapsed time per iteration (ms): 8146.7 | throughput per GPU (TFLOP/s/GPU): 103.7 | '
    'learning rate: 1.000000E-05 | global batch size:    16 | lm loss: 1.191648E+01 | '
    'loss scale: 1.0 | grad norm: 6.579 | number of skipped iterations:   0 | '
    'number of nan iterations:   0 |\n'
)

PRIMUS_STEP4 = (
    'iteration        4/      20 | consumed samples:           64 | '
    'elapsed time per iteration (ms): 654.5/735.1 | '
    'throughput per GPU (TFLOP/s/GPU): 1290.3/1162.8  | '
    'tokens/s/GPU inst/harmonic mean: 25032.8/22288.1 | '
    'learning rate: 9.698463E-06 | global batch size:    16 | lm loss: 1.101008E+01 | '
    'loss scale: 1.0 | grad norm: 19.806 | number of skipped iterations:   0 | '
    'number of nan iterations:   0 |\n'
)

PRIMUS_ANSI = (
    '\x1b[32mINFO\x1b[0m [\x1b[32m20260923 15:34:02\x1b[0m][\x1b[36mrank-7/8\x1b[0m]'
    '\x1b[34m\x1b[1m[DEBUG]    \x1b[0m\x1b[34m\x1b[1m[---------------utils.py:425] : '
    '[2026-09-23 15:34:04.269468] ' + PRIMUS_STEP4 + '\x1b[0m\n'
)


class TestDialectFromImage(unittest.TestCase):
    def test_primus_substring(self):
        self.assertEqual(dialect_from_image('rocm/primus:latest'), 'primus')

    def test_megatron_default(self):
        self.assertEqual(dialect_from_image('rocm/megatron-lm:latest'), 'megatron')
        self.assertEqual(dialect_from_image(''), 'megatron')


class TestMegatronDialect(unittest.TestCase):
    def test_step2_fields_and_derived_tokens(self):
        rows = parse_iteration_metrics(
            MEGATRON_MEM + MEGATRON_STEP2,
            'megatron',
            seq_length=8192,
            world_size=8,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['step'], 2)
        self.assertAlmostEqual(row['throughput_per_gpu'], 1441.4)
        self.assertAlmostEqual(row['learning_rate'], 1e-4)
        self.assertAlmostEqual(row['loss'], 12.50274)
        self.assertAlmostEqual(row['perplexity'], math.exp(12.50274))
        self.assertAlmostEqual(row['grad_norm'], 2.772)
        self.assertAlmostEqual(row['tokens_per_gpu'], 16 * 8192 * 1000 / (658.3 * 8))

    def test_no_tokens_without_seq_world(self):
        rows = parse_iteration_metrics(MEGATRON_STEP2, 'megatron')
        self.assertNotIn('tokens_per_gpu', rows[0])

    def test_memory_dump_is_ignored(self):
        rows = parse_iteration_metrics(MEGATRON_MEM, 'megatron')
        self.assertEqual(rows, [])


class TestPrimusDialect(unittest.TestCase):
    def test_step2_single_tflops_no_tokens(self):
        rows = parse_iteration_metrics(PRIMUS_STEP2, 'primus')
        self.assertEqual(rows[0]['step'], 2)
        self.assertAlmostEqual(rows[0]['throughput_per_gpu'], 103.7)
        self.assertNotIn('tokens_per_gpu', rows[0])
        self.assertAlmostEqual(rows[0]['grad_norm'], 6.579)

    def test_step4_uses_instantaneous_x(self):
        rows = parse_iteration_metrics(PRIMUS_STEP4, 'primus')
        self.assertAlmostEqual(rows[0]['throughput_per_gpu'], 1290.3)
        self.assertNotAlmostEqual(rows[0]['throughput_per_gpu'], 1162.8)
        self.assertAlmostEqual(rows[0]['tokens_per_gpu'], 25032.8)
        self.assertNotAlmostEqual(rows[0]['tokens_per_gpu'], 22288.1)
        self.assertAlmostEqual(rows[0]['learning_rate'], 9.698463e-6)

    def test_ansi_wrapped_line(self):
        rows = parse_iteration_metrics(PRIMUS_ANSI, 'primus')
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['tokens_per_gpu'], 25032.8)

    def test_does_not_use_megatron_token_formula(self):
        rows = parse_iteration_metrics(
            PRIMUS_STEP4,
            'primus',
            seq_length=8192,
            world_size=8,
        )
        self.assertAlmostEqual(rows[0]['tokens_per_gpu'], 25032.8)


class TestSampleMetricCurve(unittest.TestCase):
    def test_first_last_and_stride(self):
        rows = [{'step': i, 'loss': float(i)} for i in range(1, 12)]
        pts = sample_metric_curve(rows, 'loss', sample_every=5)
        # 11 observed steps → skip first int(11*0.1)=1 as warmup
        self.assertEqual(pts[0], (2, 2.0))
        self.assertEqual(pts[-1], (11, 11.0))
        self.assertIn((5, 5.0), pts)
        self.assertIn((10, 10.0), pts)

    def test_skips_first_ten_percent_as_warmup(self):
        rows = [{'step': i, 'total': 20, 'loss': float(i), 'grad_norm': float(i)} for i in range(1, 21)]
        pts = sample_metric_curve(rows, 'loss', sample_every=1)
        self.assertEqual(pts[0][0], 3)
        self.assertNotIn(1, [p[0] for p in pts])
        self.assertNotIn(2, [p[0] for p in pts])
        self.assertEqual(pts[-1][0], 20)

    def test_warmup_uses_planned_total_not_sparse_observed_steps(self):
        rows = parse_iteration_metrics(MEGATRON_STEP2, 'megatron')
        pts = sample_metric_curve(rows, 'loss', sample_every=1)
        self.assertEqual(pts, [])

    def test_dedupes_step(self):
        rows = parse_iteration_metrics(PRIMUS_STEP4 + PRIMUS_STEP4, 'primus')
        self.assertEqual(len(rows), 1)

    def test_sample_training_curves_skips_empty(self):
        rows = parse_iteration_metrics(PRIMUS_STEP4, 'primus')
        stored = sample_training_curves(rows, sample_every=10)
        self.assertIn('_loss_curve', stored)
        self.assertIn('_perplexity_curve', stored)
        self.assertIn('_learning_rate_curve', stored)
        self.assertIn('_grad_norm_curve', stored)
        self.assertIn('_throughput_curve', stored)
        self.assertIn('_tokens_curve', stored)
        self.assertEqual(stored['_loss_curve'][0][0], 4)
        self.assertAlmostEqual(stored['_perplexity_curve'][0][1], math.exp(11.01008))
        self.assertAlmostEqual(stored['_learning_rate_curve'][0][1], 9.698463e-6)


class TestDeriveStepTimeStats(unittest.TestCase):
    def test_mean_p50_p95_skip_warmup(self):
        rows = [{'step': i, 'total': 20, 'elapsed_ms': float(i)} for i in range(1, 21)]
        stats = derive_step_time_stats(rows)
        # 20 planned steps → skip first 2; remaining 3..20 (n=18)
        self.assertNotIn('step_time_mean_ms', stats)
        self.assertAlmostEqual(float(stats['step_time_p50_ms'][0]), 11.5)
        self.assertAlmostEqual(float(stats['step_time_p95_ms'][0]), 19.15)

    def test_empty_without_elapsed(self):
        self.assertEqual(derive_step_time_stats([]), {})
        self.assertEqual(derive_step_time_stats([{'step': 1, 'loss': 1.0}]), {})

    def test_primus_uses_instantaneous_elapsed_x(self):
        rows = parse_iteration_metrics(
            ''.join(
                f'iteration {i}/20 | elapsed time per iteration (ms): {100.0 + i}/999.0 | '
                f'throughput per GPU (TFLOP/s/GPU): 1.0/1.0 | lm loss: 1.0 | '
                f'learning rate: 1e-4 | global batch size: 16 | grad norm: 1.0 |\n'
                for i in range(1, 21)
            ),
            'primus',
        )
        stats = derive_step_time_stats(rows)
        # remaining elapsed X is 103..120; not the 999 Y average
        self.assertAlmostEqual(float(stats['step_time_p50_ms'][0]), 111.5)
        self.assertNotIn('999', stats['step_time_p50_ms'][0])


if __name__ == '__main__':
    unittest.main()
