'''Unit tests for TorchTitan per-step log parsing.'''

import math
import unittest

from cvs.lib.training.torchtitan.utils.iteration_metrics import (
    derive_step_time_stats,
    parse_iteration_metrics,
    sample_metric_curve,
    sample_training_curves,
    summarize_step_metrics,
)
from cvs.lib.training.torchtitan.utils.torchtitan_metrics import METRIC_UNITS, tier_metric_specs

STEP = (
    'step:  {step}  loss:  {loss:.5f}  grad_norm:  {grad:.4f}  '
    'memory: {mem:.2f}GiB(55.00%)  tps: {tps}  tflops: {tflops:.2f}  mfu: 20.00%\n'
)


def _log(n=10, loss0=10.0):
    lines = []
    for step in range(1, n + 1):
        lines.append(
            STEP.format(
                step=step,
                loss=loss0 - step * 0.1,
                grad=1.0 + step,
                mem=40.0 + step,
                tps=f'{1000 * step:,}',
                tflops=100.0 + step,
            )
        )
    return ''.join(lines)


class TestParseIterationMetrics(unittest.TestCase):
    def test_colored_line_fields_and_perplexity(self):
        raw = (
            '\x1b[31mstep:  4 \x1b[32mloss:  8.50000 \x1b[33mgrad_norm:  1.2500 '
            '\x1b[36mmemory: 45.50GiB(60.00%) \x1b[34mtps: 12,345 '
            '\x1b[36mtflops: 234.56 \x1b[35mmfu: N/A\x1b[0m\n'
        )
        rows = parse_iteration_metrics(raw)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['step'], 4)
        self.assertAlmostEqual(row['loss'], 8.5)
        self.assertAlmostEqual(row['perplexity'], math.exp(8.5))
        self.assertAlmostEqual(row['grad_norm'], 1.25)
        self.assertAlmostEqual(row['memory_gib'], 45.5)
        self.assertAlmostEqual(row['tps'], 12345)
        self.assertAlmostEqual(row['tflops'], 234.56)
        self.assertNotIn('mfu', row)

    def test_pipe_format_and_tok_per_s_alias(self):
        raw = 'step: 3 | loss: 1.5 | tok/s: 1,000 | mem: 12.0 GB\n'
        row = parse_iteration_metrics(raw)[0]
        self.assertEqual(row['step'], 3)
        self.assertAlmostEqual(row['loss'], 1.5)
        self.assertAlmostEqual(row['tps'], 1000)
        self.assertAlmostEqual(row['memory_gib'], 12.0)

    def test_ignores_step_without_loss_or_tps(self):
        self.assertEqual(parse_iteration_metrics('step: 1 loading checkpoint\n'), [])

    def test_duplicate_step_keeps_last(self):
        raw = 'step: 2  loss: 3.0  tps: 10\nstep: 2  loss: 2.0  tps: 20\n'
        rows = parse_iteration_metrics(raw)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['loss'], 2.0)

    def test_elapsed_from_tps(self):
        raw = 'step: 2  loss: 2.0  tps: 8,000\n'
        row = parse_iteration_metrics(raw, seq_length=8192, global_batch_size=48, world_size=8)[0]
        self.assertAlmostEqual(row['elapsed_ms'], 48 * 8192 / 8 / 8000 * 1000.0)

    def test_no_elapsed_without_shape(self):
        row = parse_iteration_metrics('step: 2  loss: 2.0  tps: 8000\n')[0]
        self.assertNotIn('elapsed_ms', row)

    def test_explicit_time_overrides_tps_estimate(self):
        raw = 'step: 2  loss: 2.0  tps: 8000  time: 1.5s\n'
        row = parse_iteration_metrics(raw, seq_length=8192, global_batch_size=48, world_size=8)[0]
        self.assertAlmostEqual(row['elapsed_ms'], 1500.0)


class TestSampleAndSummarize(unittest.TestCase):
    def test_warmup_and_stride(self):
        rows = parse_iteration_metrics(_log(20))
        pts = sample_metric_curve(rows, 'loss', sample_every=5)
        steps = [step for step, _ in pts]
        self.assertNotIn(1, steps)
        self.assertNotIn(2, steps)
        self.assertEqual(steps[0], 3)
        self.assertIn(5, steps)
        self.assertEqual(steps[-1], 20)

    def test_sample_training_curves(self):
        rows = parse_iteration_metrics(_log(4))
        stored = sample_training_curves(rows, sample_every=1)
        self.assertIn('_loss_curve', stored)
        self.assertIn('_perplexity_curve', stored)
        self.assertIn('_tps_curve', stored)
        self.assertIn('_tflops_curve', stored)
        self.assertIn('_mfu_curve', stored)

    def test_summarize_skips_warmup(self):
        rows = parse_iteration_metrics(_log(10, loss0=10.0))
        summary = summarize_step_metrics(rows)
        # steps 2..10, loss = 10 - 0.1*step
        expected_loss = sum(10.0 - 0.1 * step for step in range(2, 11)) / 9
        self.assertAlmostEqual(float(summary['loss'][0]), expected_loss)
        self.assertAlmostEqual(float(summary['tokens_per_sec'][0]), sum(1000 * step for step in range(2, 11)) / 9)
        self.assertIn('mem_usage_gb', summary)
        self.assertIn('tflops', summary)

    def test_step_time_percentiles(self):
        rows = [{'step': i, 'total': 20, 'elapsed_ms': float(i)} for i in range(1, 21)]
        stats = derive_step_time_stats(rows)
        self.assertAlmostEqual(float(stats['step_time_p50_ms'][0]), 11.5)
        self.assertAlmostEqual(float(stats['step_time_p95_ms'][0]), 19.15)

    def test_step_time_empty_without_elapsed(self):
        self.assertEqual(derive_step_time_stats([{'step': 1, 'total': 1, 'loss': 1.0}]), {})


class TestTorchTitanMetrics(unittest.TestCase):
    def test_unknown_tier_empty(self):
        self.assertEqual(tier_metric_specs({'training.tokens_per_sec': {'kind': 'min', 'value': 1}}, 'nope'), {})

    def test_throughput_tier_filters_cell(self):
        cell = {
            'training.tokens_per_sec': {'kind': 'min', 'value': 1},
            'training.loss': {'kind': 'max', 'value': 15},
        }
        specs = tier_metric_specs(cell, 'throughput')
        self.assertIn('training.tokens_per_sec', specs)
        self.assertNotIn('training.loss', specs)

    def test_units(self):
        self.assertEqual(METRIC_UNITS['tokens_per_sec'], 'tok/s/device')
        self.assertEqual(METRIC_UNITS['step_time_p95_ms'], 'ms')


if __name__ == '__main__':
    unittest.main()
