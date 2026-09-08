'''Unit tests for shared vLLM metric verification stages.'''

import unittest
from types import SimpleNamespace
from unittest import mock

from _pytest.outcomes import Skipped

from cvs.lib.report import benchmark_metric_registry as registry
from cvs.tests.inference.vllm import _common


class _FakeStash(dict):
    pass


class TestVerifyCellMetrics(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()

    def test_record_only_cell_registers_rows_without_running_assertions(self):
        cell = SimpleNamespace(
            key='ISL=1024,OSL=1024,TP=8,PP=1,CONC=16',
            isl=1024,
            osl=1024,
            concurrency=16,
        )
        run = SimpleNamespace(cell=cell)
        variant = SimpleNamespace(
            model_id='org/model',
            thresholds={
                cell.key: {
                    'client.output_throughput': {'kind': 'min_tok_s', 'value': 100},
                }
            },
            enforce_thresholds=False,
        )
        result_key = _common._cell_result_key(variant, run)
        inf_res_dict = {
            result_key: {
                'head': {
                    'client.output_throughput': 99,
                    'client.mean_ttft_ms': 40,
                }
            }
        }
        node = SimpleNamespace(
            nodeid='cvs/tests/inference/vllm/vllm_single.py::test_verify_cell_metrics[cell]',
            stash=_FakeStash(),
        )
        lifecycle = SimpleNamespace(record=mock.Mock())
        subtests = SimpleNamespace(test=mock.Mock())

        with self.assertRaises(Skipped):
            _common.test_verify_cell_metrics(
                run,
                inf_res_dict,
                variant,
                lifecycle,
                SimpleNamespace(node=node),
                subtests,
            )

        rows = registry.benchmark_metric_rows_for_nodeid(node.nodeid)
        self.assertEqual([(row['metric'], row['status']) for row in rows], [('client.output_throughput', 'record')])
        subtests.test.assert_not_called()
        lifecycle.record.assert_called_once()


if __name__ == '__main__':
    unittest.main()
