'''Unit tests for shared vLLM metric verification stages.'''

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from _pytest.outcomes import Skipped

from cvs.lib.report import benchmark_metric_registry as registry
from cvs.tests.inference.vllm import _common


class _FakeStash(dict):
    pass


class _CapturingSubtests:
    def __init__(self):
        self.calls = []
        self.failures = []

    @contextmanager
    def test(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield
        except AssertionError as exc:
            self.failures.append(str(exc))


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
                    'output_throughput': {'kind': 'min', 'value': 100},
                }
            },
            enforce_thresholds=False,
        )
        result_key = _common._cell_result_key(variant, run)
        inf_res_dict = {
            result_key: {
                'head': {
                    'output_throughput': 99,
                    'mean_ttft_ms': 40,
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
        self.assertEqual(
            [(row['metric'], row['status']) for row in rows],
            [('output_throughput', 'record'), ('mean_ttft_ms', 'record')],
        )
        subtests.test.assert_not_called()
        lifecycle.record.assert_called_once()
        properties = [
            value
            for key, value in node.user_properties
            if key == _common.VLLM_JUNIT_PROPERTY
        ]
        self.assertEqual(len(properties), 1)
        payload = json.loads(properties[0])
        self.assertEqual(payload["metric_contract"], {"id": "vllm-bare", "version": 1})
        self.assertEqual(
            payload["actuals_by_host"]["head"],
            {"mean_ttft_ms": 40, "output_throughput": 99},
        )

    def test_mixed_pass_fail_siblings_and_multiple_hosts_all_run(self):
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
                    'output_throughput': {'kind': 'min', 'value': 100},
                    'mean_ttft_ms': {'kind': 'max', 'value': 50},
                }
            },
            enforce_thresholds=True,
        )
        result_key = _common._cell_result_key(variant, run)
        inf_res_dict = {
            result_key: {
                'head': {'output_throughput': 99, 'mean_ttft_ms': 40},
                'worker': {'output_throughput': 101, 'mean_ttft_ms': 60},
            }
        }
        node = SimpleNamespace(
            nodeid='cvs/tests/inference/vllm/vllm_single.py::test_verify_cell_metrics[cell]',
            stash=_FakeStash(),
            user_properties=[],
        )
        lifecycle = SimpleNamespace(record=mock.Mock())
        subtests = _CapturingSubtests()

        _common.test_verify_cell_metrics(
            run,
            inf_res_dict,
            variant,
            lifecycle,
            SimpleNamespace(node=node),
            subtests,
        )

        self.assertEqual(
            subtests.calls,
            [
                {'node': 'head', 'metric': 'output_throughput'},
                {'node': 'head', 'metric': 'mean_ttft_ms'},
                {'node': 'worker', 'metric': 'output_throughput'},
                {'node': 'worker', 'metric': 'mean_ttft_ms'},
            ],
        )
        self.assertEqual(len(subtests.failures), 2)
        keys = [key for key, _value in node.user_properties]
        self.assertEqual(keys.count(_common.VLLM_JUNIT_PROPERTY), 1)


if __name__ == '__main__':
    unittest.main()
