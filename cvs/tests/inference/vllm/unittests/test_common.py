'''Unit tests for shared vLLM metric verification stages.'''

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import cvs
from cvs.lib.conf_lib.loader import ConfigError
from cvs.lib.report import benchmark_metric_registry as registry
from cvs.tests.inference.vllm import _common
from cvs.unittests.support import write_filled_config

PACKAGE_DIR = Path(cvs.__file__).resolve().parent
OLD_CONFIG = PACKAGE_DIR / "config_validation/migrate/unittests/old/mi3xx_vllm_llama33-70b_fp8_distributed.json"
SHIPPED_CONFIG = PACKAGE_DIR / "input/config_file/inference/vllm/mi3xx_vllm_llama33-70b_fp8_distributed.json"


class _FakeStash(dict):
    pass


# Records only pytest.fail(); an AssertionError escapes, since bare asserts are stripped under python -O.
class _CapturingSubtests:
    def __init__(self):
        self.calls = []
        self.failures = []

    @contextmanager
    def test(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield
        except pytest.fail.Exception as exc:
            self.failures.append(str(exc))


def _metafunc(config_file):
    return SimpleNamespace(
        config=SimpleNamespace(getoption={"config_file": str(config_file)}.get),
        fixturenames=["run"],
        parametrize=mock.Mock(),
    )


class TestPytestGenerateTests(unittest.TestCase):
    def test_each_run_is_one_test_in_runs_order_and_no_cluster_file_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = write_filled_config(SHIPPED_CONFIG, tmp)
            cells = json.loads(config_file.read_text())["runs"]
            metafunc = _metafunc(config_file)
            with mock.patch("cvs.config_validation.loaders.suite_config.read_cluster") as read:
                _common.pytest_generate_tests(metafunc)
        read.assert_not_called()
        (name, runs), kwargs = metafunc.parametrize.call_args
        self.assertEqual((name, [run.cell.key for run in runs], kwargs["ids"]), ("run", cells, cells))

    def test_an_old_format_config_fails_at_collection_naming_the_migrate_command(self):
        metafunc = _metafunc(OLD_CONFIG)
        with self.assertRaises(ConfigError) as ctx:
            _common.pytest_generate_tests(metafunc)
        self.assertTrue(ctx.exception.problems[0].endswith(f"run cvs config migrate vllm {OLD_CONFIG} <new config>"))
        metafunc.parametrize.assert_not_called()


class TestInferenceCellMetrics(unittest.TestCase):
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
        host_dict = {
            'head': {
                'output_throughput': 99,
                'mean_ttft_ms': 40,
                'queue_time_p50_ms': None,
            }
        }
        node = SimpleNamespace(
            nodeid='cvs/tests/inference/vllm/vllm_single.py::test_vllm_inference[cell]',
            stash=_FakeStash(),
        )
        lifecycle = SimpleNamespace(record=mock.Mock())
        subtests = SimpleNamespace(test=mock.Mock())

        _common._verify_cell_metrics(
            host_dict,
            run,
            variant,
            lifecycle,
            SimpleNamespace(node=node),
            subtests,
        )

        rows = registry.benchmark_metric_rows_for_nodeid(node.nodeid)
        self.assertEqual(
            [(row['metric'], row['status']) for row in rows],
            [('output_throughput', 'pass'), ('mean_ttft_ms', 'pass')],
        )
        subtests.test.assert_not_called()
        lifecycle.record.assert_called_once()
        properties = [value for key, value in node.user_properties if key == _common.VLLM_JUNIT_PROPERTY]
        self.assertEqual(len(properties), 1)
        payload = json.loads(properties[0])
        self.assertEqual(payload["metric_contract"], {"id": "vllm-bare", "version": 1})
        self.assertEqual(
            payload["actuals_by_host"]["head"],
            {
                "mean_ttft_ms": 40,
                "output_throughput": 99,
                "queue_time_p50_ms": None,
            },
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
        host_dict = {
            'head': {'output_throughput': 99, 'mean_ttft_ms': 40},
            'worker': {'output_throughput': 101, 'mean_ttft_ms': 60},
        }
        node = SimpleNamespace(
            nodeid='cvs/tests/inference/vllm/vllm_single.py::test_vllm_inference[cell]',
            stash=_FakeStash(),
            user_properties=[],
        )
        lifecycle = SimpleNamespace(record=mock.Mock())
        subtests = _CapturingSubtests()

        _common._verify_cell_metrics(
            host_dict,
            run,
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
        self.assertEqual(
            subtests.failures,
            ['output_throughput: actual 99 < min 100', 'mean_ttft_ms: actual 60 > max 50'],
        )
        keys = [key for key, _value in node.user_properties]
        self.assertEqual(keys.count(_common.VLLM_JUNIT_PROPERTY), 1)


if __name__ == '__main__':
    unittest.main()
