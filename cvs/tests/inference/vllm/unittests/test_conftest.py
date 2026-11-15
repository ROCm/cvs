'''Unit tests for the vLLM suite conftest: its schema and orch wiring, and its pytest-html metric hooks.'''

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cvs.config_validation.schemas.inference.engines import VllmConfig
from cvs.core.orchestrators.factory import OrchestratorFactory
from cvs.lib.report import benchmark_metric_registry as registry
from cvs.lib.report.render.perf_metric_table import is_benchmark_metrics_extra
from cvs.tests import conftest as root_conftest
from cvs.tests.inference.vllm import conftest as vllm_conftest
from cvs.unittests.support import write_filled_config

CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
# The cluster file's head is not its first node, so a missed head rewrite for vllm_single shows.
CLUSTER = {
    "username": "u",
    "priv_key_file": "/keys/id",
    "node_dict": {"n1": {"vpc_ip": "v1"}, "n0": {"vpc_ip": "v0"}},
    "head_node_dict": {"mgmt_ip": "n0"},
}


class TestVllmOrch(unittest.TestCase):
    """The suite's orch comes from the root fixture through VllmConfig and launches nothing itself."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        cluster_file = self.tmp / "cluster.json"
        cluster_file.write_text(json.dumps(CLUSTER))
        self.options = {"cluster_file": str(cluster_file)}
        self.pytestconfig = mock.Mock()
        self.pytestconfig.getoption.side_effect = lambda name: self.options[name]
        self.orch = mock.MagicMock(orchestrator_type="container")
        patcher = mock.patch.object(OrchestratorFactory, "create_orchestrator", return_value=self.orch)
        self.create = patcher.start()
        self.addCleanup(patcher.stop)

    def _check_orch(self, module, config_name, hosts, head):
        schema = vllm_conftest.config_schema.__wrapped__()
        self.options["config_file"] = str(write_filled_config(CONFIG_DIR / config_name, self.tmp))
        loaded = root_conftest.suite_config.__wrapped__(self.pytestconfig, schema)
        lifecycle = vllm_conftest._Lifecycle()
        request = SimpleNamespace(
            module=SimpleNamespace(__name__=f"cvs.tests.inference.vllm.{module}"),
            getfixturevalue=mock.Mock(return_value=lifecycle),
        )
        mode = root_conftest.suite_mode.__wrapped__(request, schema)
        gen = root_conftest.orch.__wrapped__(request, self.pytestconfig, loaded, mode)

        self.assertIs(next(gen), self.orch)
        (_, cfg), _ = self.create.call_args
        self.assertEqual((list(cfg.node_dict), cfg.head_node_dict["mgmt_ip"]), (hosts, head))
        self.assertEqual(cfg.orchestrator, "container")
        self.assertIs(self.orch.suite_config, loaded.config)
        request.getfixturevalue.assert_called_once_with("lifecycle")
        # The launch and sshd setup are tests of their own, so the fixture calls nothing before it yields.
        self.assertEqual(self.orch.method_calls, [])

        with self.assertRaises(StopIteration):
            next(gen)
        self.assertEqual(self.orch.method_calls, [mock.call.teardown_containers(), mock.call.close()])

    def test_config_schema_is_vllm_config(self):
        self.assertIs(vllm_conftest.config_schema.__wrapped__(), VllmConfig)

    def test_vllm_single_runs_on_the_first_node_as_head(self):
        self._check_orch("vllm_single", "mi3xx_vllm_llama33-70b_fp8_single.json", ["n1"], "n1")

    def test_vllm_distributed_keeps_every_node_and_the_cluster_head(self):
        self._check_orch("vllm_distributed", "mi3xx_vllm_llama33-70b_fp8_distributed.json", ["n1", "n0"], "n0")


class _FakeStash(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Outcome:
    def __init__(self, report):
        self.report = report

    def get_result(self):
        return self.report


class TestVllmMetricReportHook(unittest.TestCase):
    def setUp(self):
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()

    def test_lifecycle_starts_without_live_server_state(self):
        lifecycle = vllm_conftest._Lifecycle()

        self.assertIsNone(lifecycle.live_server_sig)
        self.assertIsNone(lifecycle.live_server_job)
        self.assertIsNone(lifecycle.model_load_s)
        self.assertIsNone(lifecycle.model_load_memory_mb)

    def test_collection_order_keeps_teardown_last(self):
        names = [
            "test_teardown",
            "test_accuracy_eval",
            "test_vllm_inference",
            "test_launch_container",
        ]
        items = [SimpleNamespace(originalname=name, name=name) for name in names]

        with mock.patch.object(vllm_conftest, "validate_vllm_execution_mode"):
            vllm_conftest.pytest_collection_modifyitems(SimpleNamespace(), items)

        self.assertEqual(
            [item.originalname for item in items],
            [
                "test_launch_container",
                "test_vllm_inference",
                "test_accuracy_eval",
                "test_teardown",
            ],
        )

    def test_makereport_attaches_and_stamps_metric_panel(self):
        nodeid = 'cvs/tests/inference/vllm/vllm_single.py::test_vllm_inference[cell]'
        item = SimpleNamespace(nodeid=nodeid, stash=_FakeStash())
        rows = [
            {
                'node': 'head',
                'metric': 'output_throughput',
                'status': 'record',
                'actual': 99,
                'spec': {'kind': 'min', 'value': 100},
                'enforced': False,
            }
        ]
        registry.record_benchmark_metric_rows(item, rows)
        full_log = {'format_type': 'url', 'name': 'Full Log', 'content': 'test_html/full.html'}
        report = SimpleNamespace(
            nodeid=nodeid,
            when='call',
            extras=[full_log, dict(full_log)],
            user_properties=[],
        )

        hook = vllm_conftest.pytest_runtest_makereport(item, None)
        next(hook)
        with self.assertRaises(StopIteration):
            hook.send(_Outcome(report))

        self.assertEqual(sum(is_benchmark_metrics_extra(extra) for extra in report.extras), 1)
        self.assertEqual(sum(extra.get('name') == 'Full Log' for extra in report.extras), 1)
        self.assertEqual(
            dict(report.user_properties)[registry.BENCHMARK_METRIC_ROWS_USER_PROPERTY],
            rows,
        )


if __name__ == '__main__':
    unittest.main()
