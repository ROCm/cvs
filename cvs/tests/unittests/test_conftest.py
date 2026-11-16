'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the root fixtures in cvs/tests/conftest.py: without a config
schema, orch runs exactly as before schemas existed; with one, it loads,
narrows, builds, launches and tears down through the schema's hooks.
'''

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic import create_model

import cvs
from cvs.config_validation.loaders.suite_config import LoadedSuiteConfig
from cvs.config_validation.schemas.base import SuiteConfig
from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.tests import conftest
from cvs.unittests.support import run_trusted

NODES = {"n0": {"vpc_ip": "v0"}, "n1": {"vpc_ip": "v1"}, "n2": {"vpc_ip": "v2"}}
CLUSTER = {"username": "u", "priv_key_file": "/keys/id", "node_dict": NODES, "head_node_dict": {"mgmt_ip": "n0"}}
CONTAINER = {"image": "img", "name": "ct", "runtime": {"name": "docker", "args": {}}}
SUITE = {"_comment": "notes", "name": "demo", "orchestrator": "container", "container": CONTAINER}
SCHEMA_ONLY_MODULES = ("pydantic", "cvs.config_validation", "cvs.lib.conf_lib", "cvs.lib.utils.config_loader")


class _DemoHooks(SuiteConfig):
    modes = {"_single": "single", "_distributed": "distributed"}

    @classmethod
    def select_hosts(cls, config, cluster, mode):
        return [next(iter(cluster["node_dict"]))] if mode == "single" else None

    @classmethod
    def after_orch(cls, orch, config, lifecycle):
        orch.after_orch_hook(config, lifecycle)


_Demo = create_model(
    "_Demo", __base__=_DemoHooks, name=(str, ...), orchestrator=(str, "baremetal"), container=(dict, {})
)


class _LaunchAsTests(_Demo):
    launch_in_fixture = False

    @classmethod
    def teardown(cls, orch, config, lifecycle):
        orch.teardown_hook(config, lifecycle)
        super().teardown(orch, config, lifecycle)


def _request(module="cvs.tests.demo.demo_single", lifecycle=None):
    return SimpleNamespace(module=SimpleNamespace(__name__=module), getfixturevalue=mock.Mock(return_value=lifecycle))


class _FixtureCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cluster_file = self._write("cluster.json", CLUSTER)
        self.config_file = self._write("suite.json", SUITE)
        self.options = {"cluster_file": self.cluster_file, "config_file": self.config_file}
        self.pytestconfig = mock.Mock()
        self.pytestconfig.getoption.side_effect = lambda name: self.options[name]
        self.orch = mock.MagicMock(orchestrator_type="container", container_config={"image": "img"})
        patcher = mock.patch.object(OrchestratorFactory, "create_orchestrator", return_value=self.orch)
        self.create = patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data))
        return str(path)

    def _cfg(self):
        (log, cfg), _ = self.create.call_args
        self.assertIs(log, conftest.log)
        return cfg

    def _finish(self, gen):
        with self.assertRaises(StopIteration):
            next(gen)


class TestWithoutSchema(_FixtureCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(OrchestratorConfig, "from_configs", wraps=OrchestratorConfig.from_configs)
        self.from_configs = patcher.start()
        self.addCleanup(patcher.stop)
        self.request = _request()

    def _orch(self):
        return conftest.orch.__wrapped__(self.request, self.pytestconfig, None, None)

    def test_schema_fixtures_return_none_and_read_no_option(self):
        self.assertIsNone(conftest.config_schema.__wrapped__())
        self.assertIsNone(conftest.suite_mode.__wrapped__(self.request, None))
        self.assertIsNone(conftest.suite_config.__wrapped__(self.pytestconfig, None))
        self.pytestconfig.getoption.assert_not_called()

    def test_importing_the_root_conftest_loads_no_schema_code(self):
        # A schema module that fails to import must not break collection for suites that never set a schema.
        code = "import sys, cvs.tests.conftest; print('\\n'.join(sorted(sys.modules)))"
        env = {**os.environ, "PYTHONPATH": str(Path(cvs.__file__).resolve().parents[1])}
        loaded = run_trusted([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        modules = loaded.stdout.split()
        self.assertIn("cvs.core.orchestrators.factory", modules)
        for prefix in SCHEMA_ONLY_MODULES:
            with self.subTest(prefix=prefix):
                self.assertEqual([m for m in modules if m == prefix or m.startswith(f"{prefix}.")], [])

    def test_a_container_cluster_is_built_from_both_files_launched_yielded_then_torn_down_and_closed(self):
        self.options["cluster_file"] = self._write("cluster.json", {**CLUSTER, "orchestrator": "container"})
        self.options["config_file"] = self._write("suite.json", {"health": {"x": 1}, "container": CONTAINER})
        gen = self._orch()
        self.assertIs(next(gen), self.orch)
        self.from_configs.assert_called_once_with(self.options["cluster_file"], self.options["config_file"])
        cfg = self._cfg()
        self.assertEqual((cfg.orchestrator, cfg.container["image"]), ("container", "img"))
        self.assertEqual(self.orch.method_calls, [mock.call.setup_containers(), mock.call.setup_sshd()])
        self._finish(gen)
        self.assertEqual(self.orch.method_calls[2:], [mock.call.teardown_containers(), mock.call.close()])
        self.request.getfixturevalue.assert_not_called()

    def test_a_baremetal_cluster_is_only_built_and_closed(self):
        self.options["config_file"] = self._write("suite.json", {"health": {"x": 1}})
        gen = self._orch()
        self.assertIs(next(gen), self.orch)
        self.assertEqual(self._cfg().orchestrator, "baremetal")
        self.assertEqual(self.orch.method_calls, [])
        self._finish(gen)
        self.assertEqual(self.orch.method_calls, [mock.call.close()])

    def test_the_launch_decision_reads_the_config_not_the_orchestrator(self):
        # A container orchestrator built from a baremetal config is not launched, as before schemas.
        self.options["config_file"] = self._write("suite.json", {"orchestrator": "baremetal"})
        gen = self._orch()
        next(gen)
        self._finish(gen)
        self.assertEqual(self.orch.method_calls, [mock.call.close()])

    def test_a_failed_launch_fails_naming_the_container_without_sshd_teardown_or_close(self):
        # As before schemas, a failed launch raises before the fixture's try, so nothing is torn down or closed.
        self.orch.setup_containers.return_value = False
        self.orch.get_container_name.return_value = "u_img"
        with self.assertRaises(pytest.fail.Exception) as ctx:
            next(self._orch())
        self.assertEqual(str(ctx.exception), "Failed to launch container : u_img")
        self.orch.get_container_name.assert_called_once_with({"image": "img"}, "img")
        self.orch.setup_sshd.assert_not_called()
        self.orch.teardown_containers.assert_not_called()
        self.orch.close.assert_not_called()

    def test_a_failed_sshd_setup_fails_without_teardown_or_close(self):
        self.orch.setup_sshd.return_value = False
        with self.assertRaisesRegex(pytest.fail.Exception, "^Failed to setup sshd in container$"):
            next(self._orch())
        self.orch.teardown_containers.assert_not_called()
        self.orch.close.assert_not_called()

    def test_a_missing_file_option_fails_before_anything_is_read(self):
        for missing in ("cluster_file", "config_file"):
            with self.subTest(missing=missing), mock.patch.dict(self.options, {missing: None}):
                with self.assertRaisesRegex(pytest.fail.Exception, "requires --cluster_file and --config_file"):
                    next(self._orch())
        self.from_configs.assert_not_called()

    def test_close_runs_when_teardown_fails(self):
        self.orch.teardown_containers.side_effect = RuntimeError("teardown")
        gen = self._orch()
        next(gen)
        with self.assertRaisesRegex(RuntimeError, "teardown"):
            next(gen)
        self.orch.close.assert_called_once_with()


class TestSchemaFixtures(_FixtureCase):
    def test_suite_config_loads_both_files_through_the_schema(self):
        loaded = conftest.suite_config.__wrapped__(self.pytestconfig, _Demo)
        self.assertIsInstance(loaded, LoadedSuiteConfig)
        self.assertIs(loaded.schema, _Demo)
        self.assertEqual((loaded.config.name, loaded.cluster, loaded.orch_inputs), ("demo", CLUSTER, SUITE))

    def test_suite_mode_maps_the_module_suffix(self):
        self.assertEqual(conftest.suite_mode.__wrapped__(_request("cvs.tests.demo.demo_single"), _Demo), "single")

    def test_suite_mode_fails_naming_a_module_that_matches_no_mode(self):
        with self.assertRaises(pytest.fail.Exception) as ctx:
            conftest.suite_mode.__wrapped__(_request("cvs.tests.demo.demo_other"), _Demo)
        self.assertIn("cvs.tests.demo.demo_other matches no _Demo mode", str(ctx.exception))


class TestOrchWithSchema(_FixtureCase):
    def _orch(self, schema, mode="single", lifecycle=None):
        self.request = _request(lifecycle=lifecycle)
        self.loaded = conftest.suite_config.__wrapped__(self.pytestconfig, schema)
        return conftest.orch.__wrapped__(self.request, self.pytestconfig, self.loaded, mode)

    def test_a_fixture_launch_narrows_hosts_builds_with_the_suite_config_launches_and_tears_down(self):
        gen = self._orch(_Demo)
        self.assertIs(next(gen), self.orch)
        cfg = self._cfg()
        self.assertEqual((cfg.node_dict, cfg.head_node_dict), ({"n0": NODES["n0"]}, {"mgmt_ip": "n0"}))
        self.assertEqual((cfg.orchestrator, cfg.container["image"]), ("container", "img"))
        self.assertIs(self.orch.suite_config, self.loaded.config)
        self.assertEqual(
            self.orch.method_calls,
            [mock.call.after_orch_hook(self.loaded.config, None), mock.call.setup_containers(), mock.call.setup_sshd()],
        )
        self.request.getfixturevalue.assert_not_called()
        self._finish(gen)
        self.assertEqual(self.orch.method_calls[3:], [mock.call.teardown_containers(), mock.call.close()])

    def test_launch_as_tests_skips_the_fixture_launch_and_hands_the_lifecycle_to_the_hooks(self):
        lifecycle = object()
        gen = self._orch(_LaunchAsTests, lifecycle=lifecycle)
        self.assertIs(next(gen), self.orch)
        self.request.getfixturevalue.assert_called_once_with("lifecycle")
        self.assertEqual(self.orch.method_calls, [mock.call.after_orch_hook(self.loaded.config, lifecycle)])
        self._finish(gen)
        self.assertEqual(
            self.orch.method_calls[1:],
            [
                mock.call.teardown_hook(self.loaded.config, lifecycle),
                mock.call.teardown_containers(),
                mock.call.close(),
            ],
        )

    def test_the_launch_decision_reads_the_orchestrator_type(self):
        self.orch.orchestrator_type = "baremetal"
        gen = self._orch(_Demo)
        next(gen)
        self._finish(gen)
        self.assertEqual(
            self.orch.method_calls, [mock.call.after_orch_hook(self.loaded.config, None), mock.call.close()]
        )

    def test_a_failed_launch_fails_and_still_closes_without_teardown(self):
        self.orch.setup_containers.return_value = False
        self.orch.get_container_name.return_value = "u_img"
        with self.assertRaisesRegex(pytest.fail.Exception, "^Failed to launch container : u_img$"):
            next(self._orch(_Demo))
        self.orch.setup_sshd.assert_not_called()
        self.orch.teardown_containers.assert_not_called()
        self.orch.close.assert_called_once_with()

    def test_close_runs_when_after_orch_fails_and_nothing_is_launched(self):
        self.orch.after_orch_hook.side_effect = RuntimeError("after_orch")
        with self.assertRaisesRegex(RuntimeError, "after_orch"):
            next(self._orch(_Demo))
        self.orch.setup_containers.assert_not_called()
        self.orch.close.assert_called_once_with()

    def test_close_runs_when_teardown_fails(self):
        self.orch.teardown_containers.side_effect = RuntimeError("teardown")
        gen = self._orch(_LaunchAsTests, lifecycle=object())
        next(gen)
        with self.assertRaisesRegex(RuntimeError, "teardown"):
            next(gen)
        self.orch.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
