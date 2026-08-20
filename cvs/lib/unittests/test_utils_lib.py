# cvs/lib/unittests/test_utils_lib.py
import os
import tempfile
import unittest
from unittest.mock import patch

import cvs.lib.utils_lib as utils_lib
from cvs.core.run_layout import RunLayout
from cvs.parsers.schemas import AortaBenchmarkConfigFile


class TestUtilsLib(unittest.TestCase):
    @patch('cvs.lib.utils_lib.fail_test')
    def test_scan_test_results_with_failure(self, mock_fail_test):
        out_dict = {'host1': 'some output test FAIL more text'}
        utils_lib.scan_test_results(out_dict)
        mock_fail_test.assert_called()

    @patch('cvs.lib.utils_lib.fail_test')
    def test_scan_test_results_no_failure(self, mock_fail_test):
        out_dict = {'host1': 'some output success'}
        utils_lib.scan_test_results(out_dict)
        mock_fail_test.assert_not_called()


class TestResolveTestConfigPlaceholdersAorta(unittest.TestCase):
    """Aorta benchmark YAML uses the same resolver as other CVS test suites (see tests/benchmark/test_aorta.py)."""

    def test_user_id_resolves_in_aorta_path(self):
        raw = {"aorta_path": "/scratch/users/{user-id}/aorta"}
        cluster = {"username": "jdoe", "home_mount_dir_name": "home", "node_dir_name": "root"}
        resolved = utils_lib.resolve_test_config_placeholders(raw, cluster)
        self.assertEqual(resolved["aorta_path"], "/scratch/users/jdoe/aorta")
        cfg = AortaBenchmarkConfigFile.model_validate(resolved)
        self.assertEqual(cfg.aorta_path, "/scratch/users/jdoe/aorta")

    def test_explicit_aorta_path_unchanged(self):
        raw = {"aorta_path": "/opt/my-aorta"}
        cluster = {"username": "jdoe", "home_mount_dir_name": "home", "node_dir_name": "root"}
        resolved = utils_lib.resolve_test_config_placeholders(raw, cluster)
        self.assertEqual(resolved["aorta_path"], "/opt/my-aorta")
        cfg = AortaBenchmarkConfigFile.model_validate(resolved)
        self.assertEqual(cfg.aorta_path, "/opt/my-aorta")


class TestResolveRunDirPlaceholder(unittest.TestCase):
    """{run_dir} comes from RunLayout, the single source of truth for the paths.

    The import is deferred into the function under test: cvs.core.run_layout pulls
    in cvs/core/__init__.py, whose orchestrator factory reaches
    cvs/core/orchestrators/baremetal.py, which imports this module back at module
    level. By call time everything is imported and the cycle is gone.
    """

    CLUSTER = {"username": "jdoe", "home_mount_dir_name": "home", "node_dir_name": "root"}

    def setUp(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_run_dir_substituted_from_the_layout(self):
        layout = RunLayout.initialize(self.tmp.name)
        raw = {"log_path": "{run_dir}/logs", "nested": ["{run_dir}/a", {"k": "{run_dir}/b"}]}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], f"{layout.run_dir}/logs")
        self.assertEqual(resolved["nested"][0], f"{layout.run_dir}/a")
        self.assertEqual(resolved["nested"][1]["k"], f"{layout.run_dir}/b")

    def test_environment_is_not_the_source_of_truth(self):
        # The layout is the one place the run directory is decided. An environment
        # that disagrees with it must not be able to redirect where artifacts land.
        layout = RunLayout.initialize(self.tmp.name)
        os.environ["CVS_RUN_DIR"] = "/somewhere/else"
        resolved = utils_lib.resolve_test_config_placeholders({"log_path": "{run_dir}/logs"}, self.CLUSTER)
        self.assertEqual(resolved["log_path"], f"{layout.run_dir}/logs")

    def test_run_dir_used_without_a_layout_exits(self):
        # Leaving the token unresolved would create a directory literally named
        # "{run_dir}" and surface as a confusing failure much later.
        raw = {"log_path": "{run_dir}/logs"}
        with self.assertRaises(SystemExit):
            utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)

    def test_config_without_placeholder_unaffected_without_a_layout(self):
        # Most test modules call this resolver; configs that never mention
        # {run_dir} must keep working with no layout resolved.
        raw = {"log_path": "/var/log/cvs", "user": "{user-id}"}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], "/var/log/cvs")
        self.assertEqual(resolved["user"], "jdoe")


if __name__ == '__main__':
    unittest.main()
