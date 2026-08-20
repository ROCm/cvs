# cvs/lib/unittests/test_utils_lib.py
import os
import unittest
from unittest.mock import patch

import cvs.lib.utils_lib as utils_lib
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
    """{run_dir} comes from CVS_RUN_DIR, which cvs run exports from RunLayout.

    utils_lib deliberately reads the environment rather than importing
    cvs.core.run_layout: cvs/core/__init__.py imports the orchestrator factory,
    which imports this module back, so a module-level import would be circular.
    """

    CLUSTER = {"username": "jdoe", "home_mount_dir_name": "home", "node_dir_name": "root"}

    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("CVS_RUN_DIR", None)

    def test_run_dir_substituted_from_env(self):
        os.environ["CVS_RUN_DIR"] = "/shared/cvs/runs/4242"
        raw = {"log_path": "{run_dir}/logs", "nested": ["{run_dir}/a", {"k": "{run_dir}/b"}]}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], "/shared/cvs/runs/4242/logs")
        self.assertEqual(resolved["nested"][0], "/shared/cvs/runs/4242/a")
        self.assertEqual(resolved["nested"][1]["k"], "/shared/cvs/runs/4242/b")

    def test_run_dir_used_without_env_exits(self):
        # Leaving the token unresolved would create a directory literally named
        # "{run_dir}" and surface as a confusing failure much later.
        raw = {"log_path": "{run_dir}/logs"}
        with self.assertRaises(SystemExit):
            utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)

    def test_empty_env_var_is_treated_as_unset(self):
        # Substituting '' would silently turn "{run_dir}/logs" into "/logs" and
        # write at the filesystem root, which succeeds when running as root.
        os.environ["CVS_RUN_DIR"] = ""
        raw = {"log_path": "{run_dir}/logs"}
        with self.assertRaises(SystemExit):
            utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)

    def test_config_without_placeholder_unaffected_when_env_missing(self):
        # Most test modules call this resolver; configs that never mention
        # {run_dir} must keep working with no CVS_RUN_DIR set.
        raw = {"log_path": "/var/log/cvs", "user": "{user-id}"}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], "/var/log/cvs")
        self.assertEqual(resolved["user"], "jdoe")


if __name__ == '__main__':
    unittest.main()
