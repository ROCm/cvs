# cvs/lib/unittests/test_utils_lib.py
import os
import unittest
from unittest.mock import patch

import cvs
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

    def test_cvs_package_root_matches_cvs_module_directory(self):
        expected = os.path.dirname(os.path.abspath(cvs.__file__))
        self.assertEqual(utils_lib.cvs_package_root(), expected)

    def test_cvs_package_root_contains_input_dir(self):
        self.assertTrue(os.path.isdir(os.path.join(utils_lib.cvs_package_root(), 'input')))


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


if __name__ == '__main__':
    unittest.main()
