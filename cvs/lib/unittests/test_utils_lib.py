# cvs/lib/unittests/test_utils_lib.py
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

    def test_get_model_from_rocm_smi_output_matches_marketing_name(self):
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI300X'), 'mi300x')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI325X'), 'mi325')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI350X'), 'mi350')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI355X'), 'mi355')

    def test_get_model_from_rocm_smi_output_falls_back_to_device_id_for_mi350(self):
        smi_output = 'Device Name:        AMD Radeon Graphics\nDevice ID:          0x75a0\nGFX Version:        gfx950\n'
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output(smi_output), 'mi350')

    def test_get_model_from_rocm_smi_output_defaults_to_mi300x_when_unrecognized(self):
        smi_output = 'Device Name:        AMD Radeon Graphics\nDevice ID:          0x1234\n'
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output(smi_output), 'mi300x')


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
