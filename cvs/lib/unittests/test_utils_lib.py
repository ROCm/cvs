# cvs/lib/unittests/test_utils_lib.py
import os
import shlex
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

    @patch('cvs.lib.utils_lib.fail_test')
    def test_scan_test_results_cpuset_allocation_abort(self, mock_fail_test):
        out_dict = {
            'host1': 'Transfer 0: DST 0: CPU 2 on rank 0 cannot allocate memory due to process memory policy/cpuset'
        }
        utils_lib.scan_test_results(out_dict)
        mock_fail_test.assert_called()

    def test_cluster_target_output_label_strips_and_sanitizes(self):
        self.assertEqual(utils_lib.cluster_target_output_label("  node1.example.com  "), "node1.example.com")
        self.assertEqual(utils_lib.cluster_target_output_label("a/b"), "a_b")
        self.assertEqual(utils_lib.cluster_target_output_label(""), "unknown_node")

    def test_get_model_from_rocm_smi_output_matches_marketing_name(self):
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI300X'), 'mi300x')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI325X'), 'mi325')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI350X'), 'mi350')
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output('Card series: AMD Instinct MI355X'), 'mi355')

    def test_get_model_from_rocm_smi_output_falls_back_to_device_id_for_mi350(self):
        smi_output = 'Device Name:        AMD Radeon Graphics\nDevice ID:          0x75a0\nGFX Version:        gfx950\n'
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output(smi_output), 'mi350')

    def test_get_model_from_rocm_smi_output_falls_back_to_device_id_for_mi355(self):
        smi_output = 'Device Name:        AMD Radeon Graphics\nDevice ID:          0x75a3\nGFX Version:        gfx950\n'
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output(smi_output), 'mi355')

    def test_get_model_from_rocm_smi_output_defaults_to_mi300x_when_unrecognized(self):
        smi_output = 'Device Name:        AMD Radeon Graphics\nDevice ID:          0x1234\n'
        self.assertEqual(utils_lib.get_model_from_rocm_smi_output(smi_output), 'mi300x')

    def test_wan_hf_snapshot_offline_check_commands_paths_quoted(self):
        snap_root = '/data/my hf cache/snapshots/abc123'
        cmds = utils_lib.wan_hf_snapshot_offline_check_commands(snap_root)
        self.assertIn('configuration.json', cmds)
        self.assertIn('low_noise diffusion shards (6 x >500MiB)', cmds)
        quoted_cfg = shlex.quote(os.path.join(snap_root, 'configuration.json'))
        self.assertIn(quoted_cfg, cmds['configuration.json'])
        for label, cmd in cmds.items():
            self.assertIn('OK', cmd, msg=label)
            self.assertIn('MISSING', cmd, msg=label)


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
        layout = RunLayout.get(self.tmp.name)
        raw = {"log_path": "{run_dir}/logs", "nested": ["{run_dir}/a", {"k": "{run_dir}/b"}]}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], f"{layout.run_dir}/logs")
        self.assertEqual(resolved["nested"][0], f"{layout.run_dir}/a")
        self.assertEqual(resolved["nested"][1]["k"], f"{layout.run_dir}/b")

    def test_environment_is_not_the_source_of_truth(self):
        # The layout is the one place the run directory is decided. An environment
        # that disagrees with it must not be able to redirect where artifacts land.
        layout = RunLayout.get(self.tmp.name)
        os.environ["CVS_RUN_DIR"] = "/somewhere/else"
        resolved = utils_lib.resolve_test_config_placeholders({"log_path": "{run_dir}/logs"}, self.CLUSTER)
        self.assertEqual(resolved["log_path"], f"{layout.run_dir}/logs")

    def test_run_dir_resolves_a_layout_when_none_exists_yet(self):
        # Leaving the token unresolved would create a directory literally named
        # "{run_dir}"; RunLayout.get() resolves one rather than substituting nothing.
        os.environ["CVS_WORKSPACE"] = self.tmp.name
        resolved = utils_lib.resolve_test_config_placeholders({"log_path": "{run_dir}/logs"}, self.CLUSTER)
        self.assertEqual(resolved["log_path"], f"{RunLayout.get().run_dir}/logs")

    def test_config_without_placeholder_resolves_no_layout(self):
        # Roughly half the test modules call this resolver and most of their configs
        # never mention {run_dir}. Reaching for the layout regardless would create a
        # run directory as a side effect of resolving an unrelated config.
        raw = {"log_path": "/var/log/cvs", "user": "{user-id}"}
        resolved = utils_lib.resolve_test_config_placeholders(raw, self.CLUSTER)
        self.assertEqual(resolved["log_path"], "/var/log/cvs")
        self.assertEqual(resolved["user"], "jdoe")
        self.assertIsNone(RunLayout._instance)


if __name__ == '__main__':
    unittest.main()
