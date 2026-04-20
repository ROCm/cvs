# cvs/lib/unittests/test_utils_lib.py
import os
import shlex
import unittest
from unittest.mock import patch

import cvs.lib.utils_lib as utils_lib


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

    def test_cluster_target_output_label_strips_and_sanitizes(self):
        self.assertEqual(utils_lib.cluster_target_output_label("  node1.example.com  "), "node1.example.com")
        self.assertEqual(utils_lib.cluster_target_output_label("a/b"), "a_b")
        self.assertEqual(utils_lib.cluster_target_output_label(""), "unknown_node")

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


if __name__ == '__main__':
    unittest.main()
