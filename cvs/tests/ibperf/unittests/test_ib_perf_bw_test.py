'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from unittest.mock import MagicMock, patch

from cvs.tests.ibperf import ib_perf_bw_test as suite

_CONFIG = {
    'install_dir': '/opt/ibperf',
    'msg_size_list': [65536],
    'qp_count_list': [8],
    'gid_index': '3',
    'port_no': '1',
    'duration': '10',
    'verify_bw': 'False',
}


class TestIbBwPerf(unittest.TestCase):
    @patch.object(suite, 'update_test_result')
    @patch.object(suite, 'verify_dmesg_for_errors')
    @patch.object(suite, 'ibperf_lib')
    @patch.object(suite, 'linux_utils')
    def test_runs_on_the_orch_head_and_all_handles(self, linux_utils, ibperf_lib, verify_dmesg, _update):
        orch = MagicMock()
        linux_utils.get_active_rdma_nic_dict.return_value = {}
        suite.test_ib_bw_perf(orch, 'ib_write_bw', _CONFIG)
        args = ibperf_lib.run_ib_perf_bw_test.call_args.args
        self.assertIs(args[0], orch.head)
        self.assertIs(args[1], orch.all)
        self.assertIs(verify_dmesg.call_args.args[0], orch.all)

    @patch.object(suite, 'update_test_result')
    @patch.object(suite, 'get_passwordless_sudo_status', return_value={'n1': True, 'n2': False})
    @patch.object(suite, 'verify_dmesg_for_errors')
    @patch.object(suite, 'ibperf_lib')
    @patch.object(suite, 'linux_utils')
    def test_skips_dmesg_check_with_warning_without_sudo(self, linux_utils, _ibperf_lib, verify_dmesg, *_):
        linux_utils.get_active_rdma_nic_dict.return_value = {}
        with self.assertLogs(suite.log, 'WARNING') as logs:
            suite.test_ib_bw_perf(MagicMock(), 'ib_write_bw', _CONFIG)
        verify_dmesg.assert_not_called()
        self.assertIn("SKIPPING the dmesg error check", logs.output[0])


if __name__ == '__main__':
    unittest.main()
