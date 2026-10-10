'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from unittest.mock import MagicMock, patch

from cvs.tests.ibperf import install_ibperf_tools as suite


class TestInstallIbPerf(unittest.TestCase):
    @patch.object(suite, 'update_test_result')
    @patch.object(suite, 'fail_test')
    @patch.object(suite.ibperf_lib, 'detect_rocm_path', return_value='/opt/rocm')
    def test_configures_perftest_with_rocm_dmabuf(self, *_mocks):
        orch = MagicMock()
        suite.test_install_ib_perf(orch, {'install_perf_package': 'True', 'install_dir': '/opt/ibperf'})
        configure = [c.args[0] for c in orch.head.exec.call_args_list if './configure' in c.args[0]]
        self.assertEqual(len(configure), 1)
        self.assertTrue(configure[0].endswith('--with-rocm=/opt/rocm --enable-rocm --enable-rocm-dmabuf'))


if __name__ == '__main__':
    unittest.main()
