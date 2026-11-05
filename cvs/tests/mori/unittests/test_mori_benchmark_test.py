'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os
import re
import unittest

import cvs
from cvs.tests.mori import mori_benchmark_test as suite


class TestSampleIoThresholds(unittest.TestCase):
    # run_mori_torch_io_test only checks a threshold whose key names a case that runs, so a stale key passes silently.
    def setUp(self):
        path = os.path.join(os.path.dirname(cvs.__file__), 'input', 'config_file', 'mori', 'mi35x_mori_config.json')
        with open(path) as f:
            self.expected = json.load(f)['expected_results']

    def _io_thresholds(self):
        for op in ('io_read', 'io_write'):
            for key, by_size in self.expected[op].items():
                m = re.fullmatch(r'BUFF_SIZE:(\d+),TRANSFER_SIZE:(\d+),QP_COUNT:(\d+)', key)
                self.assertIsNotNone(m, key)
                yield op, key, tuple(int(v) for v in m.groups()), by_size

    def test_every_io_threshold_names_a_matrix_case(self):
        thresholds = list(self._io_thresholds())
        self.assertEqual({op for op, _, _, _ in thresholds}, {'io_read', 'io_write'})
        for op, key, case, by_size in thresholds:
            with self.subTest(op=op, key=key):
                self.assertIn(case, suite.input_test_matrix)
                self.assertTrue(all(int(size) <= case[0] for size in by_size), 'message size above the sweep max')

    def test_latency_ceiling_is_not_stricter_than_the_bandwidth_floor(self):
        # Avg latency is the time per batch, so at the bandwidth floor it is size * batch / BW.
        for op, key, (_, batch, _), by_size in self._io_thresholds():
            for size, limits in by_size.items():
                with self.subTest(op=op, key=key, size=size):
                    lat_at_floor_us = int(size) * batch / (float(limits['max_bw']) * 1e3)
                    self.assertGreaterEqual(float(limits['avg_lat']), lat_at_floor_us)


if __name__ == '__main__':
    unittest.main()
