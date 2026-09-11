'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import unittest
from types import SimpleNamespace

from cvs.tests.mori import conftest as mori_conftest

_MORI_FILE = os.path.join(mori_conftest._HERE, 'mori_benchmark_test.py')
_OTHER_FILE = os.path.join(os.path.dirname(mori_conftest._HERE), 'rccl', 'rccl_perf.py')


def _item(name, path=_MORI_FILE):
    base = name.split('[')[0]
    return SimpleNamespace(name=name, originalname=base, path=SimpleNamespace(parent=os.path.dirname(path)), file=path)


def _names(items):
    return [(os.path.basename(it.file), it.name) for it in items]


class TestCollectionOrder(unittest.TestCase):
    def test_mori_items_follow_the_lifecycle(self):
        items = [
            _item('test_teardown'),
            _item('test_io_write[16384-128-1]'),
            _item('test_verify_dmesg'),
            _item('test_io_read[16384-128-1]'),
            _item('test_setup_ibv_devices'),
            _item('test_launch_mori_container'),
            _item('test_cleanup_stale_containers'),
        ]
        mori_conftest.pytest_collection_modifyitems(items)
        self.assertEqual(
            [it.name for it in items],
            [
                'test_cleanup_stale_containers',
                'test_launch_mori_container',
                'test_setup_ibv_devices',
                'test_io_read[16384-128-1]',
                'test_io_write[16384-128-1]',
                'test_verify_dmesg',
                'test_teardown',
            ],
        )

    def test_other_suites_keep_their_positions(self):
        # The hook runs for the whole session, e.g. `pytest cvs/tests/mori cvs/tests/rccl`.
        # A rank that collides with mori's (test_teardown) must not pull a foreign item into mori's order.
        items = [
            _item('test_teardown', _OTHER_FILE),
            _item('test_teardown'),
            _item('test_b', _OTHER_FILE),
            _item('test_launch_mori_container'),
            _item('test_a', _OTHER_FILE),
        ]
        mori_conftest.pytest_collection_modifyitems(items)
        self.assertEqual(
            _names(items),
            [
                ('rccl_perf.py', 'test_teardown'),
                ('mori_benchmark_test.py', 'test_launch_mori_container'),
                ('rccl_perf.py', 'test_b'),
                ('mori_benchmark_test.py', 'test_teardown'),
                ('rccl_perf.py', 'test_a'),
            ],
        )

    def test_parametrized_cases_keep_their_relative_order(self):
        cases = [f'test_io_read[{p}]' for p in ('32768-256-8', '16384-128-1', '32768-128-1')]
        items = [_item('test_teardown')] + [_item(c) for c in cases]
        mori_conftest.pytest_collection_modifyitems(items)
        self.assertEqual([it.name for it in items], cases + ['test_teardown'])


class TestRunDeckFixtures(unittest.TestCase):
    def _variant(self, mori_dict, orchestrator_type, image='img:1'):
        orch = SimpleNamespace(
            orchestrator_type=orchestrator_type, hosts=['n0', 'n1'], container_config={'image': image}
        )
        return mori_conftest.mori_variant_config.__wrapped__(mori_dict, orch)

    def test_container_variant_reads_config_and_orch(self):
        mori_dict = {'gpu_name': 'mi325x', 'env': {'MORI_RDMA_DEVICES': 'rdma0,rdma1'}}
        self.assertEqual(
            self._variant(mori_dict, 'container'),
            {'gpu_name': 'mi325x', 'node_count': 2, 'mori_device_list': 'rdma0,rdma1', 'container_image': 'img:1'},
        )

    def test_baremetal_variant_reports_no_image(self):
        # The merged container block may still carry an image, but nothing is launched under baremetal.
        variant = self._variant({'env': {}}, 'baremetal')
        self.assertIsNone(variant['container_image'])
        self.assertIsNone(variant['gpu_name'])
        self.assertIsNone(variant['mori_device_list'])

    def test_results_dict_is_fresh_per_module(self):
        first = mori_conftest.cvs_results_dict.__wrapped__()
        first['x'] = {}
        self.assertEqual(mori_conftest.cvs_results_dict.__wrapped__(), {})


if __name__ == '__main__':
    unittest.main()
