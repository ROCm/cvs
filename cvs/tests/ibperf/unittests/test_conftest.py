'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cvs.lib import ibperf_lib
from cvs.tests.ibperf import conftest


def _cluster(nodes):
    return {
        'username': 'user',
        'priv_key_file': '/keys/id',
        'node_dict': {n: {'vpc_ip': n} for n in nodes},
    }


class TestOrchFixture(unittest.TestCase):
    def _start(self, nodes):
        orch = MagicMock()
        patcher = patch.object(conftest.OrchestratorFactory, 'create_orchestrator', return_value=orch)
        create = patcher.start()
        self.addCleanup(patcher.stop)
        gen = conftest.orch.__wrapped__(_cluster(nodes))
        self.assertIs(next(gen), orch)
        return gen, orch, create.call_args.args[1]

    def test_odd_node_count_drops_the_last_node(self):
        _, _, cfg = self._start(['n1', 'n2', 'n3'])
        self.assertEqual(list(cfg.node_dict), ['n1', 'n2'])

    def test_teardown_closes_the_orchestrator(self):
        gen, orch, _ = self._start(['n1', 'n2'])
        orch.close.assert_not_called()
        with self.assertRaises(StopIteration):
            next(gen)
        orch.close.assert_called_once_with()


class TestRunDeckFixtures(unittest.TestCase):
    def test_variant_reads_config_and_orch(self):
        config = {
            'ib_bw_test_list': ['ib_write_bw'],
            'msg_size_list': [2, 8192],
            'qp_count_list': ['8'],
            'duration': '10',
            'expected_results': {'ib_write_bw': {}},
            'verify_bw': 'True',
        }
        variant = conftest.ibperf_variant_config.__wrapped__(config, SimpleNamespace(hosts=['n1', 'n2']))
        self.assertEqual(variant['hosts'], ['n1', 'n2'])
        self.assertEqual(variant['node_count'], 2)
        self.assertEqual(variant['ib_bw_test_list'], ['ib_write_bw'])
        self.assertEqual(variant['ib_lat_test_list'], ibperf_lib.configured_lat_tests({}))
        self.assertEqual(variant['verify_bw'], 'True')


if __name__ == '__main__':
    unittest.main()
