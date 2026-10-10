'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == '__main__':
    unittest.main()
