"""Unit tests for the topology suite's container lifecycle fixtures."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import pytest

from cvs.tests.topology import conftest


def _cluster(nodes):
    return {
        'username': 'user',
        'priv_key_file': '/keys/id',
        'node_dict': {n: {'vpc_ip': n} for n in nodes},
    }


class TestOrchFixture(unittest.TestCase):
    def _start(self, cluster):
        orch = MagicMock(orchestrator_type=cluster.get('orchestrator', 'baremetal'))
        patcher = patch.object(conftest.OrchestratorFactory, 'create_orchestrator', return_value=orch)
        patcher.start()
        self.addCleanup(patcher.stop)
        lifecycle = conftest.lifecycle.__wrapped__()
        gen = conftest.orch.__wrapped__(cluster, lifecycle)
        return gen, orch, lifecycle

    def test_teardown_closes_the_orchestrator(self):
        gen, orch, _ = self._start(_cluster(['n1', 'n2']))
        next(gen)
        orch.close.assert_not_called()
        with self.assertRaises(StopIteration):
            next(gen)
        orch.close.assert_called_once_with()

    def test_container_mode_launch_then_teardown(self):
        cluster = {**_cluster(['n1', 'n2']), 'orchestrator': 'container'}
        gen, orch, lifecycle = self._start(cluster)
        o = next(gen)
        orch.setup_containers.assert_not_called()
        conftest.launch_containers(o, lifecycle)
        orch.setup_containers.assert_called_once_with()
        conftest.teardown_containers(o, lifecycle)
        orch.teardown_containers.assert_called_once_with()
        with self.assertRaises(StopIteration):
            next(gen)
        # test_teardown_containers already ran: the fixture's own leak guard must not run it twice.
        orch.teardown_containers.assert_called_once_with()

    def test_baremetal_mode_skips_launch_and_teardown(self):
        gen, orch, lifecycle = self._start(_cluster(['n1', 'n2']))
        o = next(gen)
        with self.assertRaises(pytest.skip.Exception):
            conftest.launch_containers(o, lifecycle)
        with self.assertRaises(pytest.skip.Exception):
            conftest.teardown_containers(o, lifecycle)
        with self.assertRaises(StopIteration):
            next(gen)
        orch.teardown_containers.assert_not_called()

    def test_failed_launch_skips_later_tests_and_leak_guard_tears_down(self):
        cluster = {**_cluster(['n1', 'n2']), 'orchestrator': 'container'}
        gen, orch, lifecycle = self._start(cluster)
        orch.setup_containers.return_value = False
        o = next(gen)

        with self.assertRaises(pytest.fail.Exception):
            conftest.launch_containers(o, lifecycle)
        self.assertTrue(lifecycle.failed)

        with self.assertRaises(pytest.skip.Exception):
            conftest.skip_after_failed_launch.__wrapped__(
                SimpleNamespace(node=SimpleNamespace(name='test_nic_bandwidth')), lifecycle
            )
        # test_teardown_containers itself must still run despite the earlier failure.
        conftest.skip_after_failed_launch.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(name='test_teardown_containers')), lifecycle
        )

        with self.assertRaises(StopIteration):
            next(gen)
        orch.teardown_containers.assert_called_once_with()
        orch.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
