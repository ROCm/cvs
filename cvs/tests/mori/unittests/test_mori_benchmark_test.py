'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from unittest.mock import MagicMock

from _pytest.outcomes import Failed, Skipped

from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.tests.mori import mori_benchmark_test as suite
from cvs.tests.mori.conftest import _Lifecycle


def _container_orch(container_config, running=True):
    orch = MagicMock()
    orch.orchestrator_type = 'container'
    orch.container_config = container_config
    orch.get_container_name.side_effect = ContainerOrchestrator.get_container_name
    orch.setup_containers.return_value = True
    orch.verify_containers_running.return_value = running
    return orch


class TestLaunchMoriContainer(unittest.TestCase):
    def test_missing_image_fails_and_marks_launch_failed(self):
        # Without lifecycle.failed every benchmark test would run against a container that doesn't exist.
        for config in ({'name': 'mori_c'}, {'name': 'mori_c', 'image': ''}):
            with self.subTest(config=config):
                orch, lifecycle = _container_orch(config), _Lifecycle()
                with self.assertRaises(Failed) as ctx:
                    suite.test_launch_mori_container(orch, lifecycle)
                self.assertIn('container.image', str(ctx.exception))
                self.assertTrue(lifecycle.failed)
                orch.setup_containers.assert_not_called()

    def test_launch_with_image_sets_up_and_verifies_the_named_container(self):
        orch, lifecycle = _container_orch({'name': 'mori_c', 'image': 'img'}), _Lifecycle()
        suite.test_launch_mori_container(orch, lifecycle)
        orch.setup_containers.assert_called_once_with()
        orch.verify_containers_running.assert_called_once_with('mori_c')
        self.assertFalse(lifecycle.failed)

    def test_container_not_running_after_setup_fails(self):
        orch, lifecycle = _container_orch({'name': 'mori_c', 'image': 'img'}, running=False), _Lifecycle()
        with self.assertRaises(Failed):
            suite.test_launch_mori_container(orch, lifecycle)
        self.assertTrue(lifecycle.failed)


class TestTeardown(unittest.TestCase):
    def test_missing_image_skips_and_disarms_the_leak_guard(self):
        orch, lifecycle = _container_orch({'name': 'mori_c'}), _Lifecycle()
        with self.assertRaises(Skipped):
            suite.test_teardown(orch, lifecycle)
        self.assertTrue(lifecycle.torn_down)
        orch.teardown_containers.assert_not_called()

    def test_per_run_container_still_running_fails(self):
        orch, lifecycle = _container_orch({'name': 'mori_c', 'image': 'img'}, running=True), _Lifecycle()
        with self.assertRaises(Failed):
            suite.test_teardown(orch, lifecycle)
        orch.teardown_containers.assert_called_once_with()
        self.assertTrue(lifecycle.torn_down)

    def test_per_run_container_removed_passes(self):
        orch, lifecycle = _container_orch({'name': 'mori_c', 'image': 'img'}, running=False), _Lifecycle()
        suite.test_teardown(orch, lifecycle)
        orch.verify_containers_running.assert_called_once_with('mori_c')


if __name__ == '__main__':
    unittest.main()
