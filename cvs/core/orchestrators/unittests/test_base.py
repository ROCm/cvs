'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import inspect
import unittest
from unittest.mock import MagicMock

from cvs.core.orchestrators.base import Orchestrator
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


class _AbstractMethodsOnly(Orchestrator):
    """A custom subclass that implements only the abstract methods."""

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        return {}

    def exec_on_host(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        return {}

    def exec_on_head(self, cmd, timeout=None, detailed=False):
        return {}

    def upload_to_head(self, local_file, remote_file):
        return {}

    def download_from_head(self, remote_file, local_file):
        return {}

    def download_file(self, remote_file, local_file, hosts=None):
        return {}

    def setup_env(self, hosts, env_script=None):
        return True

    def cleanup(self, hosts):
        return True


class TestOrchestratorExecContract(unittest.TestCase):
    def test_abstract_exec_declares_print_console_true(self):
        parameters = inspect.signature(Orchestrator.exec).parameters
        self.assertEqual(list(parameters), ['self', 'cmd', 'hosts', 'timeout', 'detailed', 'print_console'])
        self.assertIs(parameters['print_console'].default, True)

    def test_concrete_exec_accepts_base_parameters_in_order(self):
        expected = list(inspect.signature(Orchestrator.exec).parameters)
        for cls in (BaremetalOrchestrator, ContainerOrchestrator):
            with self.subTest(cls=cls):
                parameters = inspect.signature(cls.exec).parameters
                self.assertEqual(list(parameters)[:6], expected)
                self.assertIs(parameters['detailed'].default, False)
                self.assertIs(parameters['print_console'].default, True)


class TestOrchestratorExecCmdListContract(unittest.TestCase):
    def test_subclass_without_exec_cmd_list_still_instantiates_and_raises_not_implemented(self):
        orch = _AbstractMethodsOnly(MagicMock(), {})
        with self.assertRaisesRegex(NotImplementedError, 'exec_cmd_list not supported by _AbstractMethodsOnly'):
            orch.exec_cmd_list(['echo a'])

    def test_concrete_backends_match_base_signature(self):
        expected = inspect.signature(Orchestrator.exec_cmd_list).parameters
        for cls in (BaremetalOrchestrator, ContainerOrchestrator):
            with self.subTest(cls=cls):
                parameters = inspect.signature(cls.exec_cmd_list).parameters
                self.assertEqual(list(parameters), list(expected))
                self.assertIsNone(parameters['timeout'].default)
                self.assertIs(parameters['print_console'].default, True)


if __name__ == '__main__':
    unittest.main()
