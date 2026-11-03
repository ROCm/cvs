'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import inspect
import unittest

from cvs.core.orchestrators.base import Orchestrator
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


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
