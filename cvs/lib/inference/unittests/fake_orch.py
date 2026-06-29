'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Minimal stand-in for :class:`ContainerOrchestrator` in inference Job unit tests.

Reusable by any suite's ``test_*_orch_parse.py`` — import ``FakeOrch`` instead of
copying the class into each test module.
'''


class FakeOrch:
    def __init__(self, exec_return=None):
        self.exec_return = exec_return if exec_return is not None else {}
        self.commands = []

    def exec(self, cmd, **kwargs):
        self.commands.append(cmd)
        return self.exec_return
