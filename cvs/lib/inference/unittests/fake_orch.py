'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Minimal stand-in for :class:`ContainerOrchestrator` in inference Job unit tests.

Reusable by any suite's ``test_*_orch_parse.py`` — import ``FakeOrch`` instead of
copying the class into each test module.
'''


class FakeOrch:
    def __init__(self, exec_return=None, hosts=None, exec_on_head_return=None):
        self.hosts = list(hosts or ["node0"])
        self.exec_return = exec_return if exec_return is not None else {}
        self.exec_on_head_return = exec_on_head_return if exec_on_head_return is not None else self.exec_return
        self.commands = []
        self.exec_on_head_commands = []

    def exec(self, cmd, hosts=None, **kwargs):
        self.commands.append((cmd, hosts))
        return self.exec_return

    def exec_on_head(self, cmd, **kwargs):
        self.exec_on_head_commands.append(cmd)
        return self.exec_on_head_return
