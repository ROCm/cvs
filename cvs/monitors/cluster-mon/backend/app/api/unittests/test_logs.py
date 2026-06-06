"""
Contract tests for the logs API.

Covers the pure grep-command validator and the /search endpoint's
exec_async -> filtered-results contract. The endpoint pulls the SSH manager via
``from app.main import app_state`` at call time, so we inject a lightweight fake
``app.main`` module into sys.modules to avoid importing the heavy real app.
"""

import sys
import types
import unittest

from app.unittests.testing import FakeSshManager
from app.api.logs import validate_grep_command


class TestValidateGrepCommand(unittest.TestCase):
    def test_valid_piped_grep(self):
        ok, msg = validate_grep_command("grep -i error | grep -v vital")
        self.assertTrue(ok, msg)

    def test_empty_is_invalid(self):
        ok, _ = validate_grep_command("")
        self.assertFalse(ok)

    def test_dangerous_character_rejected(self):
        ok, _ = validate_grep_command("grep error; rm -rf /")
        self.assertFalse(ok)

    def test_forbidden_keyword_rejected(self):
        ok, _ = validate_grep_command("grep error | cat /etc/passwd")
        self.assertFalse(ok)

    def test_segment_must_start_with_grep(self):
        ok, _ = validate_grep_command("awk '{print $1}'")
        self.assertFalse(ok)


class TestSearchEndpoint(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fake_main = types.ModuleType("app.main")

        class _State:
            ssh_manager = None

        self._state = _State()
        fake_main.app_state = self._state
        self._saved_main = sys.modules.get("app.main")
        sys.modules["app.main"] = fake_main

    async def asyncTearDown(self):
        if self._saved_main is not None:
            sys.modules["app.main"] = self._saved_main
        else:
            sys.modules.pop("app.main", None)

    async def test_search_filters_errors_and_empty(self):
        ssh = FakeSshManager(
            ["n1", "n2", "n3"],
            command_map={
                "dmesg -T": {
                    "n1": "matching line\n",
                    "n2": "ABORT: Host Unreachable Error",
                    "n3": "",
                }
            },
        )
        self._state.ssh_manager = ssh

        from app.api.logs import search_dmesg_logs

        resp = await search_dmesg_logs(grep_command="grep -i match")

        self.assertEqual(resp["results"], {"n1": "matching line"})
        self.assertEqual(resp["nodes_with_results"], 1)
        self.assertEqual(resp["total_nodes_searched"], 3)
        self.assertEqual(resp["grep_command"], "grep -i match")

    async def test_search_rejects_invalid_grep(self):
        from fastapi import HTTPException
        from app.api.logs import search_dmesg_logs

        self._state.ssh_manager = FakeSshManager(["n1"])

        with self.assertRaises(HTTPException) as ctx:
            await search_dmesg_logs(grep_command="rm -rf /")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
