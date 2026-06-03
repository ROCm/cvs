"""
SSH-manager contract test.

Characterizes the behavior cluster-mon relies on, run against the CURRENT
implementation (cvs_parallel_ssh_reliable.Pssh). The same assertions are meant
to be re-run against the future ClusterSshManager adapter to prove parity.

No network: parallel-ssh's ParallelSSHClient and the TCP probe
(discover_reachable_hosts) are mocked.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from app.core.cvs_parallel_ssh_reliable import Pssh

MODULE = "app.core.cvs_parallel_ssh_reliable"


def make_item(host, stdout=None, stderr=None, exception=None):
    item = MagicMock()
    item.host = host
    item.stdout = stdout or []
    item.stderr = stderr or []
    item.exception = exception
    return item


class TestPsshContract(unittest.TestCase):
    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_exec_returns_host_to_str_map(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1", "h2"], [])
        client = MagicMock()
        mock_cls.return_value = client
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        client.run_command.return_value = [
            make_item("h1", stdout=["ok-one"]),
            make_item("h2", stdout=["ok-two"]),
        ]

        result = pssh.exec("echo hi")

        self.assertEqual(set(result), {"h1", "h2"})
        self.assertIsInstance(result["h1"], str)
        self.assertIn("ok-one", result["h1"])
        self.assertIn("ok-two", result["h2"])

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_unreachable_hosts_get_abort_marker(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1"], ["h2"])
        client = MagicMock()
        mock_cls.return_value = client
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        client.run_command.return_value = [make_item("h1", stdout=["ok"])]

        result = pssh.exec("echo hi")

        self.assertIn("ABORT: Host Unreachable Error", result["h2"])
        self.assertNotIn("ABORT", result["h1"])

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_no_reachable_hosts_returns_all_abort(self, mock_discover, mock_cls):
        mock_discover.return_value = ([], ["h1", "h2"])
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        result = pssh.exec("echo hi")

        self.assertEqual(
            result,
            {"h1": "ABORT: Host Unreachable Error", "h2": "ABORT: Host Unreachable Error"},
        )

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_exec_cmd_list_returns_host_to_str_map(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1", "h2"], [])
        client = MagicMock()
        mock_cls.return_value = client
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        client.run_command.return_value = [
            make_item("h1", stdout=["out-a"]),
            make_item("h2", stdout=["out-b"]),
        ]

        result = pssh.exec_cmd_list(["echo a", "echo b"])

        self.assertEqual(set(result), {"h1", "h2"})
        self.assertIsInstance(result["h1"], str)
        self.assertIn("out-a", result["h1"])

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_attributes_reflect_probe_result(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1"], ["h2"])
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        # host_list keeps the original full list; reachable/unreachable reflect the probe.
        self.assertEqual(pssh.host_list, ["h1", "h2"])
        self.assertEqual(pssh.reachable_hosts, ["h1"])
        self.assertEqual(pssh.unreachable_hosts, ["h2"])

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_get_hosts_return_copies(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1"], ["h2"])
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        reachable = pssh.get_reachable_hosts()
        reachable.append("mutated")

        self.assertEqual(pssh.get_reachable_hosts(), ["h1"])
        self.assertEqual(pssh.get_unreachable_hosts(), ["h2"])

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_refresh_host_reachability_returns_bool_and_updates(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1", "h2"], [])
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        # h2 went offline -> change expected.
        mock_discover.return_value = (["h1"], ["h2"])
        changed = pssh.refresh_host_reachability()
        self.assertTrue(changed)
        self.assertEqual(pssh.get_reachable_hosts(), ["h1"])
        self.assertIn("h2", pssh.get_unreachable_hosts())

        # Same probe result again -> no change.
        self.assertFalse(pssh.refresh_host_reachability())

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_recreate_client_rebuilds(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1", "h2"], [])
        pssh = Pssh(MagicMock(), ["h1", "h2"], user="u", password="p", stop_on_errors=False)

        mock_cls.reset_mock()
        pssh.recreate_client()
        mock_cls.assert_called_once()

    @patch(f"{MODULE}.ParallelSSHClient")
    @patch(f"{MODULE}.discover_reachable_hosts")
    def test_exec_async_does_not_block_event_loop(self, mock_discover, mock_cls):
        mock_discover.return_value = (["h1"], [])
        pssh = Pssh(MagicMock(), ["h1"], user="u", password="p", stop_on_errors=False)

        ticks = {"n": 0}

        def slow_exec(*_args, **_kwargs):
            time.sleep(0.3)
            return {"h1": "ok"}

        async def ticker():
            for _ in range(5):
                await asyncio.sleep(0.02)
                ticks["n"] += 1

        async def run():
            with patch.object(pssh, "exec", side_effect=slow_exec):
                result, _ = await asyncio.gather(pssh.exec_async("cmd"), ticker())
            return result

        result = asyncio.run(run())

        self.assertEqual(result, {"h1": "ok"})
        # The event loop kept ticking while exec() blocked in a worker thread.
        self.assertEqual(ticks["n"], 5)


if __name__ == "__main__":
    unittest.main()
