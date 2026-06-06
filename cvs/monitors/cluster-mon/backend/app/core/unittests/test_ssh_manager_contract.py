"""
SSH-manager contract test.

Characterizes the behavior cluster-mon relies on from its SSH manager. The
assertions live in ``SshManagerContractMixin`` and run against the
``cluster_ssh_manager.ClusterSshManager`` adapter (the production SSH manager).

The mixin was originally also exercised against the legacy
``cvs_parallel_ssh_reliable.Pssh`` to prove parity during the migration; that
implementation has since been removed, so only the adapter contract remains.

Each concrete TestCase supplies the impl-specific mocking via the hook methods
(``make_manager`` / ``seed_exec_output`` / ``seed_cmd_list_output`` /
``set_refresh_probe`` / ``assert_recreate_rebuilds``). No network:
``MultiProcessPssh`` and the TCP probe (``discover_reachable_hosts``) are mocked.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from app.core.cluster_ssh_manager import ClusterSshManager

ADAPTER_MODULE = "app.core.cluster_ssh_manager"

ABORT = "ABORT: Host Unreachable Error"


class SshManagerContractMixin:
    """Behavioral contract every cluster-mon SSH manager must satisfy.

    Subclasses must implement the hook methods below; the ``test_*`` methods are
    shared and run for each concrete implementation.
    """

    # ------------------------------------------------------------------ hooks
    def make_manager(self, reachable, unreachable):
        """Build a manager whose pre-probe yields (reachable, unreachable).

        The full original host_list is ``reachable + unreachable``. The
        underlying SSH client is mocked; store handles for the seed_* hooks.
        """
        raise NotImplementedError

    def seed_exec_output(self, mapping):
        """Arrange the next ``manager.exec(...)`` to yield ``{host: text}``
        for the reachable hosts (before any ABORT merge)."""
        raise NotImplementedError

    def seed_cmd_list_output(self, mapping):
        """Same as ``seed_exec_output`` but for ``exec_cmd_list``."""
        raise NotImplementedError

    def set_refresh_probe(self, reachable, unreachable):
        """Arrange the next reachability re-probe to yield (reachable, unreachable)."""
        raise NotImplementedError

    def assert_recreate_rebuilds(self, manager):
        """Call ``manager.recreate_client()`` and assert the underlying client
        was rebuilt."""
        raise NotImplementedError

    # ------------------------------------------------------------- assertions
    def test_exec_returns_host_to_str_map(self):
        mgr = self.make_manager(["h1", "h2"], [])
        self.seed_exec_output({"h1": "ok-one", "h2": "ok-two"})

        result = mgr.exec("echo hi")

        self.assertEqual(set(result), {"h1", "h2"})
        self.assertIsInstance(result["h1"], str)
        self.assertIn("ok-one", result["h1"])
        self.assertIn("ok-two", result["h2"])

    def test_unreachable_hosts_get_abort_marker(self):
        mgr = self.make_manager(["h1"], ["h2"])
        self.seed_exec_output({"h1": "ok"})

        result = mgr.exec("echo hi")

        self.assertIn(ABORT, result["h2"])
        self.assertNotIn("ABORT", result["h1"])

    def test_no_reachable_hosts_returns_all_abort(self):
        mgr = self.make_manager([], ["h1", "h2"])

        result = mgr.exec("echo hi")

        self.assertEqual(result, {"h1": ABORT, "h2": ABORT})

    def test_exec_cmd_list_returns_host_to_str_map(self):
        mgr = self.make_manager(["h1", "h2"], [])
        self.seed_cmd_list_output({"h1": "out-a", "h2": "out-b"})

        result = mgr.exec_cmd_list(["echo a", "echo b"])

        self.assertEqual(set(result), {"h1", "h2"})
        self.assertIsInstance(result["h1"], str)
        self.assertIn("out-a", result["h1"])

    def test_attributes_reflect_probe_result(self):
        mgr = self.make_manager(["h1"], ["h2"])

        # host_list keeps the original full list; reachable/unreachable reflect the probe.
        self.assertEqual(mgr.host_list, ["h1", "h2"])
        self.assertEqual(mgr.reachable_hosts, ["h1"])
        self.assertEqual(mgr.unreachable_hosts, ["h2"])

    def test_get_hosts_return_copies(self):
        mgr = self.make_manager(["h1"], ["h2"])

        reachable = mgr.get_reachable_hosts()
        reachable.append("mutated")

        self.assertEqual(mgr.get_reachable_hosts(), ["h1"])
        self.assertEqual(mgr.get_unreachable_hosts(), ["h2"])

    def test_refresh_host_reachability_returns_bool_and_updates(self):
        mgr = self.make_manager(["h1", "h2"], [])

        # h2 went offline -> change expected.
        self.set_refresh_probe(["h1"], ["h2"])
        changed = mgr.refresh_host_reachability()
        self.assertTrue(changed)
        self.assertEqual(mgr.get_reachable_hosts(), ["h1"])
        self.assertIn("h2", mgr.get_unreachable_hosts())

        # Same probe result again -> no change.
        self.assertFalse(mgr.refresh_host_reachability())

    def test_recreate_client_rebuilds(self):
        mgr = self.make_manager(["h1", "h2"], [])
        self.assert_recreate_rebuilds(mgr)

    def test_exec_async_does_not_block_event_loop(self):
        mgr = self.make_manager(["h1"], [])

        ticks = {"n": 0}

        def slow_exec(*_args, **_kwargs):
            time.sleep(0.3)
            return {"h1": "ok"}

        async def ticker():
            for _ in range(5):
                await asyncio.sleep(0.02)
                ticks["n"] += 1

        async def run():
            with patch.object(mgr, "exec", side_effect=slow_exec):
                result, _ = await asyncio.gather(mgr.exec_async("cmd"), ticker())
            return result

        result = asyncio.run(run())

        self.assertEqual(result, {"h1": "ok"})
        # The event loop kept ticking while exec() blocked in a worker thread.
        self.assertEqual(ticks["n"], 5)


class TestClusterSshManagerContract(SshManagerContractMixin, unittest.TestCase):
    """Run the contract against ClusterSshManager (mock MultiProcessPssh + probe)."""

    def make_manager(self, reachable, unreachable):
        disc = patch(f"{ADAPTER_MODULE}.discover_reachable_hosts", return_value=(list(reachable), list(unreachable)))
        mp = patch(f"{ADAPTER_MODULE}.MultiProcessPssh")
        self._mock_discover = disc.start()
        self.addCleanup(disc.stop)
        self._mock_mp_cls = mp.start()
        self.addCleanup(mp.stop)
        self._mp = MagicMock()
        self._mock_mp_cls.return_value = self._mp
        return ClusterSshManager(list(reachable) + list(unreachable), user="u", password="p")

    def seed_exec_output(self, mapping):
        self._mp.exec.return_value = dict(mapping)

    def seed_cmd_list_output(self, mapping):
        self._mp.exec_cmd_list.return_value = dict(mapping)

    def set_refresh_probe(self, reachable, unreachable):
        self._mock_discover.return_value = (list(reachable), list(unreachable))

    def assert_recreate_rebuilds(self, manager):
        self._mock_mp_cls.reset_mock()
        manager.recreate_client()
        self._mock_mp_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
