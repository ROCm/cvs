import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.agent import lifecycle


class FakeAgent:
    def __init__(self, _agent_dir, _rank, _world_size):
        self.host = "rank0"

    def start(self):
        pass

    def wait_until_ready(self, _timeout):
        return 9000

    def wait_for_registrations(self, _timeout):
        raise TimeoutError

    def stop(self):
        pass


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agent_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_managed_rank_reads_scheduler_environment(self):
        with patch.dict(os.environ, {"SLURM_PROCID": "1", "SLURM_NTASKS": "2"}, clear=True):
            self.assertEqual(lifecycle.managed_rank(), (1, 2))

    def test_rank0_publishes_rendezvous_and_secret(self):
        coordinator = lifecycle.start_rank0(
            self.agent_dir,
            1,
            agent_factory=FakeAgent,
            heartbeat_interval=60,
        )
        self.addCleanup(coordinator.close)
        self.assertEqual(lifecycle.read_rank0_rendezvous(self.agent_dir), ("rank0", 9000, 60.0))
        self.assertEqual(len((self.agent_dir / "secret").read_text(encoding="utf-8").strip()), 64)
        self.assertEqual(stat.S_IMODE((self.agent_dir / "secret").stat().st_mode), 0o600)
        coordinator.close()
        self.assertTrue((self.agent_dir / lifecycle.RANK0_DONE_FILENAME).exists())

    def test_agent_runner_starts_and_registers_rank0(self):
        lifecycle.write_auth_token(self.agent_dir)
        agent = lifecycle.AgentRunner(self.agent_dir, rank=0, world_size=1, host="localhost")
        self.addCleanup(agent.stop)
        agent.start()
        self.assertEqual(agent.wait_until_ready(timeout=5), agent.port)
        self.assertIn(0, agent.wait_for_registrations(timeout=1))

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    def test_worker_registration_uses_bearer_token(self, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        lifecycle.register_worker("http://rank0:9000", "token", 1, "worker1", 9001, timeout=1)
        mock_client_class.assert_called_once_with(headers={"Authorization": "Bearer token"})
        self.assertTrue(client.post.call_args.args[0].endswith("/v1/register"))
        client.post.return_value.raise_for_status.assert_called_once()

    def test_worker_watcher_stops_for_done_or_stale_heartbeat(self):
        lifecycle.mark_done(self.agent_dir)
        self.assertEqual(lifecycle.watch_rank0(self.agent_dir, heartbeat_interval=1), 0)
        (self.agent_dir / lifecycle.RANK0_DONE_FILENAME).unlink()
        self.assertEqual(lifecycle.watch_rank0(self.agent_dir, heartbeat_interval=0), 1)


if __name__ == "__main__":
    unittest.main()
