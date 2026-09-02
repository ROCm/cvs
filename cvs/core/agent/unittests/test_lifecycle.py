import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

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

    def registered_agents(self):
        return {}

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
        self.assertIn(0, agent.registered_agents())

    @patch("cvs.core.agent.lifecycle.socket.socket")
    def test_agent_runner_prefers_scheduler_hostname(self, mock_socket):
        mock_socket.return_value.getsockname.return_value = ("0.0.0.0", 9000)
        with patch.dict(os.environ, {"SLURM_NODENAME": "scheduler-host"}, clear=True):
            agent = lifecycle.AgentRunner(self.agent_dir, rank=1, world_size=2)
        self.addCleanup(agent.stop)
        self.assertEqual(agent.host, "scheduler-host")

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    def test_worker_registration_uses_bearer_token(self, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        lifecycle.register_worker("http://rank0:9000", "token", 1, "worker1", 9001, timeout=1)
        mock_client_class.assert_called_once_with(headers={"Authorization": "Bearer token"})
        self.assertTrue(client.post.call_args.args[0].endswith("/v1/register"))
        client.post.return_value.raise_for_status.assert_called_once()

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", return_value=0)
    def test_worker_registration_retries_transient_failure(self, _monotonic, mock_sleep, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        response = MagicMock()
        response.raise_for_status.side_effect = [
            httpx.HTTPStatusError("unavailable", request=MagicMock(), response=httpx.Response(503)),
            None,
        ]
        client.post.return_value = response

        lifecycle.register_worker("http://rank0:9000", "token", 1, "worker1", 9001, timeout=1)

        self.assertEqual(client.post.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", return_value=0)
    def test_worker_registration_retries_connection_failure(self, _monotonic, mock_sleep, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        client.post.side_effect = [httpx.ConnectError("down", request=MagicMock()), MagicMock()]

        lifecycle.register_worker("http://rank0:9000", "token", 1, "worker1", 9001, timeout=1)

        self.assertEqual(client.post.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.read_rank0_rendezvous", side_effect=[ValueError, ("rank0", 9000, 10)])
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", return_value=0)
    def test_wait_for_rendezvous_retries_then_succeeds(self, _monotonic, mock_sleep, mock_rendezvous):
        (self.agent_dir / "secret").write_text("token\n")

        actual = lifecycle.wait_for_rendezvous(self.agent_dir, timeout=1)

        self.assertEqual(actual, ("rank0", 9000, 10, "token"))
        self.assertEqual(mock_rendezvous.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.read_rank0_rendezvous", side_effect=FileNotFoundError)
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", side_effect=[0, 0, 1])
    def test_wait_for_rendezvous_times_out(self, _monotonic, mock_sleep, _mock_rendezvous):
        with self.assertRaisesRegex(TimeoutError, "rank-0 rendezvous did not appear"):
            lifecycle.wait_for_rendezvous(self.agent_dir, timeout=1)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.watch_rank0", return_value=0)
    @patch("cvs.core.agent.lifecycle.register_worker")
    @patch("cvs.core.agent.lifecycle.wait_for_rendezvous", return_value=("rank0", 9000, 10, "token"))
    def test_worker_starts_registers_and_stops(self, mock_rendezvous, mock_register, mock_watch):
        agent = MagicMock(host="worker1")
        agent.wait_until_ready.return_value = 9001

        status = lifecycle.run_worker(self.agent_dir, 1, 2, agent_factory=MagicMock(return_value=agent))

        self.assertEqual(status, 0)
        mock_rendezvous.assert_called_once_with(self.agent_dir, lifecycle.BOOTSTRAP_TIMEOUT_SECONDS)
        mock_register.assert_called_once_with("http://rank0:9000", "token", 1, "worker1", 9001, 60)
        mock_watch.assert_called_once_with(self.agent_dir, 10)
        agent.stop.assert_called_once()

    def test_worker_watcher_stops_for_done_or_stale_heartbeat(self):
        lifecycle.mark_done(self.agent_dir)
        self.assertEqual(lifecycle.watch_rank0(self.agent_dir, heartbeat_interval=1), 0)
        (self.agent_dir / lifecycle.RANK0_DONE_FILENAME).unlink()
        self.assertEqual(lifecycle.watch_rank0(self.agent_dir, heartbeat_interval=0), 1)


if __name__ == "__main__":
    unittest.main()
