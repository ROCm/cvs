import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from cvs.core.agent import lifecycle, messages


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agent_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _worker_runner(self):
        runner = lifecycle.WorkerRunner(MagicMock(agent_dir=self.agent_dir), 1, 2, "node02")
        runner._http_agent = MagicMock(host="worker1")
        return runner

    @patch("cvs.core.agent.lifecycle.HttpAgentServer")
    @patch("cvs.core.agent.lifecycle.scheduler_rank", return_value=(0, 1))
    @patch("cvs.core.agent.lifecycle.scheduler_hosts", return_value=["rank0"])
    def test_rank0_publishes_rendezvous_and_secret(self, _hosts, _rank, mock_http_agent):
        http_agent = mock_http_agent.return_value
        http_agent.host = "rank0"
        http_agent.wait_until_ready.return_value = 9000
        runner = lifecycle.AgentRunner(MagicMock(agent_dir=self.agent_dir))
        self.addCleanup(runner.stop)
        runner.start()
        worker = lifecycle.WorkerRunner(MagicMock(agent_dir=self.agent_dir), 1, 1, "rank0")
        self.assertEqual(
            worker._read_rendezvous(),
            ("rank0", 9000, float(lifecycle.HEARTBEAT_INTERVAL_SECONDS)),
        )
        self.assertEqual(len((self.agent_dir / "secret").read_text(encoding="utf-8").strip()), 64)
        self.assertEqual(stat.S_IMODE((self.agent_dir / "secret").stat().st_mode), 0o600)
        runner.stop()
        self.assertTrue((self.agent_dir / lifecycle.RANK0_DONE_FILENAME).exists())

    def test_http_agent_starts_and_registers_rank0(self):
        (self.agent_dir / messages.AUTH_TOKEN_FILENAME).write_text("token\n")
        agent = lifecycle.HttpAgentServer(self.agent_dir, rank=0, world_size=1, host="localhost")
        self.addCleanup(agent.stop)
        agent.start()
        self.assertEqual(agent.wait_until_ready(timeout=5), agent.port)
        self.assertIn(0, agent.wait_for_registrations(timeout=1))
        self.assertIn(0, agent.registered_agents())

    @patch("cvs.core.agent.lifecycle.socket.socket")
    def test_http_agent_prefers_scheduler_hostname(self, mock_socket):
        mock_socket.return_value.getsockname.return_value = ("0.0.0.0", 9000)
        with patch.dict(os.environ, {"SLURM_NODENAME": "scheduler-host"}, clear=True):
            agent = lifecycle.HttpAgentServer(self.agent_dir, rank=1, world_size=2)
        self.addCleanup(agent.stop)
        self.assertEqual(agent.host, "scheduler-host")

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    def test_worker_registration_uses_bearer_token(self, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        self._worker_runner()._register("http://rank0:9000", "token", 9001)
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

        self._worker_runner()._register("http://rank0:9000", "token", 9001)

        self.assertEqual(client.post.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.httpx.Client")
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", return_value=0)
    def test_worker_registration_retries_connection_failure(self, _monotonic, mock_sleep, mock_client_class):
        client = mock_client_class.return_value.__enter__.return_value
        client.post.side_effect = [httpx.ConnectError("down", request=MagicMock()), MagicMock()]

        self._worker_runner()._register("http://rank0:9000", "token", 9001)

        self.assertEqual(client.post.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.WorkerRunner._read_rendezvous", side_effect=[ValueError, ("rank0", 9000, 10)])
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", return_value=0)
    def test_rendezvous_retries_then_succeeds(self, _monotonic, mock_sleep, mock_rendezvous):
        (self.agent_dir / "secret").write_text("token\n")

        actual = self._worker_runner()._rendezvous(timeout=1)

        self.assertEqual(actual, ("rank0", 9000, 10, "token"))
        self.assertEqual(mock_rendezvous.call_count, 2)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.WorkerRunner._read_rendezvous", side_effect=FileNotFoundError)
    @patch("cvs.core.agent.lifecycle.time.sleep")
    @patch("cvs.core.agent.lifecycle.time.monotonic", side_effect=[0, 0, 1])
    def test_rendezvous_times_out(self, _monotonic, mock_sleep, _mock_rendezvous):
        with self.assertRaisesRegex(TimeoutError, "rank-0 rendezvous did not appear"):
            self._worker_runner()._rendezvous(timeout=1)
        mock_sleep.assert_called_once_with(lifecycle.POLL_INTERVAL_SECONDS)

    @patch("cvs.core.agent.lifecycle.WorkerRunner._watch", return_value=0)
    @patch("cvs.core.agent.lifecycle.WorkerRunner._register")
    @patch("cvs.core.agent.lifecycle.WorkerRunner._rendezvous", return_value=("rank0", 9000, 10, "token"))
    @patch("cvs.core.agent.lifecycle.HttpAgentServer")
    @patch("cvs.core.agent.lifecycle.scheduler_rank", return_value=(1, 2))
    @patch("cvs.core.agent.lifecycle.scheduler_hosts", return_value=["node01", "node02"])
    def test_worker_starts_registers_and_stops(
        self, _hosts, _rank, mock_http_agent, mock_rendezvous, mock_register, mock_watch
    ):
        http_agent = mock_http_agent.return_value
        http_agent.host = "node02"
        http_agent.wait_until_ready.return_value = 9001
        runner = lifecycle.AgentRunner(MagicMock(agent_dir=self.agent_dir))
        self.assertFalse(runner.is_rank0)
        status = runner.start()

        self.assertEqual(status, 0)
        mock_http_agent.assert_called_once_with(self.agent_dir, 1, 2, host="node02")
        mock_rendezvous.assert_called_once_with(lifecycle.BOOTSTRAP_TIMEOUT_SECONDS)
        mock_register.assert_called_once_with("http://rank0:9000", "token", 9001)
        mock_watch.assert_called_once_with(10)
        http_agent.stop.assert_called_once()

    def test_worker_watcher_stops_for_done_or_stale_heartbeat(self):
        runner = self._worker_runner()
        (self.agent_dir / lifecycle.RANK0_DONE_FILENAME).write_text("\n")
        self.assertEqual(runner._watch(heartbeat_interval=1), 0)
        (self.agent_dir / lifecycle.RANK0_DONE_FILENAME).unlink()
        self.assertEqual(runner._watch(heartbeat_interval=0), 1)

    @patch("cvs.core.agent.lifecycle.ClusterFile")
    @patch("cvs.core.agent.lifecycle.HttpAgentServer")
    @patch("cvs.core.agent.lifecycle.scheduler_rank", return_value=(0, 2))
    @patch("cvs.core.agent.lifecycle.scheduler_hosts", return_value=["node01", "node02"])
    def test_agent_runner_rank0_flow(self, _hosts, _rank, mock_http_agent, mock_managed_cluster):
        layout = MagicMock(agent_dir=self.agent_dir)
        http_agent = mock_http_agent.return_value
        http_agent.host = "node01"
        http_agent.wait_until_ready.return_value = 9000
        http_agent.wait_for_registrations.return_value = {0: MagicMock(), 1: MagicMock()}
        managed = mock_managed_cluster.return_value
        managed.create.return_value = "/run/cluster_agents.json"

        overlay = self.agent_dir / "cluster.json"
        overlay.write_text('{"username": "operator"}', encoding="utf-8")
        runner = lifecycle.AgentRunner(layout, cluster_file=str(overlay))
        self.assertTrue(runner.is_rank0)
        self.assertIsNone(runner.start())
        path = runner.wait()
        runner.stop()

        mock_http_agent.assert_called_once_with(self.agent_dir, 0, 2, host="node01")
        mock_managed_cluster.assert_called_once_with(["node01", "node02"], layout, {"username": "operator"})
        managed.create.assert_called_once_with(http_agent.wait_for_registrations.return_value)
        self.assertEqual(path, "/run/cluster_agents.json")
        http_agent.stop.assert_called_once()


class TestClusterFile(unittest.TestCase):
    def test_cluster_file_cannot_add_nodes(self):
        layout = SimpleNamespace(run_dir=Path("/tmp"), agent_dir=Path("/tmp/agent"))
        cluster_file = lifecycle.ClusterFile(["node01"], layout, {"node_dict": {"node02": {}}})
        with self.assertRaisesRegex(ValueError, "outside this scheduler job"):
            cluster_file.build()

    def test_uses_scheduler_hosts_and_cluster_metadata(self):
        layout = SimpleNamespace(run_dir=Path("/tmp"), agent_dir=Path("/tmp/agent"))
        cluster_file = lifecycle.ClusterFile(
            ["node01", "node02"],
            layout,
            {
                "username": "operator",
                "container": {"image": "image"},
                "node_dict": {"node02": {"bmc_ip": "192.0.2.2"}},
            },
        )
        cluster = cluster_file.build()
        self.assertEqual(list(cluster["node_dict"]), ["node01", "node02"])
        self.assertEqual(cluster["node_dict"]["node01"]["vpc_ip"], "node01")
        self.assertEqual(cluster["node_dict"]["node02"]["bmc_ip"], "192.0.2.2")
        self.assertEqual(cluster["head_node_dict"], {"mgmt_ip": "node01"})
        self.assertEqual(cluster["username"], "operator")

    def test_create_annotates_ports_and_token_path(self):
        with tempfile.TemporaryDirectory() as root:
            layout = type(
                "Layout",
                (),
                {
                    "run_dir": Path(root) / "run",
                    "agent_dir": Path(root) / "run" / "agent",
                },
            )()
            layout.run_dir.mkdir(parents=True)
            layout.agent_dir.mkdir()
            cluster_file = lifecycle.ClusterFile(["node01", "node02"], layout)
            snapshot = {
                0: SimpleNamespace(hostname="node01", port=9000),
                1: SimpleNamespace(hostname="node02", port=9001),
            }
            path = cluster_file.create(snapshot)
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["node_dict"]["node01"]["agent_port"], 9000)
            self.assertEqual(payload["agent_token_file"], str(layout.agent_dir / "secret"))

    def test_create_fails_when_agent_missing(self):
        with tempfile.TemporaryDirectory() as root:
            layout = type(
                "Layout",
                (),
                {
                    "run_dir": Path(root) / "run",
                    "agent_dir": Path(root) / "run" / "agent",
                },
            )()
            layout.run_dir.mkdir(parents=True)
            cluster_file = lifecycle.ClusterFile(["node01", "node02"], layout)
            snapshot = {0: SimpleNamespace(hostname="node01", port=9000)}
            with self.assertRaisesRegex(ValueError, "did not register"):
                cluster_file.create(snapshot)


if __name__ == "__main__":
    unittest.main()
