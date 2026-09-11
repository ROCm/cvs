import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.agent import messages, uds
from cvs.core.agent.uds import UdsLaunchCoordinator, UdsWorker


class TestUdsLaunchCoordinator(unittest.IsolatedAsyncioTestCase):
    async def test_fans_out_to_coordinator_and_uds_worker(self):
        with tempfile.TemporaryDirectory() as root:
            socket_path = Path(root) / "agent.sock"
            manager = UdsLaunchCoordinator(global_rank=0, expected_workers=1, socket_path=socket_path)
            await manager.start()
            worker = asyncio.create_task(UdsWorker(global_rank=1, socket_path=socket_path).start())
            try:
                await manager.wait_until_ready(timeout=2)
                request = messages.LaunchRequest(
                    argv=["bash", "-c", 'printf "%s" "$PMIX_RANK"'],
                    env={},
                    cwd=Path(root),
                    timeout=5,
                    launch_id="two",
                    out_path=Path(root) / "output",
                    world_size=2,
                )
                coordinator_env = {
                    "PMIX_RANK": "0",
                    "PMIX_SIZE": "2",
                }
                # The real workers have distinct inherited PMIX_RANK values. This in-process
                # protocol test checks fan-out/result folding; rank-scoped files prove both ran.
                with patch.dict(os.environ, coordinator_env, clear=False):
                    response = await manager.launch(request)
                self.assertEqual([result.rank for result in response.results], [0, 1])
                self.assertTrue(all(result.exit_code == 0 for result in response.results))
                self.assertTrue((request.out_path / "rank-0000.stdout").exists())
                self.assertTrue((request.out_path / "rank-0001.stdout").exists())
            finally:
                await manager.stop()
                await asyncio.wait_for(worker, timeout=2)

    async def test_idle_cancel_does_not_drop_the_worker(self):
        with tempfile.TemporaryDirectory() as root:
            socket_path = Path(root) / "agent.sock"
            manager = UdsLaunchCoordinator(global_rank=0, expected_workers=1, socket_path=socket_path)
            await manager.start()
            worker = asyncio.create_task(UdsWorker(global_rank=1, socket_path=socket_path).start())
            try:
                await manager.wait_until_ready(timeout=2)
                idle = next(iter(manager._workers.values()))
                idle.writer.write(b'{"kind":"cancel"}\n')
                await idle.writer.drain()
                request = messages.LaunchRequest(
                    argv=["true"],
                    env={},
                    cwd=Path(root),
                    timeout=5,
                    launch_id="idle-cancel",
                    out_path=Path(root) / "output",
                    world_size=2,
                )
                response = await manager.launch(request)
                self.assertEqual([result.rank for result in response.results], [0, 1])
                self.assertTrue(all(result.exit_code == 0 for result in response.results))
            finally:
                await manager.stop()
                await asyncio.wait_for(worker, timeout=2)

    async def test_launch_fails_fast_when_a_worker_never_starts_its_child(self):
        with tempfile.TemporaryDirectory() as root:
            socket_path = Path(root) / "agent.sock"
            manager = UdsLaunchCoordinator(global_rank=0, expected_workers=1, socket_path=socket_path)
            await manager.start()
            # Registers like a worker but never acks a spawn, i.e. its rank's child never started.
            _, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(b'{"kind": "register", "rank": 1}\n')
            await writer.drain()
            try:
                await manager.wait_until_ready(timeout=2)
                request = messages.LaunchRequest(
                    argv=["sleep", "30"],
                    env={},
                    cwd=Path(root),
                    timeout=60,
                    launch_id="three",
                    out_path=Path(root) / "output",
                    world_size=2,
                )
                with patch.object(uds, "LAUNCH_SPAWN_TIMEOUT_SECONDS", 1):
                    with self.assertRaisesRegex(TimeoutError, r"local ranks \[1\]"):
                        await manager.launch(request)
            finally:
                writer.close()
                await manager.stop()

    async def test_launch_fails_immediately_when_a_ready_worker_disconnects(self):
        with tempfile.TemporaryDirectory() as root:
            socket_path = Path(root) / "agent.sock"
            manager = UdsLaunchCoordinator(global_rank=0, expected_workers=1, socket_path=socket_path)
            await manager.start()
            _, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(b'{"kind": "register", "rank": 1}\n')
            await writer.drain()
            try:
                await manager.wait_until_ready(timeout=2)
                writer.close()

                async def dropped():
                    while manager._workers:
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(dropped(), timeout=2)
                request = messages.LaunchRequest(
                    argv=["true"],
                    env={},
                    cwd=Path(root),
                    timeout=5,
                    launch_id="four",
                    out_path=Path(root) / "output",
                    world_size=2,
                )
                with self.assertRaisesRegex(RuntimeError, r"local UDS workers 0/1"):
                    await manager.launch(request)
            finally:
                await manager.stop()

    async def test_start_skips_uds_when_no_local_workers(self):
        with tempfile.TemporaryDirectory() as root:
            socket_path = Path(root) / "agent.sock"
            manager = UdsLaunchCoordinator(global_rank=0, expected_workers=0, socket_path=socket_path)
            await manager.start()
            try:
                self.assertFalse(socket_path.exists())
                self.assertIsNone(manager._server)
            finally:
                await manager.stop()


if __name__ == "__main__":
    unittest.main()
