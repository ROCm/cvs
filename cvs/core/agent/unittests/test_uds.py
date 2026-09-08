import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.agent import messages
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
