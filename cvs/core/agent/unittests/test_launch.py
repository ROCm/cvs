import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvs.core.agent import messages
from cvs.core.agent.launch import run_launch_child


class TestLaunchChild(unittest.IsolatedAsyncioTestCase):
    async def test_inherits_parent_env_and_returns_child_code_without_exiting(self):
        with tempfile.TemporaryDirectory() as root:
            out_path = Path(root) / "output"
            request = messages.LaunchRequest(
                argv=["bash", "-c", 'printf "%s" "$PMIX_RANK"; exit 7'],
                env={},
                cwd=Path(root),
                timeout=5,
                launch_id="one",
                out_path=out_path,
                world_size=1,
            )
            with patch.dict(os.environ, {"PMIX_RANK": "0"}, clear=False):
                result = await run_launch_child(request, rank=0)

            self.assertEqual(result.exit_code, 7)
            self.assertFalse(result.timed_out)
            self.assertEqual(result.stdout_path.read_text(), "0")


if __name__ == "__main__":
    unittest.main()
