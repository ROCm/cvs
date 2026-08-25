'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/agent/http_agent.py: AgentRegistry bookkeeping/readiness,
# the bearer-token auth dependency, and the /v1/register, /v1/health route behavior.

import asyncio
import os
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cvs.core.agent import messages
from cvs.core.agent.http_agent import AgentInfo, AgentRegistry, create_app, _terminate_process_group


class TestAgentRegistry(unittest.IsolatedAsyncioTestCase):
    async def test_register_stores_agent_info(self):
        registry = AgentRegistry(expected_count=2)
        await registry.register(rank=1, hostname="node1", port=9000)
        self.assertEqual(registry.snapshot(), {1: AgentInfo(hostname="node1", port=9000)})

    async def test_reregistering_rank_overwrites_previous_entry(self):
        registry = AgentRegistry(expected_count=1)
        await registry.register(rank=1, hostname="node1", port=9000)
        await registry.register(rank=1, hostname="node1", port=9001)
        self.assertEqual(registry.snapshot(), {1: AgentInfo(hostname="node1", port=9001)})

    async def test_wait_until_ready_returns_once_expected_count_reached(self):
        registry = AgentRegistry(expected_count=2)
        await registry.register(rank=1, hostname="node1", port=9000)
        await registry.register(rank=2, hostname="node2", port=9000)
        snapshot = await registry.wait_until_ready(timeout=1)
        self.assertEqual(len(snapshot), 2)

    async def test_wait_until_ready_times_out_when_short(self):
        registry = AgentRegistry(expected_count=2)
        await registry.register(rank=1, hostname="node1", port=9000)
        with self.assertRaises(asyncio.TimeoutError):
            await registry.wait_until_ready(timeout=0.05)


class TestTerminateProcessGroup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_sigterm_and_awaits_exit(self):
        process = await asyncio.create_subprocess_shell("sleep 30", start_new_session=True)
        await _terminate_process_group(process, grace_period=5)
        self.assertEqual(process.returncode, -signal.SIGTERM)

    async def test_escalates_to_sigkill_when_process_ignores_sigterm(self):
        process = await asyncio.create_subprocess_shell(
            "trap '' TERM; echo ready; sleep 30", stdout=asyncio.subprocess.PIPE, start_new_session=True
        )
        await process.stdout.readline()  # ensures the trap is installed before we signal the group
        await _terminate_process_group(process, grace_period=0.2)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    async def test_no_error_when_process_already_exited(self):
        process = await asyncio.create_subprocess_shell("true", start_new_session=True)
        await process.wait()
        await _terminate_process_group(process, grace_period=1)


class HttpAgentTestBase(unittest.TestCase):
    TOKEN = "test-token-123"

    def _make_client(self, own_rank: int, expected_agent_count: int = 1) -> TestClient:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        agent_dir = Path(tmp_dir.name)
        (agent_dir / messages.AUTH_TOKEN_FILENAME).write_text(self.TOKEN + "\n")
        app = create_app(agent_dir=agent_dir, own_rank=own_rank, expected_agent_count=expected_agent_count)
        client = TestClient(app)
        client.__enter__()  # runs the app's lifespan startup so app.state.auth_token is populated
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def _auth_headers(self, token: str | None = None) -> dict[str, str]:
        return {
            messages.AUTH_HEADER: f"{messages.AUTH_SCHEME} {token if token is not None else self.TOKEN}",
            "content-type": "application/json",
        }


class TestAuth(HttpAgentTestBase):
    def test_missing_authorization_header_is_rejected(self):
        client = self._make_client(own_rank=0)
        response = client.get(messages.HEALTH_PATH)
        self.assertEqual(response.status_code, 401)

    def test_wrong_token_is_rejected(self):
        client = self._make_client(own_rank=0)
        response = client.get(messages.HEALTH_PATH, headers=self._auth_headers(token="wrong-token"))
        self.assertEqual(response.status_code, 401)

    def test_correct_token_is_accepted(self):
        client = self._make_client(own_rank=0)
        response = client.get(messages.HEALTH_PATH, headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(messages.HealthResponse(**response.json()).ok)


class TestRegisterAgent(HttpAgentTestBase):
    def test_rank0_accepts_registration(self):
        client = self._make_client(own_rank=0)
        req = messages.RegisterRequest(rank=1, hostname="node1", port=9000)
        response = client.post(messages.REGISTER_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(messages.RegisterResponse(**response.json()).ok)

    def test_non_rank0_rejects_registration(self):
        client = self._make_client(own_rank=1)
        req = messages.RegisterRequest(rank=2, hostname="node2", port=9000)
        response = client.post(messages.REGISTER_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 403)


class TestExec(HttpAgentTestBase):
    def _exec_request(self, **overrides) -> messages.ExecRequest:
        kwargs = dict(
            cmd="true",
            env={},
            cwd=Path("/tmp"),
            timeout=None,
            inactivity_timeout=None,
            cmd_id="cmd-1",
            out_path=None,
            output_mode=messages.ExecOutputMode.EXIT_CODE_ONLY,
        )
        kwargs.update(overrides)
        return messages.ExecRequest(**kwargs)

    def test_exit_code_only_reports_success(self):
        client = self._make_client(own_rank=0)
        req = self._exec_request(cmd="true")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 0)
        self.assertIsNone(exec_response.stdout)
        self.assertIsNone(exec_response.stdout_path)

    def test_exit_code_only_reports_failure(self):
        client = self._make_client(own_rank=0)
        req = self._exec_request(cmd="false")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 1)

    def test_exit_code_only_does_not_deadlock_on_large_output(self):
        client = self._make_client(own_rank=0)
        req = self._exec_request(cmd="head -c 1000000 /dev/zero | tr '\\0' 'a'")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(messages.ExecResponse(**response.json()).exit_code, 0)

    def test_file_mode_writes_output_and_returns_preview(self):
        client = self._make_client(own_rank=0)
        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir)
            req = self._exec_request(
                cmd="echo hello; echo world 1>&2",
                cmd_id="cmd-file",
                out_path=out_path,
                output_mode=messages.ExecOutputMode.FILE,
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            self.assertEqual(response.status_code, 200)
            exec_response = messages.ExecResponse(**response.json())
            self.assertEqual(exec_response.exit_code, 0)
            self.assertEqual(exec_response.stdout, ["hello"])
            self.assertEqual(exec_response.stderr, ["world"])
            self.assertEqual((out_path / "cmd-file.stdout").read_text(), "hello\n")
            self.assertEqual((out_path / "cmd-file.stderr").read_text(), "world\n")

    def test_file_mode_previews_only_trailing_lines(self):
        client = self._make_client(own_rank=0)
        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir)
            req = self._exec_request(
                cmd="seq 1 30",
                cmd_id="cmd-tail",
                out_path=out_path,
                output_mode=messages.ExecOutputMode.FILE,
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            exec_response = messages.ExecResponse(**response.json())
            self.assertEqual(exec_response.stdout, [str(n) for n in range(11, 31)])
            self.assertEqual((out_path / "cmd-tail.stdout").read_text(), "\n".join(str(n) for n in range(1, 31)) + "\n")

    def test_env_is_passed_to_command(self):
        client = self._make_client(own_rank=0)
        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir)
            req = self._exec_request(
                cmd="echo $MY_TEST_VAR",
                cmd_id="cmd-env",
                out_path=out_path,
                output_mode=messages.ExecOutputMode.FILE,
                env={"MY_TEST_VAR": "hello-env"},
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            exec_response = messages.ExecResponse(**response.json())
            self.assertEqual(exec_response.stdout, ["hello-env"])

    def test_cwd_is_passed_to_command(self):
        client = self._make_client(own_rank=0)
        with tempfile.TemporaryDirectory() as cwd_dir, tempfile.TemporaryDirectory() as out_dir:
            req = self._exec_request(
                cmd="pwd",
                cmd_id="cmd-cwd",
                cwd=Path(cwd_dir),
                out_path=Path(out_dir),
                output_mode=messages.ExecOutputMode.FILE,
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            exec_response = messages.ExecResponse(**response.json())
            self.assertEqual(exec_response.stdout, [cwd_dir])


class TestExecConcurrency(HttpAgentTestBase):
    def _exec_request(self, **overrides) -> messages.ExecRequest:
        kwargs = dict(
            cmd="true",
            env={},
            cwd=Path("/tmp"),
            timeout=None,
            inactivity_timeout=None,
            cmd_id="cmd-1",
            out_path=None,
            output_mode=messages.ExecOutputMode.EXIT_CODE_ONLY,
        )
        kwargs.update(overrides)
        return messages.ExecRequest(**kwargs)

    def test_concurrent_exec_is_rejected_with_409_and_flag_resets_after(self):
        client = self._make_client(own_rank=0)
        with tempfile.TemporaryDirectory() as marker_dir:
            # the marker file is only written once /v1/exec has spawned the process, which happens
            # strictly after exec_busy is set, so waiting for it rules out a race against the second request
            marker_path = Path(marker_dir) / "started"
            first_req = self._exec_request(cmd=f"touch {marker_path}; sleep 1", cmd_id="cmd-slow")
            responses = {}

            def run_first():
                responses["first"] = client.post(
                    messages.EXEC_PATH, content=first_req.model_dump_json(), headers=self._auth_headers()
                )

            first_thread = threading.Thread(target=run_first)
            first_thread.start()
            deadline = time.monotonic() + 5
            while not marker_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker_path.exists(), "first exec never started")

            second_req = self._exec_request(cmd="true", cmd_id="cmd-second")
            second_response = client.post(
                messages.EXEC_PATH, content=second_req.model_dump_json(), headers=self._auth_headers()
            )
            self.assertEqual(second_response.status_code, 409)

            first_thread.join(timeout=5)
            self.assertEqual(responses["first"].status_code, 200)
            self.assertEqual(messages.ExecResponse(**responses["first"].json()).exit_code, 0)

            third_req = self._exec_request(cmd="true", cmd_id="cmd-third")
            third_response = client.post(
                messages.EXEC_PATH, content=third_req.model_dump_json(), headers=self._auth_headers()
            )
            self.assertEqual(third_response.status_code, 200)


class TestShutdown(HttpAgentTestBase):
    def test_shutdown_with_no_running_processes_signals_self_and_returns_ok(self):
        client = self._make_client(own_rank=0)
        with patch("cvs.core.agent.http_agent.os.kill") as mock_kill:
            response = client.post(
                messages.SHUTDOWN_PATH,
                content=messages.ShutdownRequest().model_dump_json(),
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(messages.ShutdownResponse(**response.json()).ok)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
