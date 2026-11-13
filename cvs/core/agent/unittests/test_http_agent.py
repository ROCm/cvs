'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/agent/http_agent.py: AgentRegistry bookkeeping/readiness,
# the bearer-token auth dependency, /v1/exec out_path confinement, and the /v1/register,
# /v1/health route behavior.

import asyncio
import os
import signal
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cvs.core.agent import messages
from cvs.core.agent.http_agent import (
    AgentInfo,
    AgentRegistry,
    ExecSpecError,
    _load_exec_spec,
    _resolve_under,
    create_app,
    _terminate_process_group,
)
from cvs.core.agent.http_client import _write_exec_spec
from cvs.unittests.support import TMP


class TestAgentRegistry(unittest.IsolatedAsyncioTestCase):
    async def test_register_stores_agent_info(self):
        registry = AgentRegistry(world_size=2)
        await registry.register(rank=1, hostname="node1", port=9000)
        self.assertEqual(registry.snapshot(), {1: AgentInfo(hostname="node1", port=9000)})

    async def test_reregistering_rank_overwrites_previous_entry(self):
        registry = AgentRegistry(world_size=1)
        await registry.register(rank=1, hostname="node1", port=9000)
        await registry.register(rank=1, hostname="node1", port=9001)
        self.assertEqual(registry.snapshot(), {1: AgentInfo(hostname="node1", port=9001)})

    async def test_wait_until_ready_returns_once_expected_count_reached(self):
        registry = AgentRegistry(world_size=2)
        await registry.register(rank=1, hostname="node1", port=9000)
        await registry.register(rank=2, hostname="node2", port=9000)
        snapshot = await registry.wait_until_ready(timeout=1)
        self.assertEqual(len(snapshot), 2)

    async def test_wait_until_ready_times_out_when_short(self):
        registry = AgentRegistry(world_size=2)
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
    AUTH_FIXTURE = "fixture-auth-123"

    def _make_client(self, world_rank: int, world_size: int = 1) -> TestClient:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.run_dir = Path(tmp_dir.name)
        agent_dir = self.run_dir / "agent"
        agent_dir.mkdir()
        self.agent_dir = agent_dir
        (agent_dir / messages.AUTH_TOKEN_FILENAME).write_text(self.AUTH_FIXTURE + "\n")
        app = create_app(
            agent_dir=agent_dir,
            world_rank=world_rank,
            world_size=world_size,
            own_hostname="rank0-host" if world_rank == 0 else None,
            own_port=9000 if world_rank == 0 else None,
        )
        client = TestClient(app)
        client.__enter__()  # runs the app's lifespan startup so app.state.auth_token is populated
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def _make_out_dir(self):
        out_dir = self.run_dir / "exec_output"
        out_dir.mkdir(exist_ok=True)
        return out_dir

    def _write_spec(self, cmd, env=None, cwd=None):
        spec_dir = self.agent_dir / messages.EXEC_SPEC_DIRNAME
        spec_dir.mkdir(mode=0o700, exist_ok=True)
        spec_id = uuid.uuid4().hex
        _write_exec_spec(spec_dir / f"{spec_id}.json", messages.ExecSpec(cmd=cmd, env=env or {}, cwd=cwd or Path(TMP)))
        return spec_id

    DEFAULT_CMD_ID = "cmd-1"
    DEFAULT_OUTPUT_MODE = messages.ExecOutputMode.EXIT_CODE_ONLY

    def _exec_request(self, cmd="true", env=None, cwd=None, **overrides):
        kwargs = dict(
            spec_id=self._write_spec(cmd, env, cwd),
            timeout=None,
            inactivity_timeout=None,
            cmd_id=self.DEFAULT_CMD_ID,
            out_path=None,
            output_mode=self.DEFAULT_OUTPUT_MODE,
        )
        kwargs.update(overrides)
        return messages.ExecRequest(**kwargs)

    def _auth_headers(self, auth: str | None = None) -> dict[str, str]:
        return {
            messages.AUTH_HEADER: f"{messages.AUTH_SCHEME} {auth if auth is not None else self.AUTH_FIXTURE}",
            "content-type": "application/json",
        }


class TestAuth(HttpAgentTestBase):
    def test_missing_authorization_header_is_rejected(self):
        client = self._make_client(world_rank=0)
        response = client.get(messages.HEALTH_PATH)
        self.assertEqual(response.status_code, 401)

    def test_wrong_token_is_rejected(self):
        client = self._make_client(world_rank=0)
        response = client.get(messages.HEALTH_PATH, headers=self._auth_headers(auth="wrong-auth"))
        self.assertEqual(response.status_code, 401)

    def test_correct_token_is_accepted(self):
        client = self._make_client(world_rank=0)
        response = client.get(messages.HEALTH_PATH, headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(messages.HealthResponse(**response.json()).ok)


class TestRegisterAgent(HttpAgentTestBase):
    def test_rank0_accepts_registration(self):
        client = self._make_client(world_rank=0)
        req = messages.RegisterRequest(rank=1, hostname="node1", port=9000)
        response = client.post(messages.REGISTER_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(messages.RegisterResponse(**response.json()).ok)

    def test_non_rank0_rejects_registration(self):
        client = self._make_client(world_rank=1)
        req = messages.RegisterRequest(rank=2, hostname="node2", port=9000)
        response = client.post(messages.REGISTER_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 403)


class TestRankZeroSelfRegistration(HttpAgentTestBase):
    def _agent_dir(self) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        agent_dir = Path(tmp_dir.name)
        (agent_dir / messages.AUTH_TOKEN_FILENAME).write_text(self.AUTH_FIXTURE + "\n")
        return agent_dir

    def test_rank0_requires_own_hostname_and_port(self):
        with self.assertRaises(ValueError):
            create_app(agent_dir=self._agent_dir(), world_rank=0, world_size=1)

    def test_non_rank0_does_not_require_own_hostname_and_port(self):
        create_app(agent_dir=self._agent_dir(), world_rank=1, world_size=1)

    def test_rank0_self_registers_and_becomes_ready_when_world_size_is_one(self):
        app = create_app(
            agent_dir=self._agent_dir(),
            world_rank=0,
            world_size=1,
            own_hostname="rank0-host",
            own_port=9000,
        )
        with TestClient(app):
            self.assertEqual(app.state.registry.snapshot(), {0: AgentInfo(hostname="rank0-host", port=9000)})

    def test_rank0_startup_does_not_wait_for_workers(self):
        app = create_app(
            agent_dir=self._agent_dir(),
            world_rank=0,
            world_size=2,
            own_hostname="rank0-host",
            own_port=9000,
        )
        with TestClient(app):
            self.assertEqual(app.state.registry.snapshot(), {0: AgentInfo(hostname="rank0-host", port=9000)})


class TestLoadExecSpec(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec_dir = Path(self.tmp.name)
        self.spec_id = "a" * 32
        self.path = self.spec_dir / f"{self.spec_id}.json"

    def _write_valid(self):
        spec = messages.ExecSpec(cmd="echo hello", env={"FOO": "bar"}, cwd=Path(TMP))
        _write_exec_spec(self.path, spec)
        return spec

    def test_accepts_owner_only_spec(self):
        spec = self._write_valid()
        self.assertEqual(_load_exec_spec(self.spec_dir, self.spec_id), spec)

    def test_rejects_missing_file(self):
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, self.spec_id)

    def test_rejects_group_or_other_access(self):
        for mode in (0o640, 0o604):
            with self.subTest(mode=mode):
                self._write_valid()
                self.path.chmod(mode)
                with self.assertRaises(ExecSpecError):
                    _load_exec_spec(self.spec_dir, self.spec_id)
                self.path.unlink()

    def test_rejects_symlink(self):
        target = self.spec_dir / "target.json"
        _write_exec_spec(target, messages.ExecSpec(cmd="true", env={}, cwd=Path(TMP)))
        self.path.symlink_to(target)
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, self.spec_id)

    def test_rejects_directory(self):
        self.path.mkdir()
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, self.spec_id)

    def test_rejects_fifo_without_blocking(self):
        os.mkfifo(self.path)
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, self.spec_id)

    def test_rejects_wrong_owner(self):
        self._write_valid()
        with patch("cvs.core.agent.http_agent.os.geteuid", return_value=os.geteuid() + 1):
            with self.assertRaises(ExecSpecError):
                _load_exec_spec(self.spec_dir, self.spec_id)

    def test_rejects_path_escape(self):
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, "../escape")

    def test_rejects_invalid_json(self):
        self.path.write_text("not json")
        self.path.chmod(0o600)
        with self.assertRaises(ExecSpecError):
            _load_exec_spec(self.spec_dir, self.spec_id)

    def test_validation_error_does_not_expose_spec_contents(self):
        marker = "payload-should-be-hidden"
        self.path.write_text(f'{{"cmd":"echo {marker}","env":{{"PROBE":["{marker}"]}},"cwd":"/tmp"}}')
        self.path.chmod(0o600)
        with self.assertRaises(ExecSpecError) as raised:
            _load_exec_spec(self.spec_dir, self.spec_id)
        self.assertIn("env.PROBE", str(raised.exception))
        self.assertNotIn(marker, str(raised.exception))


class TestResolveUnder(unittest.TestCase):
    def test_accepts_root_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            self.assertEqual(_resolve_under(root, root), root)

    def test_accepts_and_resolves_path_beneath_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            self.assertEqual(_resolve_under(os.path.join(root, "a", "..", "b"), root), os.path.join(root, "b"))

    def test_rejects_dotdot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            with self.assertRaises(ValueError):
                _resolve_under(os.path.join(root, ".."), root)

    def test_rejects_absolute_path_outside_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = os.path.realpath(directory)
            with self.assertRaises(ValueError):
                _resolve_under(outside, root)

    def test_rejects_sibling_sharing_root_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            with self.assertRaises(ValueError):
                _resolve_under(root + "-other", root)

    def test_rejects_symlink_escaping_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = os.path.realpath(directory)
            link = os.path.join(root, "link")
            os.symlink(outside, link)
            with self.assertRaises(ValueError):
                _resolve_under(link, root)

    def test_rejects_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            with self.assertRaises(ValueError):
                _resolve_under("sub", root)

    def test_accepts_pathlib_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.realpath(directory)
            self.assertEqual(_resolve_under(Path(root) / "x", root), os.path.join(root, "x"))


class TestExec(HttpAgentTestBase):
    def test_unknown_spec_id_returns_400_and_spawns_nothing(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request()
        missing = req.model_copy(update={"spec_id": "f" * 32})
        response = client.post(messages.EXEC_PATH, content=missing.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(client.app.state.process_registry.snapshot(), {})
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)

    def test_request_carrying_cmd_is_rejected_with_422(self):
        client = self._make_client(world_rank=0)
        body = self._exec_request().model_dump(mode="json")
        body["cmd"] = "id"
        response = client.post(messages.EXEC_PATH, json=body, headers=self._auth_headers())
        self.assertEqual(response.status_code, 422)

    def test_malformed_spec_id_is_rejected_with_422(self):
        client = self._make_client(world_rank=0)
        body = self._exec_request().model_dump(mode="json")
        body["spec_id"] = "../escape"
        response = client.post(messages.EXEC_PATH, json=body, headers=self._auth_headers())
        self.assertEqual(response.status_code, 422)

    def test_group_readable_spec_is_rejected_with_400(self):
        client = self._make_client(world_rank=0)
        marker = "payload-should-be-hidden"
        req = self._exec_request(cmd=f"echo {marker}")
        (self.agent_dir / messages.EXEC_SPEC_DIRNAME / f"{req.spec_id}.json").chmod(0o640)
        with self.assertLogs("cvs.core.agent.http_agent", level="WARNING") as logs:
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 400)
        self.assertIn(req.spec_id, logs.output[0])
        self.assertIn("open to group or other", logs.output[0])
        self.assertNotIn(marker, logs.output[0])

    def test_rejected_spec_warning_strips_newlines_from_path(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request()
        client.app.state.exec_spec_dir = self.agent_dir / "exec\r\nextra-line"
        with self.assertLogs("cvs.core.agent.http_agent", level="WARNING") as logs:
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 400)
        self.assertIn(req.spec_id, logs.output[0])
        self.assertIn("cannot open exec spec", logs.output[0])
        self.assertNotIn("\r", logs.output[0])
        self.assertNotIn("\n", logs.output[0])

    def test_command_runs_under_bin_sh_like_before(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="echo $0", output_mode=messages.ExecOutputMode.INLINE)
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(messages.ExecResponse(**response.json()).stdout, ["/bin/sh"])

    def test_exit_code_only_reports_success(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="true")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 0)
        self.assertIsNone(exec_response.stdout)
        self.assertIsNone(exec_response.stdout_path)

    def test_exit_code_only_reports_failure(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="false")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 1)

    def test_exit_code_only_does_not_deadlock_on_large_output(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="head -c 1000000 /dev/zero | tr '\\0' 'a'")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(messages.ExecResponse(**response.json()).exit_code, 0)

    def test_file_mode_writes_output_and_returns_preview(self):
        client = self._make_client(world_rank=0)
        out_path = self._make_out_dir()
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
        client = self._make_client(world_rank=0)
        out_path = self._make_out_dir()
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
        client = self._make_client(world_rank=0)
        out_path = self._make_out_dir()
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
        client = self._make_client(world_rank=0)
        with tempfile.TemporaryDirectory() as cwd_dir:
            req = self._exec_request(
                cmd="pwd",
                cmd_id="cmd-cwd",
                cwd=Path(cwd_dir),
                out_path=self._make_out_dir(),
                output_mode=messages.ExecOutputMode.FILE,
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            exec_response = messages.ExecResponse(**response.json())
            self.assertEqual(exec_response.stdout, [cwd_dir])


class TestExecPathConfinement(HttpAgentTestBase):
    DEFAULT_CMD_ID = "cmd-confine"

    def test_out_path_outside_run_dir_is_rejected_without_spawning(self):
        client = self._make_client(world_rank=0)
        marker = self.run_dir / "spawned"
        with tempfile.TemporaryDirectory() as outside:
            req = self._exec_request(
                cmd=f"touch {marker}", out_path=Path(outside), output_mode=messages.ExecOutputMode.FILE
            )
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            self.assertEqual(response.status_code, 403)
            self.assertEqual(list(Path(outside).iterdir()), [])
        self.assertFalse(marker.exists())

    def test_out_path_dotdot_escape_is_rejected(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(out_path=self.run_dir / "..", output_mode=messages.ExecOutputMode.FILE)
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 403)

    def test_out_path_symlink_escape_is_rejected(self):
        client = self._make_client(world_rank=0)
        with tempfile.TemporaryDirectory() as outside:
            link = self.run_dir / "link"
            link.symlink_to(outside)
            req = self._exec_request(out_path=link, output_mode=messages.ExecOutputMode.FILE)
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
            self.assertEqual(response.status_code, 403)
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_relative_out_path_is_rejected(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(out_path=Path("exec_output"), output_mode=messages.ExecOutputMode.FILE)
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 403)

    def test_rejected_request_does_not_leave_agent_busy(self):
        client = self._make_client(world_rank=0)
        with tempfile.TemporaryDirectory() as outside:
            rejected = self._exec_request(out_path=Path(outside), output_mode=messages.ExecOutputMode.FILE)
        response = client.post(messages.EXEC_PATH, content=rejected.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 403)
        accepted = self._exec_request()
        response = client.post(messages.EXEC_PATH, content=accepted.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(messages.ExecResponse(**response.json()).exit_code, 0)

    def test_cmd_id_with_path_traversal_is_rejected(self):
        client = self._make_client(world_rank=0)
        out_path = self._make_out_dir()
        body = self._exec_request(output_mode=messages.ExecOutputMode.FILE, out_path=out_path).model_dump(mode="json")
        body["cmd_id"] = "../escape"
        response = client.post(messages.EXEC_PATH, json=body, headers=self._auth_headers())
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.run_dir / "escape.stdout").exists())


class TestExecTimeouts(HttpAgentTestBase):
    DEFAULT_CMD_ID = "cmd-timeout"

    def test_exit_code_only_completes_normally_within_timeout(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="true", timeout=5)
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertFalse(exec_response.timed_out)
        self.assertEqual(exec_response.exit_code, 0)

    def test_exit_code_only_is_killed_when_timeout_exceeded(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="sleep 30", timeout=1)
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertTrue(exec_response.timed_out)
        self.assertEqual(exec_response.exit_code, -signal.SIGTERM)

    def test_inline_mode_captures_partial_output_when_timeout_exceeded(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(
            cmd="echo partial; sleep 30",
            timeout=1,
            output_mode=messages.ExecOutputMode.INLINE,
        )
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertTrue(exec_response.timed_out)
        self.assertEqual(exec_response.stdout, ["partial"])
        self.assertEqual(exec_response.exit_code, -signal.SIGTERM)

    def test_inline_mode_is_killed_after_falling_silent(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(
            cmd="echo first; sleep 30",
            inactivity_timeout=1,
            output_mode=messages.ExecOutputMode.INLINE,
        )
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertTrue(exec_response.timed_out)
        self.assertEqual(exec_response.stdout, ["first"])
        self.assertEqual(exec_response.exit_code, -signal.SIGTERM)

    def test_inline_mode_does_not_time_out_on_output_that_arrives_within_inactivity_window(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(
            cmd="echo a; sleep 0.3; echo b",
            inactivity_timeout=5,
            output_mode=messages.ExecOutputMode.INLINE,
        )
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertFalse(exec_response.timed_out)
        self.assertEqual(exec_response.stdout, ["a", "b"])
        self.assertEqual(exec_response.exit_code, 0)

    def test_inline_mode_does_not_time_out_when_only_one_stream_stays_active(self):
        # stderr never writes anything, but stdout keeps producing within the inactivity window -
        # inactivity must be judged on combined stream activity, not on each stream independently,
        # or a permanently silent stderr would falsely trigger a kill.
        client = self._make_client(world_rank=0)
        req = self._exec_request(
            cmd="for i in 1 2 3; do echo tick; sleep 0.3; done",
            inactivity_timeout=1,
            output_mode=messages.ExecOutputMode.INLINE,
        )
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertFalse(exec_response.timed_out)
        self.assertEqual(exec_response.stdout, ["tick", "tick", "tick"])
        self.assertEqual(exec_response.stderr, [])
        self.assertEqual(exec_response.exit_code, 0)


class TestInlineExec(HttpAgentTestBase):
    DEFAULT_CMD_ID = "cmd-inline"
    DEFAULT_OUTPUT_MODE = messages.ExecOutputMode.INLINE

    def test_reports_full_stdout_and_stderr_untruncated(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="seq 1 30; echo err 1>&2")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 0)
        self.assertEqual(exec_response.stdout, [str(n) for n in range(1, 31)])
        self.assertEqual(exec_response.stderr, ["err"])
        self.assertIsNone(exec_response.stdout_path)
        self.assertIsNone(exec_response.stderr_path)
        self.assertFalse(exec_response.truncated)

    def test_reports_nonzero_exit_code(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="false")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.exit_code, 1)

    def test_output_beyond_byte_cap_is_tail_truncated(self):
        client = self._make_client(world_rank=0)
        with patch("cvs.core.agent.http_agent.messages.MAX_INLINE_RESPONSE_BYTES", 10):
            req = self._exec_request(cmd="printf '0123456789abcdefghij'")
            response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        exec_response = messages.ExecResponse(**response.json())
        self.assertEqual(exec_response.stdout, ["abcdefghij"])
        self.assertTrue(exec_response.truncated)

    def test_does_not_deadlock_on_large_output(self):
        client = self._make_client(world_rank=0)
        req = self._exec_request(cmd="head -c 1000000 /dev/zero | tr '\\0' 'a'")
        response = client.post(messages.EXEC_PATH, content=req.model_dump_json(), headers=self._auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(messages.ExecResponse(**response.json()).exit_code, 0)


class TestExecConcurrency(HttpAgentTestBase):
    def test_concurrent_exec_is_rejected_with_409_and_flag_resets_after(self):
        client = self._make_client(world_rank=0)
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
        client = self._make_client(world_rank=0)
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
