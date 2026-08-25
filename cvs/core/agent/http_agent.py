'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import asyncio
import hmac
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status

from . import messages

FILE_MODE_PREVIEW_LINES = 20
SHUTDOWN_GRACE_PERIOD_SECONDS = 10.0


def _read_secret(file_path: Path) -> str:
    with open(file_path) as file:
        return file.readline().strip()


async def _write_text(path: Path, content: str) -> None:
    await asyncio.to_thread(path.write_text, content)


def _tail_lines(text: str, max_lines: int) -> list[str]:
    return text.splitlines()[-max_lines:]


def _merged_env(overrides: dict[str, str]) -> dict[str, str]:
    return {**os.environ, **overrides}


@dataclass
class AgentInfo:
    hostname: str
    port: int


class AgentRegistry:
    '''In-memory record of registered agents, keyed by rank. Lives on rank 0 only.'''

    def __init__(self, expected_count: int) -> None:
        self._agents: dict[int, AgentInfo] = {}
        self._lock = asyncio.Lock()
        self._expected_count: int = expected_count
        self._all_registered = asyncio.Event()

    async def register(self, rank: int, hostname: str, port: int) -> None:
        async with self._lock:
            # last write wins: a re-registering rank (e.g. restarted with a new port) replaces its old entry
            self._agents[rank] = AgentInfo(hostname, port)
            if len(self._agents) >= self._expected_count:
                self._all_registered.set()

    def snapshot(self) -> dict[int, AgentInfo]:
        return dict(self._agents)

    async def wait_until_ready(self, timeout: float | None = None) -> dict[int, AgentInfo]:
        await asyncio.wait_for(self._all_registered.wait(), timeout=timeout)
        return self.snapshot()


class ProcessRegistry:
    '''In-memory record of processes spawned via /v1/exec, keyed by cmd_id. Every rank's own agent has one,
    since a spawned process only exists in that rank's local kernel process table.'''

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    async def register(self, cmd_id: str, process: asyncio.subprocess.Process) -> None:
        async with self._lock:
            self._processes[cmd_id] = process

    async def unregister(self, cmd_id: str) -> None:
        async with self._lock:
            self._processes.pop(cmd_id, None)

    def snapshot(self) -> dict[str, asyncio.subprocess.Process]:
        return dict(self._processes)


async def _terminate_process_group(process: asyncio.subprocess.Process, grace_period: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return  # already exited (e.g. its own /v1/exec call completed concurrently)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_period)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _spawn_process(
    request: messages.ExecRequest,
    registry: ProcessRegistry,
    stdout: int,
    stderr: int,
) -> asyncio.subprocess.Process:
    # start_new_session detaches the child into its own process group so a future kill can target
    # the whole group (via os.killpg) without also signaling this agent process itself.
    process = await asyncio.create_subprocess_shell(
        request.cmd,
        stdout=stdout,
        stderr=stderr,
        cwd=request.cwd,
        env=_merged_env(request.env),
        start_new_session=True,
    )
    await registry.register(request.cmd_id, process)
    return process


async def _run_cmd(request: messages.ExecRequest, registry: ProcessRegistry) -> messages.ExecResponse:
    if request.output_mode == messages.ExecOutputMode.EXIT_CODE_ONLY:
        process = await _spawn_process(
            request, registry, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            await process.wait()
        finally:
            await registry.unregister(request.cmd_id)
        return messages.ExecResponse(
            exit_code=process.returncode,
            stdout=None,
            stderr=None,
            stdout_path=None,
            stderr_path=None,
            truncated=None,
        )

    if request.output_mode == messages.ExecOutputMode.FILE:
        process = await _spawn_process(
            request, registry, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout_bytes, stderr_bytes = await process.communicate()
        finally:
            await registry.unregister(request.cmd_id)
        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")
        stdout_path = None
        stderr_path = None
        if request.out_path:
            stdout_path = request.out_path / f"{request.cmd_id}.stdout"
            stderr_path = request.out_path / f"{request.cmd_id}.stderr"
            await _write_text(stdout_path, stdout_text)
            await _write_text(stderr_path, stderr_text)
        return messages.ExecResponse(
            exit_code=process.returncode,
            stdout=_tail_lines(stdout_text, FILE_MODE_PREVIEW_LINES),
            stderr=_tail_lines(stderr_text, FILE_MODE_PREVIEW_LINES),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            truncated=None,
        )

    raise NotImplementedError(f"output_mode {request.output_mode} is not yet implemented")


def _extract_bearer_token(header_value: str | None) -> str | None:
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    return token if scheme == messages.AUTH_SCHEME and token else None


async def verify_auth(request: Request) -> None:
    provided = _extract_bearer_token(request.headers.get(messages.AUTH_HEADER))
    if provided is None or not hmac.compare_digest(provided, request.app.state.auth_token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid or missing bearer token")


def create_app(agent_dir: Path, own_rank: int, expected_agent_count: int) -> FastAPI:
    '''Build a FastAPI app for one rank's agent process. own_rank gates /v1/register to rank 0 only.'''

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_token = _read_secret(agent_dir / messages.AUTH_TOKEN_FILENAME)
        yield

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_auth)])
    app.state.own_rank = own_rank
    app.state.registry = AgentRegistry(expected_agent_count)
    app.state.process_registry = ProcessRegistry()
    app.state.exec_busy = False

    @app.post(messages.REGISTER_PATH)
    async def register_agent(request: messages.RegisterRequest, http_request: Request) -> messages.RegisterResponse:
        '''Record that a worker rank has started and is reachable at hostname:port'''
        if http_request.app.state.own_rank != 0:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="only rank 0 accepts registrations")
        await http_request.app.state.registry.register(request.rank, request.hostname, request.port)
        return messages.RegisterResponse(ok=True)

    @app.get(messages.HEALTH_PATH)
    async def get_health_status() -> messages.HealthResponse:
        '''Return status of agent connection health'''
        return messages.HealthResponse(ok=True)

    @app.post(messages.EXEC_PATH)
    async def run_cmd(request: messages.ExecRequest, http_request: Request) -> messages.ExecResponse:
        '''Run cmd on host'''
        # Checked and set with no `await` in between: the event loop can't switch to another
        # request's coroutine mid-way, so this is race-free without needing an explicit lock.
        if http_request.app.state.exec_busy:
            raise HTTPException(status.HTTP_409_CONFLICT, detail="an exec is already in progress on this agent")
        http_request.app.state.exec_busy = True
        try:
            return await _run_cmd(request, http_request.app.state.process_registry)
        finally:
            http_request.app.state.exec_busy = False

    @app.post(messages.SHUTDOWN_PATH)
    async def run_shutdown(http_request: Request) -> messages.ShutdownResponse:
        '''Terminate every process this agent has spawned, then exit the agent itself'''
        registry: ProcessRegistry = http_request.app.state.process_registry
        processes = registry.snapshot()
        await asyncio.gather(
            *(_terminate_process_group(process, SHUTDOWN_GRACE_PERIOD_SECONDS) for process in processes.values())
        )
        # Self-signal rather than depending on a Server reference: uvicorn installs a SIGTERM
        # handler that drains in-flight requests (this one included) before exiting.
        os.kill(os.getpid(), signal.SIGTERM)
        return messages.ShutdownResponse(ok=True)

    return app
