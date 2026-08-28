'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import atexit
import asyncio
import os
import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from . import messages
from .http_agent import AgentInfo, create_app

RANK0_HOST_FILENAME = "rank0.host"
RANK0_PORT_FILENAME = "rank0.port"
RANK0_ALIVE_FILENAME = "rank0.alive"
RANK0_HEARTBEAT_INTERVAL_FILENAME = "rank0.heartbeat_interval"
RANK0_DONE_FILENAME = "rank0.done"
STARTUP_TIMEOUT_SECONDS = 30
BOOTSTRAP_TIMEOUT_SECONDS = 60
REGISTRATION_TIMEOUT_SECONDS = 60
HEARTBEAT_INTERVAL_SECONDS = 10
POLL_INTERVAL_SECONDS = 1


def _write_file(path: Path, value: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except OSError:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def managed_rank() -> tuple[int, int]:
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("managed CVS run requires SLURM_PROCID and SLURM_NTASKS") from exc
    if not 0 <= rank < world_size:
        raise RuntimeError(f"invalid managed rank {rank} for world size {world_size}")
    return rank, world_size


def clear_rank0_markers(agent_dir: Path) -> None:
    for filename in (
        messages.AUTH_TOKEN_FILENAME,
        RANK0_HOST_FILENAME,
        RANK0_PORT_FILENAME,
        RANK0_ALIVE_FILENAME,
        RANK0_HEARTBEAT_INTERVAL_FILENAME,
        RANK0_DONE_FILENAME,
    ):
        (agent_dir / filename).unlink(missing_ok=True)


def write_auth_token(agent_dir: Path) -> None:
    _write_file(agent_dir / messages.AUTH_TOKEN_FILENAME, f"{secrets.token_hex(32)}\n", mode=0o600)


def publish_rank0(agent_dir: Path, host: str, port: int, heartbeat_interval: float) -> None:
    _write_file(agent_dir / RANK0_HOST_FILENAME, f"{host}\n")
    _write_file(agent_dir / RANK0_HEARTBEAT_INTERVAL_FILENAME, f"{heartbeat_interval}\n")
    _write_file(agent_dir / RANK0_PORT_FILENAME, f"{port}\n")
    touch_alive(agent_dir)


def read_rank0_rendezvous(agent_dir: Path) -> tuple[str, int, float]:
    host = _read_file(agent_dir / RANK0_HOST_FILENAME)
    port = int(_read_file(agent_dir / RANK0_PORT_FILENAME))
    heartbeat_interval = float(_read_file(agent_dir / RANK0_HEARTBEAT_INTERVAL_FILENAME))
    if not host or not 0 < port <= 65535 or heartbeat_interval <= 0:
        raise ValueError("invalid rank-0 rendezvous")
    return host, port, heartbeat_interval


def touch_alive(agent_dir: Path) -> None:
    (agent_dir / RANK0_ALIVE_FILENAME).touch()


def mark_done(agent_dir: Path) -> None:
    _write_file(agent_dir / RANK0_DONE_FILENAME, "\n")


class Heartbeat:
    def __init__(self, agent_dir: Path, interval: float):
        self._agent_dir = agent_dir
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 1)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            touch_alive(self._agent_dir)


class AgentRunner:
    '''Run the AIMVT-302 application in the background for one scheduler rank.'''

    def __init__(self, agent_dir: Path, rank: int, world_size: int, host: str | None = None):
        self.host = host or socket.gethostname()
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", 0))
        self._socket.listen(2048)
        self.port = self._socket.getsockname()[1]
        self._app = create_app(
            agent_dir,
            rank,
            world_size,
            own_hostname=self.host if rank == 0 else None,
            own_port=self.port if rank == 0 else None,
        )
        self._server = uvicorn.Server(uvicorn.Config(self._app, log_level="warning"))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._event_loop = None

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        await self._server.serve(sockets=[self._socket])

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started and self._event_loop is not None:
                return self.port
            if not self._thread.is_alive():
                raise RuntimeError("HTTP agent stopped before becoming ready")
            time.sleep(0.01)
        raise TimeoutError("HTTP agent did not become ready")

    def wait_for_registrations(self, timeout: float) -> dict[int, AgentInfo]:
        future = asyncio.run_coroutine_threadsafe(self._app.state.registry.wait_until_ready(timeout), self._event_loop)
        return future.result(timeout=timeout + 1)

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        else:
            self._socket.close()


class Rank0Coordinator:
    def __init__(self, agent_dir: Path, agent: AgentRunner, heartbeat: Heartbeat):
        self._agent_dir = agent_dir
        self._agent = agent
        self._heartbeat = heartbeat
        self._closed = False

    def wait_for_registrations(self, timeout: float) -> dict:
        return self._agent.wait_for_registrations(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat.stop()
        mark_done(self._agent_dir)
        self._agent.stop()


def start_rank0(
    agent_dir: Path,
    world_size: int,
    *,
    agent_factory=AgentRunner,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
) -> Rank0Coordinator:
    clear_rank0_markers(agent_dir)
    write_auth_token(agent_dir)
    agent = agent_factory(agent_dir, 0, world_size)
    try:
        agent.start()
        port = agent.wait_until_ready(startup_timeout)
        publish_rank0(agent_dir, agent.host, port, heartbeat_interval)
        heartbeat = Heartbeat(agent_dir, heartbeat_interval)
        heartbeat.start()
        coordinator = Rank0Coordinator(agent_dir, agent, heartbeat)
        atexit.register(coordinator.close)
        return coordinator
    except Exception:
        mark_done(agent_dir)
        agent.stop()
        raise


def register_worker(endpoint: str, token: str, rank: int, host: str, port: int, timeout: float) -> None:
    request = messages.RegisterRequest(rank=rank, hostname=host, port=port)
    deadline = time.monotonic() + timeout
    headers = {messages.AUTH_HEADER: f"{messages.AUTH_SCHEME} {token}"}
    with httpx.Client(headers=headers) as client:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = client.post(
                    f"{endpoint}{messages.REGISTER_PATH}",
                    json=request.model_dump(),
                    timeout=min(STARTUP_TIMEOUT_SECONDS, remaining),
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
            except httpx.TransportError:
                pass
            time.sleep(min(POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))
    raise TimeoutError("could not register with rank-0 HTTP agent")


def wait_for_rendezvous(agent_dir: Path, timeout: float) -> tuple[str, int, float, str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            host, port, heartbeat_interval = read_rank0_rendezvous(agent_dir)
            return host, port, heartbeat_interval, _read_file(agent_dir / messages.AUTH_TOKEN_FILENAME)
        except (OSError, ValueError):
            time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError("rank-0 rendezvous did not appear")


def watch_rank0(agent_dir: Path, heartbeat_interval: float) -> int:
    last_progress = time.monotonic()
    last_mtime_ns = None
    while True:
        if (agent_dir / RANK0_DONE_FILENAME).exists():
            return 0
        try:
            mtime_ns = (agent_dir / RANK0_ALIVE_FILENAME).stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        if mtime_ns is not None and mtime_ns != last_mtime_ns:
            last_mtime_ns = mtime_ns
            last_progress = time.monotonic()
        if time.monotonic() - last_progress >= 3 * heartbeat_interval:
            return 1
        time.sleep(POLL_INTERVAL_SECONDS)


def run_worker(
    agent_dir: Path,
    rank: int,
    world_size: int,
    *,
    agent_factory=AgentRunner,
    bootstrap_timeout: float = BOOTSTRAP_TIMEOUT_SECONDS,
    startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    registration_timeout: float = REGISTRATION_TIMEOUT_SECONDS,
) -> int:
    host, rank0_port, heartbeat_interval, token = wait_for_rendezvous(agent_dir, bootstrap_timeout)
    agent = agent_factory(agent_dir, rank, world_size)
    try:
        agent.start()
        port = agent.wait_until_ready(startup_timeout)
        register_worker(
            f"http://{host}:{rank0_port}",
            token,
            rank,
            agent.host,
            port,
            registration_timeout,
        )
        return watch_rank0(agent_dir, heartbeat_interval)
    finally:
        agent.stop()
