'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status

from . import messages


def _read_secret(file_path: Path) -> str:
    with open(file_path) as file:
        return file.readline().strip()


@dataclass
class AgentInfo:
    hostname: str
    port: int


class AgentRegistry:
    '''In-memory record of registered agents, keyed by rank. Lives on rank 0 only.'''

    def __init__(self, expected_count: int) -> None:
        self._agents: dict[int, AgentInfo] = {}
        self._lock = asyncio.Lock()
        self._expected_count = expected_count
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

    @app.post(messages.SHUTDOWN_PATH)
    async def run_shutdown() -> messages.ShutdownResponse:
        '''Initiate graceful shutdown'''
        ...

    @app.post(messages.EXEC_PATH)
    async def run_cmd() -> messages.ExecResponse:
        '''Run cmd on host'''
        ...

    return app
