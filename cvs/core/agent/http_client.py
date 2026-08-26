'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import messages


@dataclass
class HostOutput:
    host: str
    stdout: list[str]
    stderr: list[str]
    exit_code: int | None
    exception: Exception | None


class ParallelHTTPClientError(Exception):
    '''Raised by run_command when stop_on_errors=True and at least one host failed to reach its agent
    or returned an unparseable response. Mirrors ParallelSSHClient's raise-on-connection-failure behavior;
    a nonzero remote exit_code is not itself a failure here, matching pssh's stop_on_errors semantics.'''


class ParallelHTTPClient:
    '''ParallelSSHClient-API-compatible client that fans a command out to per-host HTTP agents.'''

    def __init__(self, agent_urls: dict[str, str], token: str, connect_timeout: float | None = None) -> None:
        self._agent_urls = agent_urls
        self._token = token
        self._connect_timeout = connect_timeout

    def _auth_header(self) -> dict[str, str]:
        return {messages.AUTH_HEADER: f"{messages.AUTH_SCHEME} {self._token}"}

    def _build_exec_requests(
        self, cmd: str, host_args: list | None, read_timeout: float | None
    ) -> dict[str, messages.ExecRequest]:
        hosts = list(self._agent_urls)
        if host_args is not None:
            if len(host_args) != len(hosts):
                raise ValueError(f"host_args has {len(host_args)} entries but there are {len(hosts)} hosts")
            commands = [cmd % args for args in host_args]
        else:
            commands = [cmd] * len(hosts)
        return {
            host: messages.ExecRequest(
                cmd=command,
                env={},
                cwd=Path.cwd(),
                timeout=read_timeout,
                inactivity_timeout=None,
                cmd_id=uuid.uuid4().hex,
                out_path=None,
                output_mode=messages.ExecOutputMode.INLINE,
            )
            for host, command in zip(hosts, commands)
        }

    async def _run_one(
        self, client: httpx.AsyncClient, host: str, url: str, request: messages.ExecRequest
    ) -> HostOutput:
        try:
            response = await client.post(f"{url}{messages.EXEC_PATH}", content=request.model_dump_json())
            response.raise_for_status()
            exec_response = messages.parse_message(messages.ExecResponse, response.text)
        except Exception as exc:  # noqa: BLE001 - captured per-host so one bad host doesn't sink the others
            return HostOutput(host=host, stdout=[], stderr=[], exit_code=None, exception=exc)
        return HostOutput(
            host=host,
            stdout=exec_response.stdout or [],
            stderr=exec_response.stderr or [],
            exit_code=exec_response.exit_code,
            exception=None,
        )

    async def _run_command_async(
        self, requests: dict[str, messages.ExecRequest], read_timeout: float | None
    ) -> list[HostOutput]:
        timeout = httpx.Timeout(read_timeout, connect=self._connect_timeout)
        async with httpx.AsyncClient(headers=self._auth_header(), timeout=timeout) as client:
            tasks = [self._run_one(client, host, self._agent_urls[host], request) for host, request in requests.items()]
            return await asyncio.gather(*tasks)

    def run_command(
        self,
        cmd: str,
        stop_on_errors: bool = True,
        read_timeout: float | None = None,
        host_args: list | None = None,
    ) -> list[HostOutput]:
        requests = self._build_exec_requests(cmd, host_args, read_timeout)
        outputs = asyncio.run(self._run_command_async(requests, read_timeout))
        if stop_on_errors:
            failed = [output for output in outputs if output.exception is not None]
            if failed:
                details = ", ".join(f"{output.host}: {output.exception}" for output in failed)
                raise ParallelHTTPClientError(f"{len(failed)} host(s) failed: {details}")
        return outputs

    def join(self) -> None:
        '''No-op: kept for API parity with ParallelSSHClient.join(), which waits on SFTP transfers this
        client never starts.'''
