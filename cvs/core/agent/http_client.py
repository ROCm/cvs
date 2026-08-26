'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import asyncio
import uuid
from dataclasses import dataclass

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
    '''Raised when stop_on_errors=True and at least one host failed to reach its agent or returned an
    unparseable response. Mirrors ParallelSSHClient's raise-on-connection-failure behavior; a nonzero
    remote exit_code is not itself a failure here, matching pssh's stop_on_errors semantics.'''


class HTTPConnectionError(Exception):
    '''Host was unreachable at the transport level: DNS failure, connection refused, TLS failure, or a
    connect/read/write/pool timeout. Wraps httpx.TransportError and its subclasses. Analogous to
    pssh.exceptions.ConnectionError/Timeout/SessionError in cvs/lib/parallel/pssh.py's
    prune_unreachable_hosts - a signal the host itself may be down, and a pruning candidate.'''


class HTTPProtocolError(Exception):
    '''Host was reached but the request failed at the HTTP/application layer: a non-2xx response
    (bad auth, exec-already-in-progress conflict, agent-side error) or an unparseable response body.
    Wraps httpx.HTTPStatusError and messages.MessageParseError. The host is alive, so this may
    succeed on retry - not a pruning candidate, analogous to pssh's non-pruned auth/protocol errors.'''


def _classify_exception(exc: Exception) -> Exception:
    '''Wrap a raw httpx/messages exception so callers can distinguish "host unreachable" from "host
    reached but request failed" via isinstance(exception, HTTPConnectionError), the same split
    cvs/lib/parallel/pssh.py's prune_unreachable_hosts draws for SSH via pssh.exceptions.'''
    if isinstance(exc, httpx.TransportError):
        wrapped: Exception = HTTPConnectionError(str(exc))
    elif isinstance(exc, (httpx.HTTPStatusError, messages.MessageParseError)):
        wrapped = HTTPProtocolError(str(exc))
    else:
        return exc
    wrapped.__cause__ = exc
    return wrapped


class ParallelHTTPClient:
    '''Async, ParallelSSHClient-inspired client that fans a command out to per-host HTTP agents.

    Holds one lazily-created, long-lived httpx.AsyncClient shared across calls so repeated commands
    reuse pooled connections instead of paying a new TCP/TLS handshake each time. Callers own the event
    loop for the lifetime of the client (there is no internal asyncio.run()); call destroy() or use
    `async with` when done to release pooled connections.'''

    def __init__(
        self,
        agent_urls: dict[str, str],
        token: str,
        connect_timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._agent_urls = dict(agent_urls)
        self._token = token
        self._connect_timeout = connect_timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ParallelHTTPClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.destroy()

    def _auth_header(self) -> dict[str, str]:
        return {messages.AUTH_HEADER: f"{messages.AUTH_SCHEME} {self._token}"}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self._auth_header(), transport=self._transport)
        return self._client

    def rebuild(self, agent_urls: dict[str, str]) -> None:
        '''Replace the host map, e.g. to drop hosts pruned after a failed health check. The shared
        client's connection pool needs no action: idle connections to removed hosts simply age out.'''
        self._agent_urls = dict(agent_urls)

    async def destroy(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_exec_requests(
        self,
        cmd: str,
        host_args: list[str] | None,
        read_timeout: float | None,
        env: dict[str, str] | None,
        inactivity_timeout: float | None,
        output_mode: messages.ExecOutputMode,
    ) -> dict[str, messages.ExecRequest]:
        # Imported here, not at module level: cvs.core.run_layout pulls in cvs/core/__init__.py's
        # orchestrator factory, which reaches back into cvs/core/agent/ in ways that risk a cycle
        # (same reasoning as cvs/lib/utils_lib.py's lazy import of RunLayout).
        from cvs.core.run_layout import RunLayout

        hosts = list(self._agent_urls)
        if host_args is not None:
            if len(host_args) != len(hosts):
                raise ValueError(f"host_args has {len(host_args)} entries but there are {len(hosts)} hosts")
            commands = [cmd % args for args in host_args]
        else:
            commands = [cmd] * len(hosts)
        run_dir = RunLayout.get().run_dir
        out_dir = None
        if output_mode == messages.ExecOutputMode.FILE:
            out_dir = run_dir / "exec_output"
            out_dir.mkdir(parents=True, exist_ok=True)
        return {
            host: messages.ExecRequest(
                cmd=command,
                env=env or {},
                cwd=run_dir,
                timeout=round(read_timeout) if read_timeout is not None else None,
                inactivity_timeout=round(inactivity_timeout) if inactivity_timeout is not None else None,
                cmd_id=uuid.uuid4().hex,
                out_path=out_dir,
                output_mode=output_mode,
            )
            for host, command in zip(hosts, commands)
        }

    async def _collect_output(self, exec_response: messages.ExecResponse) -> tuple[list[str], list[str]]:
        '''FILE mode ships only a tail preview inline; the full output lives on the shared FS at
        stdout_path/stderr_path, so read it back here to give callers the same list[str] shape
        regardless of which output_mode produced the response (INLINE/EXIT_CODE_ONLY never set
        stdout_path, so this falls through to the inline fields for those unchanged).'''
        if exec_response.stdout_path is not None and exec_response.stderr_path is not None:
            stdout_text, stderr_text = await asyncio.gather(
                asyncio.to_thread(exec_response.stdout_path.read_text),
                asyncio.to_thread(exec_response.stderr_path.read_text),
            )
            return stdout_text.splitlines(), stderr_text.splitlines()
        return exec_response.stdout or [], exec_response.stderr or []

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        host: str,
        url: str,
        request: messages.ExecRequest,
        read_timeout: float | None,
    ) -> HostOutput:
        try:
            response = await client.post(
                f"{url}{messages.EXEC_PATH}",
                content=request.model_dump_json(),
                timeout=httpx.Timeout(read_timeout, connect=self._connect_timeout),
            )
            response.raise_for_status()
            exec_response = messages.parse_message(messages.ExecResponse, response.text)
            stdout, stderr = await self._collect_output(exec_response)
        except Exception as exc:  # noqa: BLE001 - captured per-host so one bad host doesn't sink the others
            return HostOutput(host=host, stdout=[], stderr=[], exit_code=None, exception=_classify_exception(exc))
        return HostOutput(
            host=host,
            stdout=stdout,
            stderr=stderr,
            exit_code=exec_response.exit_code,
            exception=None,
        )

    async def run_command(
        self,
        cmd: str,
        stop_on_errors: bool = True,
        read_timeout: float | None = None,
        host_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        inactivity_timeout: float | None = None,
        output_mode: messages.ExecOutputMode = messages.ExecOutputMode.INLINE,
    ) -> list[HostOutput]:
        requests = self._build_exec_requests(cmd, host_args, read_timeout, env, inactivity_timeout, output_mode)
        client = self._get_client()
        outputs = await asyncio.gather(
            *(
                self._run_one(client, host, self._agent_urls[host], request, read_timeout)
                for host, request in requests.items()
            )
        )
        if stop_on_errors:
            failed = [output for output in outputs if output.exception is not None]
            if failed:
                details = ", ".join(f"{output.host}: {output.exception}" for output in failed)
                raise ParallelHTTPClientError(f"{len(failed)} host(s) failed: {details}")
        return outputs

    async def _fan_out(self, method: str, path: str, stop_on_errors: bool) -> dict[str, bool]:
        client = self._get_client()

        async def call(host: str, url: str) -> tuple[str, bool | Exception]:
            try:
                response = await client.request(method, f"{url}{path}")
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001 - captured per-host, reported rather than raised
                return host, _classify_exception(exc)
            return host, True

        results = await asyncio.gather(*(call(host, url) for host, url in self._agent_urls.items()))
        failed = {host: outcome for host, outcome in results if isinstance(outcome, Exception)}
        if stop_on_errors and failed:
            details = ", ".join(f"{host}: {exc}" for host, exc in failed.items())
            raise ParallelHTTPClientError(f"{len(failed)} host(s) failed: {details}")
        return {host: outcome is True for host, outcome in results}

    async def health(self) -> dict[str, bool]:
        '''Liveness probe per host; never raises regardless of failures - an unreachable host is the
        answer this call exists to produce (feeds rebuild()'s pruning decision), not an error.'''
        return await self._fan_out("GET", messages.HEALTH_PATH, stop_on_errors=False)

    async def shutdown(self, stop_on_errors: bool = False) -> dict[str, bool]:
        '''Ask every host's agent to terminate its spawned processes and exit. Defaults to best-effort
        (stop_on_errors=False), unlike run_command: one already-dead straggler during cleanup shouldn't
        stop the rest from being told to shut down.'''
        return await self._fan_out("POST", messages.SHUTDOWN_PATH, stop_on_errors)
