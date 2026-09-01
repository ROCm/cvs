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

# /v1/exec returns only after the process finishes. A timed-out process then spends
# TERMINATE_GRACE_PERIOD_SECONDS in SIGTERM-then-SIGKILL before ExecResponse can be sent;
# the extra buffer covers response serialization and FILE-mode writes on the shared FS.
_EXEC_RESPONSE_BUFFER_SECONDS = 1.0
# Fallback read deadline for requests that carry no execution cost of their own (health, and any
# future endpoint). Matches httpx's implicit default, stated explicitly so it can't drift silently.
_DEFAULT_READ_TIMEOUT_SECONDS = 5.0
# Shutdown waits out the agent's process-group termination grace before returning.
_SHUTDOWN_READ_TIMEOUT_SECONDS = messages.TERMINATE_GRACE_PERIOD_SECONDS + _EXEC_RESPONSE_BUFFER_SECONDS


def _exec_http_read_timeout(read_timeout: float | None) -> float | None:
    '''HTTP read deadline for /v1/exec: the agent's process deadline plus termination grace.'''
    if read_timeout is None:
        return None
    return round(read_timeout) + messages.TERMINATE_GRACE_PERIOD_SECONDS + _EXEC_RESPONSE_BUFFER_SECONDS


def _validated_exec_output_path(reported: Path | None, out_dir: Path, cmd_id: str, stream: str) -> Path:
    '''Accept a FILE-mode path only if it resolves to <out_dir>/<cmd_id>.stdout|stderr.'''
    expected_name = f"{cmd_id}.{stream}"
    if reported is None:
        raise HTTPProtocolError(f"FILE-mode response omitted {stream} path")
    expected_parent = out_dir.resolve()
    resolved = reported.resolve()
    if resolved.parent != expected_parent or resolved.name != expected_name:
        raise HTTPProtocolError(f"FILE-mode {stream} path {reported} is not {expected_parent / expected_name}")
    return resolved


@dataclass
class HostOutput:
    host: str
    stdout: list[str]
    stderr: list[str]
    exit_code: int | None
    exception: Exception | None
    timed_out: bool = False
    truncated: bool | None = None


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

    def _http_timeout(self, read_timeout: float | None) -> httpx.Timeout:
        return httpx.Timeout(read_timeout, connect=self._connect_timeout)

    def _pool_limits(self) -> httpx.Limits:
        # httpx caps the pool at 100 connections by default, which would serialize the tail of a
        # fan-out on a larger cluster (and turn into PoolTimeout once a request deadline is set).
        # Concurrency here is already bounded by the host count - one connection per agent - so the
        # pool needs no bound of its own, and staying unbounded keeps rebuild() to a larger host set
        # from having to tear down a live pool and lose its keep-alives.
        return httpx.Limits(max_connections=None, max_keepalive_connections=None)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self._auth_header(),
                transport=self._transport,
                timeout=self._http_timeout(_DEFAULT_READ_TIMEOUT_SECONDS),
                limits=self._pool_limits(),
            )
        return self._client

    def rebuild(self, agent_urls: dict[str, str]) -> None:
        '''Replace the host map, e.g. to drop hosts pruned after a failed health check. The shared
        client's connection pool needs no action either way: idle connections to removed hosts age
        out, and the pool is unbounded so an added host opens a connection without evicting anyone.'''
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

    async def _collect_output(
        self, request: messages.ExecRequest, exec_response: messages.ExecResponse
    ) -> tuple[list[str], list[str]]:
        '''FILE mode ships only a tail preview inline; the full output lives on the shared FS at
        stdout_path/stderr_path, so read it back here to give callers the same list[str] shape
        regardless of which output_mode produced the response (INLINE/EXIT_CODE_ONLY never set
        stdout_path, so this falls through to the inline fields for those unchanged).'''
        if exec_response.stdout_path is None and exec_response.stderr_path is None:
            return exec_response.stdout or [], exec_response.stderr or []
        if request.out_path is None:
            raise HTTPProtocolError("agent returned FILE-mode paths but no out_path was requested")
        stdout_path = _validated_exec_output_path(exec_response.stdout_path, request.out_path, request.cmd_id, "stdout")
        stderr_path = _validated_exec_output_path(exec_response.stderr_path, request.out_path, request.cmd_id, "stderr")
        try:
            stdout_text, stderr_text = await asyncio.gather(
                asyncio.to_thread(stdout_path.read_text),
                asyncio.to_thread(stderr_path.read_text),
            )
        except OSError as exc:
            raise HTTPProtocolError(f"failed to read FILE-mode output: {exc}") from exc
        return stdout_text.splitlines(), stderr_text.splitlines()

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
                json=request.model_dump(mode="json"),
                timeout=self._http_timeout(_exec_http_read_timeout(read_timeout)),
            )
            response.raise_for_status()
            exec_response = messages.parse_message(messages.ExecResponse, response.text)
            stdout, stderr = await self._collect_output(request, exec_response)
        except Exception as exc:  # noqa: BLE001 - captured per-host so one bad host doesn't sink the others
            return HostOutput(host=host, stdout=[], stderr=[], exit_code=None, exception=_classify_exception(exc))
        return HostOutput(
            host=host,
            stdout=stdout,
            stderr=stderr,
            exit_code=exec_response.exit_code,
            exception=None,
            timed_out=exec_response.timed_out,
            truncated=exec_response.truncated,
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

    async def _fan_out(self, method: str, path: str, stop_on_errors: bool, read_timeout: float) -> dict[str, bool]:
        client = self._get_client()
        timeout = self._http_timeout(read_timeout)

        async def call(host: str, url: str) -> tuple[str, bool | Exception]:
            try:
                response = await client.request(method, f"{url}{path}", timeout=timeout)
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
        return await self._fan_out(
            "GET", messages.HEALTH_PATH, stop_on_errors=False, read_timeout=_DEFAULT_READ_TIMEOUT_SECONDS
        )

    async def shutdown(self, stop_on_errors: bool = False) -> dict[str, bool]:
        '''Ask every host's agent to terminate its spawned processes and exit. Defaults to best-effort
        (stop_on_errors=False), unlike run_command: one already-dead straggler during cleanup shouldn't
        stop the rest from being told to shut down.'''
        return await self._fan_out(
            "POST", messages.SHUTDOWN_PATH, stop_on_errors, read_timeout=_SHUTDOWN_READ_TIMEOUT_SECONDS
        )
