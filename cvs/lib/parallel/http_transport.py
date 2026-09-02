'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import asyncio
import shutil
import threading
import time
from pathlib import Path

from pssh.exceptions import Timeout

from cvs.core.agent.http_client import HTTPConnectionError, ParallelHTTPClient
from cvs.lib import globals
from cvs.lib.parallel.transport import BaseTransport

log = globals.log

_HEALTH_POLL_SECONDS = 0.1
_LOOP_START_TIMEOUT_SECONDS = 5


class _Completed:
    """Stand-in for a parallel-ssh copy greenlet. ``get()`` is already finished."""

    def __init__(self, error=None):
        self._error = error

    def get(self):
        if self._error is not None:
            raise self._error


class _NoopPool:
    def join(self):
        return None


class _LoopThread:
    """A dedicated asyncio loop so ParallelHandle can call the async HTTP client synchronously."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="http-transport-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=_LOOP_START_TIMEOUT_SECONDS):
            raise RuntimeError("HTTP transport event loop failed to start")

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()
        self._loop.close()

    def run(self, coro):
        if not self._thread.is_alive() or not self._loop.is_running():
            raise RuntimeError("HTTP transport event loop is stopped")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


_loop_lock = threading.Lock()
_loop = None


def _shared_loop():
    """One daemon loop thread per process, shared by every HttpTransport.

    Transports are created per handle and per host subset (client_for_hosts,
    orchestrator subset execs), so a loop per transport would leak a thread on every
    one that outlives its destroy(). The loop holds no per-transport state -- each
    transport owns its own httpx pool -- so sharing costs nothing and the daemon
    thread goes away with the process.
    """
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = _LoopThread()
        return _loop


def _copy_path(src, dst, recurse):
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.resolve() == dst_path.resolve() and dst_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if recurse and src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        return
    shutil.copy2(src_path, dst_path)


def _interpolate(template, args):
    if args is None:
        return template
    try:
        return template % args
    except (TypeError, ValueError, KeyError):
        return template


def _annotate_agent_outcomes(outputs):
    """Make agent-side timeout and truncation visible to ParallelHandle._process_output.

    The agent reports both out-of-band on HostOutput, but _process_output only reads
    stdout/stderr/exception/exit_code. Left alone, a killed-on-timeout command reads as
    an ordinary nonzero exit and a truncated capture reads as complete output. SSH
    surfaces a read timeout as pssh's Timeout on the result, so raise the same exception
    type here: _process_output already appends its "ABORT: Timeout Error" marker and
    forces exit_code -1 for it.
    """
    for output in outputs:
        if getattr(output, 'timed_out', False) and output.exception is None:
            output.exception = Timeout(f"agent terminated the command after its timeout on {output.host}")
        if getattr(output, 'truncated', False):
            output.stderr = list(output.stderr) + [f"ABORT: Output Truncated by agent on Host: {output.host}"]
    return outputs


class _SyncHTTPClient:
    """ParallelSSHClient-shaped facade over ParallelHTTPClient plus shared-FS copy ops."""

    def __init__(self, runner, http_client, hosts):
        self._runner = runner
        self._http = http_client
        self._hosts = list(hosts)
        self.pool = _NoopPool()

    def run_command(
        self,
        cmd,
        stop_on_errors=True,
        read_timeout=None,
        host_args=None,
        inactivity_timeout=None,
    ):
        return _annotate_agent_outcomes(
            self._runner(
                self._http.run_command(
                    cmd,
                    stop_on_errors=stop_on_errors,
                    read_timeout=read_timeout,
                    host_args=host_args,
                    inactivity_timeout=inactivity_timeout,
                )
            )
        )

    def health(self):
        return self._runner(self._http.health())

    def shutdown(self, stop_on_errors=False):
        return self._runner(self._http.shutdown(stop_on_errors=stop_on_errors))

    def rebuild(self, agent_urls):
        self._http.rebuild(agent_urls)
        self._hosts = list(agent_urls)

    def destroy(self):
        self._runner(self._http.destroy())

    def _copy_one(self, src, dst, recurse):
        try:
            _copy_path(src, dst, recurse)
        except Exception as exc:  # noqa: BLE001 - surfaced via greenlet-style get()
            return _Completed(exc)
        return _Completed()

    def copy_file(self, local_file, remote_file, recurse=False, copy_args=None):
        if copy_args is not None:
            return [
                self._copy_one(
                    _interpolate(local_file, args),
                    _interpolate(remote_file, args),
                    recurse,
                )
                for args in copy_args
            ]
        # One shared-FS write serves every host; get() is idempotent, so callers zipping
        # the list against the host list can share the single outcome object.
        completed = self._copy_one(local_file, remote_file, recurse)
        return [completed] * len(self._hosts)

    def copy_remote_file(self, remote_file, local_file, recurse=False, suffix_separator='_'):
        return [self._copy_one(remote_file, f'{local_file}{suffix_separator}{host}', recurse) for host in self._hosts]


class HttpTransport(BaseTransport):
    """HTTP wire protocol via ParallelHTTPClient, with a sync client for ParallelHandle."""

    # An agent that answers at all is reachable, so only a transport-level failure is a
    # pruning candidate. Notably a command timeout is not: the agent reported it.
    prune_exception_types = (HTTPConnectionError,)

    # /v1/exec returns finished output, so the agent applies the inactivity timeout.
    remote_inactivity_timeout = True

    def __init__(self, hosts, *, agent_urls, token, connect_timeout=None, **client_kwargs):
        if not token:
            raise ValueError("HTTP transport requires a non-empty token")
        if not agent_urls:
            raise ValueError("HTTP transport requires agent_urls")
        self._all_urls = dict(agent_urls)
        self._token = token
        self._connect_timeout = connect_timeout
        self._client_kwargs = client_kwargs
        self.hosts = list(hosts)
        self._loop = _shared_loop()
        self.client = self._make_client(self.hosts)

    def _urls_for(self, hosts):
        missing = [host for host in hosts if host not in self._all_urls]
        if missing:
            raise ValueError(
                f"No agent URL for host(s): {missing}. Registered agents: {sorted(self._all_urls)}. "
                f"Cluster-file host names must match the names the agents registered under."
            )
        return {host: self._all_urls[host] for host in hosts}

    def _new_http_client(self, hosts):
        return ParallelHTTPClient(
            self._urls_for(hosts),
            self._token,
            connect_timeout=self._connect_timeout,
            **self._client_kwargs,
        )

    def _make_client(self, hosts):
        return _SyncHTTPClient(self._loop.run, self._new_http_client(hosts), hosts)

    def rebuild(self, hosts):
        self.hosts = list(hosts)
        client = getattr(self, 'client', None)
        if client is None:
            # Recreate after a destroy(), matching SshTransport.rebuild, which builds a
            # fresh client unconditionally. The pool is per-transport, so this is cheap.
            self.client = self._make_client(self.hosts)
            return
        client.rebuild(self._urls_for(self.hosts))

    def check_connectivity(self, hosts):
        if not hosts:
            return []
        # A host with no URL at all can never be reached; probe only the rest. Scoping the
        # probe mirrors SshTransport's throwaway short-timeout client and keeps one flaky
        # node from triggering a cluster-wide health fan-out on every exec.
        unknown = [host for host in hosts if host not in self._all_urls]
        known = [host for host in hosts if host in self._all_urls]
        if not known:
            return unknown
        probe = self._new_http_client(known)
        try:
            results = self._loop.run(probe.health())
        finally:
            self._loop.run(probe.destroy())
        return unknown + [host for host in known if not results.get(host)]

    def client_for_hosts(self, hosts):
        return HttpTransport(
            hosts,
            agent_urls=self._all_urls,
            token=self._token,
            connect_timeout=self._connect_timeout,
            **self._client_kwargs,
        )

    def destroy(self):
        client = getattr(self, 'client', None)
        if client is None:
            return
        try:
            client.destroy()
        except Exception as exc:
            log.debug("Error closing HTTP client: %s", exc)
        del self.client

    def wait_until_healthy(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            results = self.client.health()
            pending = [host for host in self.hosts if not results.get(host)]
            if not pending:
                return results
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"HTTP agents not healthy: {pending}")
            time.sleep(min(_HEALTH_POLL_SECONDS, remaining))

    def shutdown_agents(self, stop_on_errors=False):
        return self.client.shutdown(stop_on_errors=stop_on_errors)
