"""Node-local Unix-socket fan-out for managed launches."""

import asyncio
import json
import os
import socket
import time
from pathlib import Path

from . import messages
from .launch import ActiveLaunches, run_launch_child

LOCAL_WORKER_STARTUP_TIMEOUT_SECONDS = 60


def launch_socket_path():
    """Return a node-local path shared by this job's tasks."""
    job_id = os.environ.get("SPUR_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "nojob"
    return Path(f"/tmp/cvs-agent.{os.getuid()}.{job_id}") / "launch.sock"


class _WorkerConnection:
    """One UDS worker: its rank, writer, and result queue."""

    def __init__(self, rank, writer, lock, responses, closed):
        """Store handles for one connected extra local rank."""
        self.rank = rank
        self.writer = writer
        self.lock = lock
        self.responses = responses
        self.closed = closed
        self.in_launch = False


class UdsLaunchCoordinator:
    """Bind launch.sock. On launch(), run request.argv as this rank's child and on every UDS worker on the node."""

    def __init__(self, global_rank, expected_workers, socket_path=None):
        """expected_workers is how many UDS workers must register (tasks_per_node - 1).

        socket_path is optional; production uses launch_socket_path(). Tests may pass a temp path.
        """
        self.global_rank = global_rank
        self.expected_workers = expected_workers
        self.socket_path = None if expected_workers == 0 else Path(socket_path) if socket_path else launch_socket_path()
        self._workers = {}
        self._ready = asyncio.Event()
        self._server = None
        self.active = ActiveLaunches()
        if expected_workers == 0:
            self._ready.set()

    async def start(self):
        """Bind launch.sock, or no-op when this node has no UDS workers."""
        if self.expected_workers == 0:
            return
        if self.socket_path is None:
            raise RuntimeError("UDS workers require a node-local socket path")
        await asyncio.to_thread(self.socket_path.parent.mkdir, parents=True, mode=0o700, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(self._accept_worker, path=self.socket_path)
        os.chmod(self.socket_path, 0o600)

    async def stop(self):
        """Cancel in-flight children, tell UDS workers to exit, unbind and unlink the socket."""
        await self.cancel()
        workers = list(self._workers.values())
        for worker in workers:
            worker.writer.write(b'{"kind":"shutdown"}\n')
        await asyncio.gather(
            *(worker.writer.drain() for worker in workers),
            return_exceptions=True,
        )
        for worker in workers:
            try:
                await asyncio.wait_for(worker.closed.wait(), timeout=1)
            except asyncio.TimeoutError:
                worker.writer.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self.socket_path is not None:
            self.socket_path.unlink(missing_ok=True)

    async def cancel(self):
        """Terminate this node's active children, including UDS worker children."""
        await self.active.terminate_all()
        workers = [worker for worker in self._workers.values() if worker.in_launch]
        for worker in workers:
            worker.writer.write(b'{"kind":"cancel"}\n')
        await asyncio.gather(
            *(worker.writer.drain() for worker in workers),
            return_exceptions=True,
        )

    async def wait_until_ready(self, timeout=LOCAL_WORKER_STARTUP_TIMEOUT_SECONDS):
        """Block until every expected UDS worker has registered."""
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)

    async def _accept_worker(self, reader, writer):
        """Handle one UDS worker: register, queue LaunchRankResult lines, drop on disconnect."""
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=LOCAL_WORKER_STARTUP_TIMEOUT_SECONDS)
            registration = json.loads(raw)
            if registration.get("kind") != "register":
                raise ValueError("first UDS message must be worker registration")
            rank = int(registration["rank"])
            connection = _WorkerConnection(rank, writer, asyncio.Lock(), asyncio.Queue(), asyncio.Event())
            old = self._workers.get(rank)
            if old is not None:
                old.closed.set()
                old.writer.close()
            self._workers[rank] = connection
            if len(self._workers) == self.expected_workers:
                self._ready.set()
            while raw := await reader.readline():
                await connection.responses.put(messages.parse_message(messages.LaunchRankResult, raw.decode()))
        except Exception:
            pass
        finally:
            if "connection" in locals():
                connection.closed.set()
                await connection.responses.put(None)
                if self._workers.get(connection.rank) is connection:
                    self._workers.pop(connection.rank, None)
                    self._ready.clear()
            writer.close()

    async def _launch_on_worker(self, worker, request):
        """Send LaunchRequest on one connection and wait for that worker's result."""
        async with worker.lock:
            worker.in_launch = True
            try:
                worker.writer.write(request.model_dump_json().encode() + b"\n")
                await worker.writer.drain()
                result = await worker.responses.get()
                if result is None:
                    raise ConnectionError("local launch worker disconnected")
                return result
            finally:
                worker.in_launch = False

    async def launch(self, request):
        """Run request.argv as this rank's child and on every connected UDS worker."""
        await self.wait_until_ready()
        own = run_launch_child(request, self.global_rank, self.active)
        connections = list(self._workers.values())
        workers = [self._launch_on_worker(worker, request) for worker in connections]
        gathered = await asyncio.gather(own, *workers, return_exceptions=True)
        results = []
        for index, item in enumerate(gathered):
            if isinstance(item, messages.LaunchRankResult):
                results.append(item)
            else:
                failed_rank = self.global_rank if index == 0 else connections[index - 1].rank
                results.append(
                    messages.LaunchRankResult(
                        rank=failed_rank,
                        hostname=socket.gethostname(),
                        exit_code=None,
                        stdout_path=request.out_path / "unknown.stdout",
                        stderr_path=request.out_path / "unknown.stderr",
                        timed_out=False,
                        error=str(item),
                    )
                )
        return messages.LaunchResponse(results=sorted(results, key=lambda result: result.rank))


class UdsWorker:
    """UDS worker: connect to launch.sock, run request.argv, stay up for the next launch."""

    def __init__(self, global_rank, socket_path=None):
        """socket_path is optional; production uses launch_socket_path(). Tests may pass a temp path."""
        self.global_rank = global_rank
        self.socket_path = Path(socket_path) if socket_path else launch_socket_path()
        self._reader = None
        self._writer = None
        self._active = ActiveLaunches()

    async def start(self):
        """Connect, register, and serve launch requests until the coordinator closes."""
        await self._connect()
        try:
            await self._serve()
            return 0
        finally:
            await self.stop()

    async def _connect(self):
        """Retry connect until launch.sock exists, then register this rank."""
        deadline = time.monotonic() + LOCAL_WORKER_STARTUP_TIMEOUT_SECONDS
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"local coordinator socket did not appear: {self.socket_path}")
                await asyncio.sleep(0.1)
        self._writer.write(json.dumps({"kind": "register", "rank": self.global_rank}).encode() + b"\n")
        await self._writer.drain()

    async def _serve(self):
        """Read launch/cancel/shutdown until the coordinator closes the socket."""
        while raw := await self._reader.readline():
            message = json.loads(raw)
            if message.get("kind") == "shutdown":
                break
            request = messages.parse_message(messages.LaunchRequest, raw.decode())
            if await self._run_one_launch(request):
                break

    async def _run_one_launch(self, request):
        """Run request.argv; return True if the coordinator asked to shut down."""
        launch_task = asyncio.create_task(run_launch_child(request, self.global_rank, self._active))
        shutdown = False
        while not launch_task.done():
            control_task = asyncio.create_task(self._reader.readline())
            done, _ = await asyncio.wait((launch_task, control_task), return_when=asyncio.FIRST_COMPLETED)
            if launch_task in done:
                control_task.cancel()
                await asyncio.gather(control_task, return_exceptions=True)
                break
            control_raw = control_task.result()
            if not control_raw:
                await self.cancel()
                shutdown = True
                break
            control = json.loads(control_raw)
            if control.get("kind") in ("cancel", "shutdown"):
                await self.cancel()
                shutdown = control.get("kind") == "shutdown"
            else:
                await self.cancel()
                raise ValueError("unexpected UDS message during launch")
        result = await launch_task
        self._writer.write(result.model_dump_json().encode() + b"\n")
        await self._writer.drain()
        return shutdown

    async def cancel(self):
        """Terminate this rank's in-flight child."""
        await self._active.terminate_all()

    async def stop(self):
        """Cancel any child and close the UDS connection."""
        await self.cancel()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
