"""Node-local Unix-socket fan-out for managed launches."""

import asyncio
import json
import os
import socket
import time
from pathlib import Path

from cvs.lib import globals

from . import messages
from .launch import ActiveLaunches, run_launch_child

log = globals.log

LOCAL_WORKER_STARTUP_TIMEOUT_SECONDS = 60
# How long a launch waits for every local rank to report that its child exists. MPI ranks then
# rendezvous in MPI_Init on their own; this only catches a slot that never started a child.
LAUNCH_SPAWN_TIMEOUT_SECONDS = 30


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
        self.spawned = asyncio.Event()


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
        self._had_full_worker_set = expected_workers == 0
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
        connection = None
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
                self._had_full_worker_set = True
            while raw := await reader.readline():
                if json.loads(raw).get("kind") == "spawned":
                    connection.spawned.set()
                    continue
                await connection.responses.put(messages.parse_message(messages.LaunchRankResult, raw.decode()))
        except (ConnectionResetError, BrokenPipeError, ConnectionError, asyncio.TimeoutError) as exc:
            rank = connection.rank if connection is not None else None
            log.debug("UDS worker disconnected (rank=%s): %s", rank, exc)
        except Exception:
            rank = connection.rank if connection is not None else None
            log.exception("UDS worker connection failed (rank=%s)", rank)
        finally:
            if connection is not None:
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
            worker.spawned.clear()
            try:
                worker.writer.write(request.model_dump_json().encode() + b"\n")
                await worker.writer.drain()
                result = await worker.responses.get()
                if result is None:
                    raise ConnectionError("local launch worker disconnected")
                return result
            finally:
                worker.in_launch = False

    async def _wait_for_spawns(self, own_spawned, connections, running):
        """Raise once it is clear a local rank never started its child.

        running finishing first means every child already ran, so no ack is owed.
        """
        acked = asyncio.gather(own_spawned.wait(), *(worker.spawned.wait() for worker in connections))
        try:
            await asyncio.wait(
                {acked, running},
                timeout=LAUNCH_SPAWN_TIMEOUT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if acked.done() or running.done():
                return
            missing = [] if own_spawned.is_set() else [self.global_rank]
            missing += [worker.rank for worker in connections if not worker.spawned.is_set()]
            raise TimeoutError(
                f"local ranks {sorted(missing)} did not start a child within {LAUNCH_SPAWN_TIMEOUT_SECONDS}s"
            )
        finally:
            acked.cancel()
            await asyncio.gather(acked, return_exceptions=True)

    def _require_local_workers(self):
        """Fail immediately when a worker that had registered is no longer connected."""
        have = len(self._workers)
        if have != self.expected_workers:
            raise RuntimeError(f"local UDS workers {have}/{self.expected_workers}")

    async def launch(self, request):
        """Run request.argv as this rank's child and on every connected UDS worker."""
        if self._had_full_worker_set:
            self._require_local_workers()
        else:
            await self.wait_until_ready()
        own_spawned = asyncio.Event()
        connections = list(self._workers.values())
        running = asyncio.gather(
            run_launch_child(request, self.global_rank, self.active, on_spawn=own_spawned.set),
            *(self._launch_on_worker(worker, request) for worker in connections),
            return_exceptions=True,
        )
        try:
            await self._wait_for_spawns(own_spawned, connections, running)
        except TimeoutError:
            # A rank that never spawned may also never answer, so drop the waiters after cancelling.
            await self.cancel()
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            raise
        results = []
        for index, item in enumerate(await running):
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

    def _send_spawned(self):
        """Tell the coordinator this rank's child exists; the result line follows when it exits."""
        self._writer.write(json.dumps({"kind": "spawned", "rank": self.global_rank}).encode() + b"\n")

    async def _run_one_launch(self, request):
        """Run request.argv; return True if the coordinator asked to shut down."""
        launch_task = asyncio.create_task(
            run_launch_child(request, self.global_rank, self._active, on_spawn=self._send_spawned)
        )
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
