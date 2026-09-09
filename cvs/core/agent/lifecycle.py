'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import atexit
import asyncio
import json
import os
import secrets
import socket
import tempfile
import threading
import time

import httpx
import uvicorn

from cvs.core.scheduler import scheduler_hosts, scheduler_rank

from . import messages
from .http_agent import create_app

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


def _write_file(path, value, mode=0o644):
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


def _read_file(path):
    return path.read_text(encoding="utf-8").strip()


class ClusterFile:
    """Job-owned cluster JSON: scheduler hosts plus optional --cluster_file metadata."""

    def __init__(self, hosts, layout, cluster=None):
        """Store scheduler hosts, run layout, and optional operator cluster input."""
        self.hosts = hosts
        self.layout = layout
        self.cluster_input = cluster or {}
        self._cluster = None

    def build(self):
        """Build the job-owned cluster config, optionally preserving operator metadata."""
        cluster_input = self.cluster_input
        if not isinstance(cluster_input, dict):
            raise ValueError("managed cluster input must be a JSON object")
        input_nodes = cluster_input.get("node_dict", {})
        if not isinstance(input_nodes, dict):
            raise ValueError("managed cluster node_dict must be an object")
        unknown = sorted(set(input_nodes) - set(self.hosts))
        if unknown:
            raise ValueError(f"cluster file contains nodes outside this scheduler job: {unknown}")
        invalid = [host for host, metadata in input_nodes.items() if not isinstance(metadata, dict)]
        if invalid:
            raise ValueError(f"cluster node metadata must be objects: {invalid}")

        cluster = {
            "orchestrator": "baremetal",
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
        }
        cluster.update(
            {key: value for key, value in cluster_input.items() if key not in ("node_dict", "head_node_dict")}
        )
        cluster["node_dict"] = {host: {"vpc_ip": host, **input_nodes.get(host, {})} for host in self.hosts}
        cluster["head_node_dict"] = {"mgmt_ip": self.hosts[0]}
        self._cluster = cluster
        return cluster

    def create(self, snapshot):
        """Build the cluster config, annotate agent ports, and write cluster_agents.json."""
        cluster = self.build()
        by_host = {}
        for info in snapshot.values():
            if info.hostname in by_host:
                raise ValueError(f"duplicate agent hostname: {info.hostname}")
            by_host[info.hostname] = info
        missing = [host for host in self.hosts if host not in by_host]
        if missing:
            raise ValueError(f"agents did not register for scheduler hosts: {missing}")

        output = dict(cluster)
        output["node_dict"] = {
            host: {**cluster["node_dict"][host], "agent_port": by_host[host].port} for host in self.hosts
        }
        output["agent_token_file"] = str(self.layout.agent_dir / messages.AUTH_TOKEN_FILENAME)
        destination = self.layout.run_dir / "cluster_agents.json"
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(output, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return str(destination)


class Heartbeat:
    """Touch rank0.alive on an interval so workers can tell rank 0 is still running."""

    def __init__(self, agent_dir, interval):
        """Bind the alive file directory and pulse interval."""
        self._agent_dir = agent_dir
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        """Write the first pulse, then keep pulsing on a background thread."""
        self.alive()
        self._thread.start()

    def stop(self):
        """Stop the background thread and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 1)

    def alive(self):
        """Update rank0.alive's mtime (the worker liveness signal)."""
        (self._agent_dir / RANK0_ALIVE_FILENAME).touch()

    def _run(self):
        """Loop until stop(): wait one interval, then call alive()."""
        while not self._stop_event.wait(self._interval):
            self.alive()


class HttpAgentServer:
    """Run the HTTP agent application in the background for one scheduler rank."""

    def __init__(self, agent_dir, rank, world_size, host=None):
        """Bind a port and build the FastAPI app for this rank."""
        self.host = (
            host or os.environ.get("SLURM_NODENAME") or os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        )
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

    def _run(self):
        """Thread entry: run the asyncio uvicorn server."""
        asyncio.run(self._serve())

    async def _serve(self):
        """Serve on the pre-bound socket and capture the event loop."""
        self._event_loop = asyncio.get_running_loop()
        await self._server.serve(sockets=[self._socket])

    def start(self):
        """Start uvicorn on a background thread."""
        self._thread.start()

    def wait_until_ready(self, timeout):
        """Block until the server is accepting connections; return the bound port."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started and self._event_loop is not None:
                return self.port
            if not self._thread.is_alive():
                raise RuntimeError("HTTP agent stopped before becoming ready")
            time.sleep(0.01)
        raise TimeoutError("HTTP agent did not become ready")

    def wait_for_registrations(self, timeout):
        """Block until every rank has registered (rank 0's in-process registry)."""
        future = asyncio.run_coroutine_threadsafe(self._app.state.registry.wait_until_ready(timeout), self._event_loop)
        return future.result(timeout=timeout + 1)

    def registered_agents(self):
        """Return the current rank → hostname/port map, even if incomplete."""
        return self._app.state.registry.snapshot()

    def stop(self):
        """Ask uvicorn to exit and join the server thread."""
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        else:
            self._socket.close()


class Rank0Runner:
    """Rank 0: HTTP agent, rendezvous files, heartbeat, and cluster_agents.json."""

    def __init__(self, layout, host, hosts, cluster=None):
        """Store this job's layout, this node's host, and the scheduler host list."""
        self._layout = layout
        self._host = host
        self._hosts = hosts
        self._world_size = len(hosts)
        self._cluster_input = cluster or {}
        self._http_agent = None
        self._heartbeat = None
        self._stopped = False

    def start(self):
        """Clear leftover markers, mint the token, serve HTTP, and publish rendezvous."""
        agent_dir = self._layout.agent_dir
        self.clear()
        self.create_auth_token()
        self._http_agent = HttpAgentServer(agent_dir, 0, self._world_size, host=self._host)
        try:
            self._http_agent.start()
            port = self._http_agent.wait_until_ready(STARTUP_TIMEOUT_SECONDS)
            self.publish(self._http_agent.host, port, HEARTBEAT_INTERVAL_SECONDS)
            self._heartbeat = Heartbeat(agent_dir, HEARTBEAT_INTERVAL_SECONDS)
            self._heartbeat.start()
            atexit.register(self.stop)
        except Exception:
            self._mark_done()
            self._http_agent.stop()
            raise

    def wait(self):
        """Wait for worker registrations and write cluster_agents.json; return its path."""
        try:
            registered = self._http_agent.wait_for_registrations(REGISTRATION_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.TimeoutError):
            registered = self._http_agent.registered_agents()
        cluster_file = ClusterFile(self._hosts, self._layout, self._cluster_input)
        return cluster_file.create(registered)

    def stop(self):
        """Stop the heartbeat, write rank0.done, and stop the HTTP server."""
        if self._stopped:
            return
        self._stopped = True
        if self._heartbeat is not None:
            self._heartbeat.stop()
        self._mark_done()
        if self._http_agent is not None:
            self._http_agent.stop()

    def clear(self):
        """Remove leftover rank-0 marker files from a previous run."""
        agent_dir = self._layout.agent_dir
        for filename in (
            messages.AUTH_TOKEN_FILENAME,
            RANK0_HOST_FILENAME,
            RANK0_PORT_FILENAME,
            RANK0_ALIVE_FILENAME,
            RANK0_HEARTBEAT_INTERVAL_FILENAME,
            RANK0_DONE_FILENAME,
        ):
            (agent_dir / filename).unlink(missing_ok=True)

    def create_auth_token(self):
        """Write a new bearer token for workers and the orchestrator."""
        _write_file(
            self._layout.agent_dir / messages.AUTH_TOKEN_FILENAME,
            f"{secrets.token_hex(32)}\n",
            mode=0o600,
        )

    def publish(self, host, port, heartbeat_interval):
        """Write rank0.host, rank0.port, and the heartbeat interval for workers."""
        agent_dir = self._layout.agent_dir
        _write_file(agent_dir / RANK0_HOST_FILENAME, f"{host}\n")
        _write_file(agent_dir / RANK0_HEARTBEAT_INTERVAL_FILENAME, f"{heartbeat_interval}\n")
        _write_file(agent_dir / RANK0_PORT_FILENAME, f"{port}\n")

    def _mark_done(self):
        """Write rank0.done so workers exit the watch loop."""
        _write_file(self._layout.agent_dir / RANK0_DONE_FILENAME, "\n")


class WorkerRunner:
    """Non-zero rank: find rank 0, serve HTTP, register, and watch rank 0's heartbeat."""

    def __init__(self, layout, rank, world_size, host):
        """Store this rank's layout, scheduler rank, world size, and hostname."""
        self._layout = layout
        self._rank = rank
        self._world_size = world_size
        self._host = host
        self._http_agent = None

    def start(self):
        """Rendezvous, start this rank's HTTP agent, register, then watch rank 0.

        Returns 0 if rank 0 finished, 1 if the heartbeat went stale.
        """
        agent_dir = self._layout.agent_dir
        rank0_host, rank0_port, heartbeat_interval, token = self._rendezvous(BOOTSTRAP_TIMEOUT_SECONDS)
        self._http_agent = HttpAgentServer(
            agent_dir,
            self._rank,
            self._world_size,
            host=self._host,
        )
        try:
            self._http_agent.start()
            port = self._http_agent.wait_until_ready(STARTUP_TIMEOUT_SECONDS)
            self._register(f"http://{rank0_host}:{rank0_port}", token, port)
            return self._watch(heartbeat_interval)
        finally:
            self._http_agent.stop()

    def wait(self):
        """Not used on workers; rank 0 writes the cluster file."""
        raise RuntimeError("wait() is only valid on rank 0")

    def stop(self):
        """No-op: start() already stopped the HTTP server."""
        return

    def _rendezvous(self, timeout):
        """Poll until rank 0's host, port, interval, and token files exist."""
        agent_dir = self._layout.agent_dir
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                host, port, heartbeat_interval = self._read_rendezvous()
                return host, port, heartbeat_interval, _read_file(agent_dir / messages.AUTH_TOKEN_FILENAME)
            except (OSError, ValueError):
                time.sleep(POLL_INTERVAL_SECONDS)
        raise TimeoutError("rank-0 rendezvous did not appear")

    def _read_rendezvous(self):
        """Parse rank0.host, rank0.port, and the heartbeat interval once."""
        agent_dir = self._layout.agent_dir
        host = _read_file(agent_dir / RANK0_HOST_FILENAME)
        port = int(_read_file(agent_dir / RANK0_PORT_FILENAME))
        heartbeat_interval = float(_read_file(agent_dir / RANK0_HEARTBEAT_INTERVAL_FILENAME))
        if not host or not 0 < port <= 65535 or heartbeat_interval <= 0:
            raise ValueError("invalid rank-0 rendezvous")
        return host, port, heartbeat_interval

    def _register(self, endpoint, token, port):
        """POST this rank's hostname and port to rank 0, retrying transport and 5xx."""
        request = messages.RegisterRequest(rank=self._rank, hostname=self._http_agent.host, port=port)
        deadline = time.monotonic() + REGISTRATION_TIMEOUT_SECONDS
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

    def _watch(self, heartbeat_interval):
        """Block until rank0.done appears or rank0.alive stops updating."""
        agent_dir = self._layout.agent_dir
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


class AgentRunner:
    """Start HTTP agents for a scheduler-managed CVS run; rank 0 writes the cluster file."""

    def __init__(self, layout, cluster_file=None):
        """Read scheduler rank/hosts and construct Rank0Runner or WorkerRunner."""
        self._rank, world_size = scheduler_rank()
        hosts = scheduler_hosts()
        if len(hosts) != world_size:
            raise RuntimeError(
                f"managed CVS requires one task per node: scheduler expanded {len(hosts)} hosts "
                f"but SLURM_NTASKS is {world_size}"
            )
        host = hosts[self._rank]
        if self._rank == 0:
            self._role = Rank0Runner(layout, host, hosts, self._load_cluster_file(cluster_file))
        else:
            self._role = WorkerRunner(layout, self._rank, world_size, host)

    def _load_cluster_file(self, cluster_file):
        """Load optional --cluster_file JSON; empty object when the path is omitted."""
        if not cluster_file:
            return {}
        try:
            with open(cluster_file, "r", encoding="utf-8") as stream:
                return json.load(stream)
        except OSError as e:
            raise RuntimeError(str(e)) from e

    @property
    def is_rank0(self):
        """True if this process is scheduler rank 0."""
        return self._rank == 0

    def start(self):
        """Start this rank's role (HTTP + rendezvous, or worker register/watch)."""
        return self._role.start()

    def wait(self):
        """Rank 0: wait for registrations and return the cluster file path."""
        return self._role.wait()

    def stop(self):
        """Rank 0: stop heartbeat and HTTP; worker: no-op."""
        return self._role.stop()
