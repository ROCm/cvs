# CVS Cluster-Mon: RCCL Monitoring Extension

> **Date:** 2026-03-29  
> **Branch:** `users/nileshnegi/add-rcclras-support`  
> **Status:** Tested with 1-node and 2-nodes on Ruby MI350 cluster

---

## Problem Statement

Large-scale distributed training and inference jobs using RCCL often suffer from opaque failure modes: hangs from communicator deadlocks, silent performance degradation from degraded network links, segfaults from GPU memory errors, and cascading failures when a single node becomes unresponsive. Today, users have no unified way to observe RCCL's internal state during a live job — they resort to ad-hoc NCCL_DEBUG log analysis after the fact, losing critical temporal context.

NCCL ships a built-in RAS (Reliability, Availability, Serviceability) subsystem that exposes communicator health, peer mesh connectivity, and lifecycle events through a dedicated TCP service (ncclras).

RCCL inherits this subsystem `rcclras`, but no tooling exists to leverage it for continuous monitoring, correlation with system-level metrics (GPU health, RDMA errors, kernel logs), or integration with application-level signals (training step progression, loss curves).

---

## What does this "CVS RCCL Monitoring Extension" do?

CVS `cluster-mon` can now monitor live RCCL jobs in real time by connecting directly to the `rcclras` TCP service that runs inside every RCCL process on the cluster. When a rank dies, hangs, or loses connectivity, the dashboard reflects it within one poll cycle — no log parsing, no application-level timeout required.

Based on a cursory search, this is the only open-source tool that uses the RAS interface for RCCL/NCCL monitoring. Industry practice relies on post-mortem log analysis and training-side watchdog timeouts. Neither gives users visibility into communicator state while the job is still running.

---

## Background: What is `rcclras`

`rcclras` is a TCP server embedded in every RCCL process (port 28028, IPv6 loopback `[::1]` only). It exposes the internal communicator state machine via a line-oriented ASCII protocol:

```
CLIENT PROTOCOL 2          →  handshake
SERVER PROTOCOL 2          ←
TIMEOUT 30                 →  set collective timeout
OK                         ←
VERBOSE STATUS             →  request full dump
<multi-line text dump>     ←  streams header, then waits for all ranks to
<EOF>                          check in before sending communicator table
```

The server streams the response in two bursts. It sends the header and job summary immediately, then blocks until all ranks report before appending the communicator table. The client must read until EOF to get the full response.

`rcclras` is not reachable directly from the CVS backend — it binds only to the IPv6 loopback interface on each compute node. Access goes through an SSH port-forward tunnel.

---

## Architecture

```
rcclras :28028 (IPv6 loopback, each compute node)
    │
    │  SSH port-forward tunnel  (Pssh / JumpHostPssh)
    ▼
RCCLRasClient  ──  VERBOSE STATUS  ──►  RCCLTextParser
                                              │
                                         RCCLSnapshot
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                    Redis Streams       app_state             /ws/rccl
                   (ring buffer)   latest_rccl_snapshot      WebSocket
                          │
                    REST API ──► Frontend (3 pages)
```

**Collector cadence:** 30-second poll interval. The collector tries each healthy node in turn until it finds one with an active rcclras listener on port 28028. One successful response per cycle is sufficient — all ranks within a job report to the same rcclras instance.

**State machine:** The collector tracks job state across polls.

```
NO_JOB ──► HEALTHY ──► DEGRADED ──► NO_JOB
              │                         ▲
              └──────── NO_JOB ─────────┘
              │
         UNREACHABLE
              │
           ERROR
```

Every state transition emits a typed event (`job_start`, `job_end`, `job_degraded`, `job_recovered`, `node_unreachable`, etc.) stored in the event stream and visible on the Timeline page.

---

## Components

### RCCLRasClient
Async TCP client for the rcclras wire protocol. Takes a pre-connected `asyncio.StreamReader/Writer` from the SSH port-forward context manager. Handles the handshake, timeout setting, and VERBOSE STATUS dump. Reads until EOF in a loop — a single `read(n)` returns only the first burst and misses the communicator table.

Includes protocol version guards for `SET FORMAT json` (protocol v3, rcclras v2.28.9) and `MONITOR` mode (protocol v4, rcclras v2.29.2) — these are not yet enabled but the client won't send unknown commands to older servers.

### RCCLTextParser
Regex parser for the rcclras v2.28.3 VERBOSE STATUS text format. Built and tested against real output captured from a live MI300X cluster. Extracts:

- **Job summary** — node count, process count, GPU count, RCCL version, HIP/driver versions
- **Communicator table** — group number, comm count, rank counts, status, error column
- **Dead peers** — IP:port of unreachable peers
- **Errors section** — raw error lines reported by rcclras

The parser determines job state from the parsed data: `NO_JOB` if no valid output, `DEGRADED` if any communicator has missing ranks, dead peers, or errors, `HEALTHY` otherwise.

### RCCLCollector
`BaseCollector` subclass running on a 30-second cycle. Key behaviours:

- **Leader selection:** tries all healthy nodes (from `node_health_status`) in order until one has an active rcclras listener on port 28028.
- **Bootstrap:** on first poll after startup, seeds `job_state` from the last stored snapshot to avoid emitting a spurious `job_start` event on backend restart.
- **State transfer on config reload:** when configuration is reloaded and the collector is restarted, the previous `job_state` is copied to the new instance — same reason.
- **Timeout handling:** if the outer `asyncio.wait_for` fires, `on_collect_timeout()` updates the state machine to UNREACHABLE so the next cycle doesn't start from a stale state.

### RCCLDataStore
Dual-mode storage backend:

| Mode | When | Storage | Capacity |
|------|------|---------|----------|
| **Redis Streams** | Redis available | `rccl:snapshots`, `rccl:events` | 1 000 snapshots, 10 000 events |
| **In-memory deque** | No Redis / Redis error | `collections.deque` | 500 events, 100 snapshots |

Redis mode uses `XADD ... MAXLEN` — atomic append and cap in a single command. Time-range event queries use Redis Stream entry IDs (millisecond timestamps embedded). The in-memory fallback activates automatically if Redis is unavailable or throws an exception mid-operation.

### REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/rccl/status` | Latest snapshot: state, job summary, communicators, errors |
| `GET /api/rccl/communicators` | Communicator list from latest snapshot |
| `GET /api/rccl/communicators/{hash}` | Single communicator detail |
| `GET /api/rccl/events?since=&until=&type=` | Time-filtered event log. Returns `{events, truncated}` — `truncated: true` when the in-memory buffer is at capacity and older events may be missing |
| `POST /api/rccl/markers` | PyTorch training step/loss callback. Stores as `training_marker` event |
| `WebSocket /ws/rccl` | Real-time snapshot push on every collector cycle |

### Frontend Pages

**RCCL Health** — primary view. Shows job state banner (Healthy / Degraded / Unreachable / No Job), a staleness indicator when the snapshot is more than 75 seconds old (2.5× the poll interval), the raw rcclras Errors section when present, and a communicator card per group showing total/responding/missing rank counts.

**RAS Topology** — peer mesh visualization. Disabled for rcclras v2.28.3, which does not include per-peer connectivity in its text output. A compatibility note is shown; peer data is expected in a future rcclras version.

**Timeline** — chronological event log with type filter (job_start, job_end, degraded, recovered, etc.) and time-range selector. Shows `from_state → to_state` for state-change events and step/loss for training markers.

---

## RCCL Health — Live Screenshots

### Healthy State
All 16 ranks across 2 nodes responding. `group_0` communicator: 16/16 responding, 0 missing.

![RCCL Health - Healthy](images/rccl_health_good.png)

### Degraded State
One rank dropped mid-job. rcclras identifies the exact rank (Rank 7), GPU (GPU 7), PID (3871587), and node (10.245.40.180) in its Errors section. The communicator card reflects 15/16 responding, 1 missing.

![RCCL Health - Degraded](images/rccl_health_degraded.png)

---

## Known Limitations (rcclras v2.28.3)

**Single communicator group visible.** rcclras v2.28.3 exposes only the communicator group that rank 0 belongs to. A job using 8 independent communicator groups across 16 GPUs will show only one group. Full multi-group visibility is expected in a later rcclras version.

**No per-peer connectivity data.** The v2.28.3 text format does not include peer-level mesh data. The RAS Topology page is present but shows a compatibility notice until the data is available.

**In-memory events do not survive restarts.** Without Redis, events are held in a bounded in-memory buffer (500 events). Restarting the backend clears this history. Redis is not required for the core health dashboard — only for event history retention across restarts.

---

## Testing

Tested cluster-mon with long-running RCCL-Tests on 1-node and 2-nodes of Ruby MI350 cluster and introducing artificial chaos (e.g. killing a rccl-tests process). CVS backend/frontend app running on local laptop could directly connect to Ruby cluster compute nodes running RCCL.

Unit-Tests: 82 tests across 9 files. Coverage includes: BaseCollector lifecycle (timeout, crash, ConnectionError, supervisor restart), Pydantic config defaults and environment variable overrides, SSH bridge (bidirectional data, EOF propagation), collectors/status API, RAS protocol client against a mock TCP server, text parser against 4 fixture files (healthy, single-node degraded, 2-node degraded with heterogeneous `7-8` ranks-per-node range, connection reset), RCCL collector state machine (all 20 transitions, bootstrap, no-duplicate-event on unchanged state), WebSocket ConnectionManager, and config reload diff detection.

---

## Future Work

| Phase | Scope |
|-------|-------|
| **2** | Switch to JSON output (rcclras v2.28.9)<br>Prometheus `/metrics` endpoint<br>InfluxDB long-term storage (structured data pipeline) |
| **3** | Persistent `MONITOR` mode (rcclras v2.29.2) for push-based event streaming (eliminates polling)<br>Per-rank structured error parsing |
| **4** | `/api/rccl/preflight` for Slurm prolog health gate<br>Slurm job ID correlation<br>Grafana dashboard templates<br>Snapshot replay for post-mortem analysis |
