# CVS Cluster-Mon: RCCL Monitoring Extension — Technical Report

> **Date:** 2026-03-29
> **Branch:** `users/nileshnegi/add-rcclras-support`
> **Scope:** 42 files changed, +3,796 / -806 lines, 15 commits, 82 tests

---

## Executive Summary

This work adds **real-time RCCL communicator health monitoring** to the CVS cluster-mon tool by connecting directly to the `rcclras` RAS TCP service (port 28028) running inside every RCCL process. This is a **novel approach** — no existing open-source tool monitors RCCL/NCCL through the RAS interface. Industry practice relies on post-mortem log parsing and application-level timeouts, neither of which provides real-time visibility into communicator state during a live job.

The implementation required two foundational changes: (1) a **robustness refactor** of the existing collector architecture (BaseCollector ABC, Pydantic config migration, SSH port-forwarding, Redis integration), and (2) the **RCCL monitoring extension** itself (RAS protocol client, text parser, collector, REST/WebSocket API, 3 React dashboard pages). The text parser was built test-first against real `rcclras -v` output captured from a live MI300X cluster.

---

## What Was Built

### Robustness Improvements (Foundation)

| Component | What It Does |
|-----------|-------------|
| **BaseCollector ABC** | Unified collector lifecycle with `asyncio.wait_for` timeout enforcement, supervisor pattern with auto-restart on crash (exponential backoff to 2 min), per-collector `critical` flag for aggregate health |
| **Pydantic Config** | Replaced hand-rolled `simple_config.py` with typed `Settings(BaseSettings)` using `_YamlSource` for YAML loading, `env_nested_delimiter="__"` for env var overrides. Added `StorageConfig`, `RCCLConfig` sections |
| **SSH Port-Forwarding** | `open_port_forward()` async context manager on both `Pssh` and `JumpHostPssh`. Uses `socketpair()` bridge (no ephemeral TCP ports, no TOCTOU race). Enables TCP tunneling to any port on compute nodes |
| **Redis Integration** | `redis.asyncio` client initialized in lifespan with graceful degradation (app continues without Redis). Docker Compose includes `redis:7-alpine` with AOF persistence, healthcheck, and auth |
| **Collectors Status API** | `GET /api/collectors/status` returns per-collector state + `overall_status` (healthy/degraded/critical) |
| **ConnectionManager** | Per-client bounded `asyncio.Queue` with dedicated sender tasks. Slow WebSocket clients are auto-disconnected instead of blocking the collector loop |

### RCCL Monitoring Extension (Phase 1)

| Component | What It Does |
|-----------|-------------|
| **RCCLRasClient** | Async TCP client speaking the rcclras wire protocol (handshake, TIMEOUT, VERBOSE STATUS). Protocol version guards for future SET FORMAT (v3) and MONITOR (v4) |
| **RCCLTextParser** | Regex-based parser for rcclras v2.28.3 VERBOSE STATUS output. Built test-first against 3 real fixture files captured from a live MI300X job (healthy, degraded with missing rank, connection reset) |
| **RCCLCollector** | `BaseCollector` subclass. Iterates all healthy nodes (not just first) to find an active rcclras listener, opens SSH port-forward to `[::1]:28028` (IPv6 loopback — required since rcclras does not bind to `0.0.0.0`), runs the RAS protocol, parses response, stores in data store, broadcasts via WebSocket. State machine: NO_JOB / UNREACHABLE / HEALTHY / DEGRADED / ERROR. Auto-generates typed state-change events (10-transition map: `job_start`, `job_start_degraded`, `job_degraded`, `job_recovered`, `job_end`, `node_unreachable`, `collector_error`) on every state transition |
| **RCCLDataStore** | Redis Streams-based ring buffer (XADD+MAXLEN — atomic, no LPUSH+LTRIM race). 1000 snapshot entries, 10000 event entries. Time-range queries via stream entry IDs. Falls back to bounded in-memory deque (500 events, 100 snapshots) when Redis is unavailable — events survive without Docker |
| **REST API** | `GET /api/rccl/status`, `/communicators`, `/communicators/{hash}`, `/events` (time-filtered), `POST /markers` (PyTorch callback endpoint, Pydantic-validated) |
| **WebSocket** | `/ws/rccl` for real-time RCCL snapshot streaming via `broadcast_rccl()` |
| **Frontend** | 3 React + TypeScript pages: **RCCLHealth** (job state banner, communicator grid with per-rank error highlighting, rcclras Errors section), **RCCLTopology** (peer mesh with v2.28.3 compatibility note), **RCCLTimeline** (event timeline with state-change event types and from/to state display). Sidebar nav with RCCL section label |

---

## Tasks Accomplished

- [x] BaseCollector ABC with collect_timeout, supervisor, probe_requested
- [x] Pydantic Settings migration (simple_config.py deleted)
- [x] SSH port-forwarding on Pssh + JumpHostPssh
- [x] Thread safety: _exec_lock + _hosts_lock separation (deadlock fix)
- [x] All GPU/NIC collector methods use exec_async() (no event loop blocking)
- [x] Redis init with auth, graceful degradation, docker-compose
- [x] /api/collectors/status with overall_status
- [x] ConnectionManager with per-client queues
- [x] RCCL data models (Pydantic) — `errors: list[str]` field added to `RCCLSnapshot`
- [x] RCCLRasClient with protocol version guards
- [x] RCCLTextParser from real MI300X fixtures
- [x] RCCLCollector — `_pick_leader` replaced by `_healthy_nodes` (tries all healthy nodes); SSH port-forward dest fixed to `"::1"` (IPv6 loopback); `_health_from_snapshot` delegates to `snapshot.state` (parser already computes correct state); state-change event generation
- [x] RCCLDataStore — Redis Streams + in-memory fallback; `latest_rccl_snapshot` cleared on job end/unreachable
- [x] RCCL REST API + /ws/rccl WebSocket
- [x] Frontend: 3 React pages + navigation
- [x] Topology-diff config reload
- [x] 82 unit tests (3 new: 2-node degraded fixture with heterogeneous `ranks_per_node` range)

## Testing Summary

**82 tests** across 9 test files covering: BaseCollector lifecycle (timeout, cancellation, ConnectionError), config defaults and env var overrides, SSH bridge (bidirectional data, EOF propagation, daemon threads), collector attributes and collect() return types, collectors/status API logic, RAS protocol client against mock TCP server, text parser against 4 real fixtures (healthy, single-node degraded, 2-node degraded with heterogeneous `ranks_per_node` range `7-8`, connection reset), RCCL collector edge cases including state-change event emission and in-memory data store fallback, WebSocket ConnectionManager, and reload diff detection.

## Known Limitations (RCCL v2.28.3)

- **Only `group_0` visible per job**: rcclras exposes only the communicator group that rank 0 belongs to. Jobs with multiple communicator groups (e.g. 8 groups across 16 GPUs) show only one group in the output. This is a known RAS limitation; full group visibility planned for rcclras v2.29+.
- **No peer mesh data**: per-peer connectivity (`RCCLPeer`) is not included in the v2.28.3 text output. RAS Topology page displays a compatibility note.
- **Events lost on restart**: in-memory fallback (no Redis) stores up to 500 events per process lifetime. Events are not persisted across backend restarts without Redis.

## TODOs (Future Phases)

- [ ] **Phase 1.5:** Re-enable `RAS_COLL_CONNS` in RCCL (3 `#if 0` blocks) for connection latency stats
- [ ] **Phase 2:** JSON parser when RCCL syncs to v2.28.9 (`SET FORMAT json`); Prometheus `/metrics` endpoint; InfluxDB long-term storage
- [ ] **Phase 3:** Monitor mode (`MONITOR all` persistent connection) when RCCL syncs to v2.29.2; real-time event streaming; per-rank structured error parsing from Errors section
- [ ] **Phase 4:** Grafana dashboard templates; `/api/rccl/preflight` endpoint for Slurm prolog; Slurm job ID correlation; snapshot replay for post-mortem
- [ ] Integration test: full collector-to-Redis-to-API pipeline with mock rcclras server
