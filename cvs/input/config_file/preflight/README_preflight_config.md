# Preflight Configuration Guide

This document explains how to configure the GPU cluster preflight checks system.

## Overview

The preflight checks system validates essential cluster health before running performance tests like IB performance tests, RCCL training, and inference workloads. It performs the following validations:

1. **Node Health** - Checks GPU visibility, AMDGPU/KFD, kernel health, and ROCm consistency
2. **MI4XX Scale-up Fabric Admission** - Optionally validates AIFM/AFM/vPOD membership, station masks, and IFoE port state
3. **IFoE L2 Connectivity (AIMVT-180; opt-in)** - Optionally runs strict `afmctl test ping` coverage before TransferBench and RDMA
4. **TransferBench** - Optionally validates the IFoE data path per node or with a multi-rank cluster run
5. **Node Smoke Tier 1 (opt-in)** - Per-node host / GPU / RDMA roll-call via `primus-cli direct -- node_smoke`
6. **Node Smoke Tier 2 (optional)** - Per-node perf sanity when `node_smoke_tier1.tier2_perf` is enabled (GEMM TFLOPS, HBM bandwidth, local RCCL)
7. **Node Smoke Tier 3 (opt-in)** - Cluster-wide Host / GPU / Network inventory via `primus-cli direct -- preflight --host --gpu --network`
8. **GID and Interface Consistency** - Ensures configured RDMA interfaces and GID entries are present and consistent
9. **RDMA Connectivity** - Tests node-to-node RDMA communication using `ibv_rc_pingpong`

## Configuration File Structure

The preflight configuration file follows this structure:

```json
{
  "preflight": {
    "node_check": {
      "enabled": true,
      "gpus_per_node": 4,
      "expected_rocm_version": "7.15.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "skip",
        "gid_index": "3",
        "interfaces": ["enp4s0np0"],
        "nodes_per_full_mesh_group": 128,
        "ibv_test_timeout": 90,
        "ibv_test_port_range": "10000-50000",
        "inter_full_mesh_group_pairs_per_wave": "auto"
      },
      "ifoe": {
        "fabric_checks": false,
        "l2ping": {
          "enabled": true,
          "pings_per_port": 3
        },
        "transferbench": {
          "enabled": true,
          "scope": "node",
          "profile": "smoketest",
          "message_sizes": ["1K", "16M"],
          "iterations": 2,
          "warmup_iterations": 0
        }
      }
    },
    "node_smoke_tier1": {
      "connectivity_mode": "skip",
      "auto_setup": true,
      "primus_dir": "/home/{user-id}/INSTALL/Primus",
      "venv_activate": "/home/{user-id}/envs/preflight/.venv/bin/activate",
      "gpus_per_node": 8,
      "tier2_perf": false
    },
    "node_smoke_tier3": {
      "connectivity_mode": "skip"
    },
    "reporting": {
      "generate_html_report": true,
      "artifacts_root_dir": "/home/{user-id}/preflight",
      "generate_rdma_pairs_csv": false
    },
    "debug": {
      "scriptlet": false
    }
  }
}
```

## Configuration Structure

The preflight configuration uses a **nested structure organized by subsystem** for better organization and future extensibility:

### Structure Overview

```
preflight/
├── node_check/                 # Generation-independent node validation
├── connectivity_check/        # Connectivity tests grouped by protocol
│   ├── rdma/                  # RDMA inventory, GID, and pairwise connectivity
│   └── ifoe/                  # MI4XX scale-up fabric checks
│       ├── l2ping/            # Strict IFoE L2 connectivity gate
│       └── transferbench/     # IFoE data-path validation
├── node_smoke_tier1/          # Node Smoke Tier 1/2 per-node JSON artifacts (opt-in)
├── node_smoke_tier3/          # Node Smoke Tier 3 cluster markdown report (opt-in)
├── reporting/                 # Output and report generation
└── debug/                     # Debug and troubleshooting options
```

Legacy config keys `node_smoke` and `tier3_info` are still accepted and normalized to
`node_smoke_tier1` and `node_smoke_tier3` at load time.

### Execution Flow
1. **node_check** - Validate individual nodes and, when enabled, MI4XX scale-up fabric admission
2. **connectivity_check.ifoe.l2ping** - Enforce strict L2 connectivity
3. **connectivity_check.ifoe.transferbench** - Validate the IFoE data path
4. **node_smoke_tier1** - Run Node Smoke Tier 1 (and Tier 2 when `tier2_perf` is enabled) per node
5. **node_smoke_tier3** - Run Node Smoke Tier 3 cluster inventory (independent of Tier 1)
6. **connectivity_check.rdma** - Validate RDMA inventory, GIDs, and connectivity unless skipped
7. **reporting** - Generate reports and outputs

## Node Smoke tiers and test counts

Preflight reports **Node Smoke Tier 1**, **Tier 2**, and **Tier 3** as separate checks in the
console summary and HTML report. Each tier appends a test-count suffix when enabled:

```
✅ Node Smoke Tier 1: PASS - 2/2 nodes passed Node Smoke Tier 1; 39 tests run per node
✅ Node Smoke Tier 2: PASS - 2/2 nodes passed Node Smoke Tier 2; 17 tests run per node
✅ Node Smoke Tier 3: PASS - 2/2 nodes passed Node Smoke Tier 3; 27 tests run cluster-wide
```

Counts follow the validation-tracker catalog in `cvs/lib/preflight/node_smoke_counts.py`. Tier 1
and Tier 2 counts are **per node** (not multiplied across the cluster in the summary). Tier 3 is
**cluster-wide** (one catalog run, not × node count).

### Tier 1 — per node (39 on an 8-GPU node)

Formula: `4 × gpus_per_node + 7` node operational collectors.

| Category | Count (8 GPU) | Notes |
|----------|---------------|-------|
| Per-GPU subprocess checks | 32 | 4 checks × 8 GPUs |
| Node operational collectors | 7 | `gpu_processes`, `nics`, `host_limits`, `gpu_low_level`, `xgmi`, `tooling`, `gpu_visibility` |

Inventory-style findings (`gpu_info`, `host_info`, `network_info`), fingerprint, clock, and dmesg
are excluded from the Tier 1 count (they are drift/inventory collectors, not operational gates).

### Tier 2 — per node (17 on an 8-GPU node)

Enabled with `node_smoke_tier1.tier2_perf: true`. Formula: `2 × gpus_per_node + 1` (RCCL omitted when `gpus_per_node < 2`).

| Check | Count (8 GPU) |
|-------|---------------|
| Large GEMM TFLOPS floor (8192³ bf16) | 8 |
| HBM device-to-device bandwidth | 8 |
| Local multi-GPU RCCL all-reduce | 1 |

### Tier 3 — cluster-wide (27 checks)

Runs `preflight --host --gpu --network` once across the cluster. CVS counts **27 individual
collector checks** from the validation tracker, not the **13** aggregated markdown report
sections Primus emits (for example, one `## CPU` table covers all hosts but the tracker still
lists CPU as its own check).

| Group (`--flag`) | Checks |
|------------------|--------|
| Host (`--host`) | Host identity (×2), CPU, Memory (×2), NUMA, PCIe inventory, PCIe link status (×3) — **10** |
| GPU (`--gpu`) | GPU enumeration, identity, occupancy, GPU/NUMA mapping, topology (×2), perf sanity (×2) — **8** |
| Network (`--network`) | Network summary, distributed intent, distributed env, network path (×2), InfiniBand/RDMA, RCCL/NCCL config (×2), runtime process group — **9** |

## Configuration Parameters

### Complete Parameter Reference

All parameters below are optional and have sensible defaults. The sample configuration file includes all available parameters with their default values and inline comments explaining their purpose.

### Important Update: RDMA Connectivity Testing

**As of this version, preflight checks now use `ibv_rc_pingpong` instead of `rping` for RDMA connectivity testing.**

**Why this change?**
- `rping` uses RDMA Connection Manager (`rdma_cm`) which is more forgiving of network issues
- `ibv_rc_pingpong` uses direct InfiniBand verbs (same as RCCL) which is more strict
- This change allows preflight checks to detect the same connectivity issues that cause RCCL failures
- **Result**: Better correlation between preflight results and actual RCCL performance

**Updated parameter names**: Configuration parameters now use accurate names (`ibv_test_timeout`, `ibv_test_port_range`) that reflect the use of `ibv_rc_pingpong` for testing.

### RDMA Batching (`connectivity_check.rdma`)

- **`nodes_per_full_mesh_group`** (default: 128)
  - Group size for parallel RDMA connectivity testing (2-512 nodes per group)
  - Smaller groups use fewer resources per node but require more rounds
  - Adjust based on cluster size and resource constraints

### Debug Settings (`debug`)

- **`scriptlet`** (default: false)
  - Enable ScriptLet debug mode: preserve generated scripts/logs on remote nodes
  - For RDMA connectivity, wraps each ibv_rc_pingpong server in strace
  - Creates per-test traces under /tmp/preflight/strace_server_<iface>_<port>.log
  - **Warning**: Can be expensive at scale due to strace overhead

### Node Check Settings (`node_check`)

- **`enabled`** (default: `true`)
  - Enables GPU visibility, AMDGPU/KFD, kernel-health, and ROCm validation
  - Set to `false` to skip node-local health checks

- **`gpus_per_node`** (default: `4`)
  - Exact number of AMD GPUs expected on every node
  - GPU visibility is generation-independent and can run on older or newer AMD hardware

- **`expected_rocm_version`** (default: "6.2.0")
  - Expected ROCm version across all cluster nodes
  - Must match the output of `amd-smi version` on all nodes
  - Format: "major.minor.patch" (e.g., "6.2.0", "5.7.1")

### Connectivity Check Settings (`connectivity_check`)

#### RDMA Settings (`connectivity_check.rdma`)

- **`connectivity_mode`** (default: "basic")
  - **"basic"**: Test adjacent node pairs (fast, ~14% coverage for 8 nodes)
  - **"full_mesh"**: Test all possible node pairs (comprehensive, 100% coverage)
  - **"skip"**: Skip RDMA interface presence, GID validation, and pairwise connectivity

- **`gid_index`** (default: "3")
  - GID index to check on all configured RDMA interfaces
  - Typically "3" for RoCE (RDMA over Converged Ethernet)
  - Must be a valid GID index for your InfiniBand/RoCE setup

- **`interfaces`** (default: `["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]`)
  - List of RDMA device names that should be present on all cluster nodes
  - Examples:
    - `["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]` - Standard 4-interface setup
    - `["mlx5_0", "mlx5_1"]` - Mellanox 2-interface setup

- **`ibv_test_timeout`** (default: 90)
  - Timeout in seconds for each ibv_rc_pingpong connectivity test
  - Integer value (seconds), used directly as configured
  - Uses `ibv_rc_pingpong` (direct InfiniBand verbs) for RCCL-compatible testing
  - Increase for slower networks or high-latency connections

- **`ibv_test_port_range`** (default: "10000-50000")
  - Port range for ibv_rc_pingpong tests to avoid conflicts
  - Format: "start-end" (e.g., "10000-50000", "10000-10999")
  - Ensure ports are not blocked by firewalls

- **`inter_full_mesh_group_pairs_per_wave`** (default: "auto")
  - Max ordered group-pairs (Gi→Gj keys) per wave during inter-group RDMA testing
  - "auto" calculates as max(1, num_groups - 1)
  - Can be set to a specific integer to control wave size and reduce memory/CPU load

- **`prune_failure_threshold`** (default: 0.5)
  - Prune nodes whose fraction of peers with ≥1 FAIL intra test is ≥ this value
  - Range: 0.0 to 1.0 (0.5 = 50% failure threshold)
  - Helps remove problematic nodes before inter-group testing
  - Lower values (0.2-0.3) are more aggressive at removing problematic nodes

- **`port_retry_max`** (default: 3)
  - Max retry attempts for pairs whose logs show PORT_LISTEN_FAILED
  - Range: 0-10 retries with new TCP ports after each wave
  - Helps handle port conflicts during large-scale testing

- **`port_retry_gap`** (default: 1000)
  - Port gap when remapping ports for PORT_LISTEN_FAILED retries
  - Range: 1-65535
  - Starts at (max port in batch) + this gap to reduce overlap with ephemeral ports

- **`exclude_failed_interface_nodes`** (default: "true")
  - Legacy hint for reporting: preflight now prunes interface/GID-failed nodes automatically
  - Interface failures are excluded from mesh testing regardless of this flag

##### Legacy RDMA paths — deprecated

Existing RDMA users may temporarily retain `node_check.gid_index` and
`node_check.rdma_interfaces`. CVS normalizes them to
`connectivity_check.rdma.gid_index` and `connectivity_check.rdma.interfaces`
and emits a deprecation warning. If a legacy and canonical value are both
present, they must match. New configurations should use the canonical RDMA
paths; the compatibility paths will be removed in a future release.

#### IFoE Settings (`connectivity_check.ifoe`) — MI4XX scale-up fabric

IFoE validation is organized into fabric admission, strict L2 connectivity,
and TransferBench data-path validation. CVS owns `afmctl` discovery, privilege
handling, BDF and port discovery, strict coverage, traffic selection, timeout
derivation, and result parsing.

The earlier configuration shape exposed those implementation details directly.
They now follow this fixed policy:

| Previous setting | Current CVS behavior |
|---|---|
| `connectivity_mode` | Replaced by `l2ping.enabled` |
| `afmctl_path` | Resolve `afmctl` from the node environment before privilege escalation |
| `use_sudo` | Use the cluster's detected privilege policy |
| `bdf_discovery` / `bdfs` | Discover admitted AFM devices and BDFs from live topology |
| `dst_accelerators` | Build strict destination coverage from reconciled vPOD membership |
| `ports` | Test admitted, station-mask-enabled ports that are operationally up |
| `traffic_types` | Enforce IFoE request, IFoE response, and non-IFoE traffic |
| `loss_threshold_pct` | Fail on any reported loss or incomplete coverage |
| `per_ping_timeout` / `ssh_timeout` | Derive conservative timeouts from the requested workload |

- **`fabric_checks`** (default: `false`)
  - Enables MI4XX-only AIFM/AFM/vPOD, station-mask, and IFoE port admission checks
  - Set to `true` only on MI4XX systems; it remains disabled for MI3XX systems
  - Requires `node_check.enabled: true`

##### L2 ping (`connectivity_check.ifoe.l2ping`)

Runs `afmctl test ping` with strict full-mesh coverage on every admitted IFoE
port and validates per-port and aggregate summary accounting.

- **`enabled`** (default: `false`)
  - Enables the mandatory L2 connectivity gate before TransferBench and RDMA
- **`pings_per_port`** (default: `3`)
  - Number of ping samples sent per selected IFoE port pair

##### TransferBench (`connectivity_check.ifoe.transferbench`)

- **`enabled`** (default: `false`)
  - Enables the TransferBench IFoE data-path gate before RDMA
- **`scope`** (default: `"node"`)
  - `"node"` runs an independent smoketest on each node
  - `"cluster"` runs one multi-rank test across the admitted cluster
- **`profile`** (default: `"smoketest"`)
  - Selects the CVS-supported test profile; `"smoketest"` is currently supported
- **`message_sizes`** (default: `["1K", "16M"]`)
  - Message sizes exercised by the selected profile
- **`iterations`** (default: `2`)
  - Validated iterations per test and message size
- **`warmup_iterations`** (default: `0`)
  - Warmup iterations performed before validation

#### Node Smoke Tier 1 (`node_smoke_tier1`) — opt-in

Runs Node Smoke Tier 1 (Primus `node_smoke`) on each reachable node via `primus-cli direct --single -- node_smoke`
over parallel SSH (no Slurm required). Reference: Primus `docs/02-user-guide/node-smoke-test-instruction.md`
on branch `dev/preflight-direct-test`.

Legacy config key `node_smoke` is accepted as an alias for `node_smoke_tier1`.

- **`connectivity_mode`** (default: `"skip"`)
  - `"run"` — execute Node Smoke Tier 1 on every reachable node
  - `"skip"` — preflight records a SKIPPED result and does not invoke Primus
- **`auto_setup`** (default: `true`)
  - Clone/update Primus and create the venv with minimal deps (ROCm PyTorch) before Node Smoke Tier 1
- **`setup_timeout`** (default: `600`)
  - SSH timeout (seconds) for the per-node Primus auto_setup step
- **`force_reclone`** (default: `false`)
  - Remove `primus_dir` and clone fresh on every run (destructive)
- **`shared_install`** (default: `true`)
  - Leader node clones/installs on shared NFS home; other nodes wait (recommended for shared home)
- **`pip_install_mode`** (default: `"minimal"`)
  - `"minimal"` — ROCm PyTorch only; `"requirements"` — `pip install -r requirements.txt`; `"skip"` — venv only
- **`torch_pip_index_url`** (default: `"https://download.pytorch.org/whl/rocm6.2"`)
  - PyTorch wheel index for minimal install; match your ROCm version
- **`primus_git_url`** (default: `"https://github.com/AMD-AIG-AIMA/Primus.git"`)
- **`primus_git_branch`** (default: `"dev/preflight-direct-test"`)
- **`primus_git_recurse_submodules`** (default: `false`)
- **`primus_dir`** (default: `"/home/{user-id}/INSTALL/Primus"`)
  - Required when `connectivity_mode` is `"run"`; `{user-id}` is resolved at runtime
- **`venv_activate`** (default: `"/home/{user-id}/envs/preflight/.venv/bin/activate"`)
  - Required when `connectivity_mode` is `"run"`
- **`gpus_per_node`** (default: `8`)
- **`master_port`** (default: `1234`)
- **`dump_path`** (default: `""`)
  - Per-node smoke JSON output; empty uses `<reporting.artifacts_root_dir>/node_smoke`
- **`expected_rdma_nics`** (default: `null`)
  - Defaults to `len(node_check.rdma_interfaces)` when null
- **`ulimit_l_min_gb`** (default: `32`) — FAIL below this memlock limit; `0` disables
- **`shm_min_gb`** (default: `8`) — FAIL below this `/dev/shm` size; `0` disables
- **`skip_dmesg`** (default: `false`)
- **`allow_foreign_procs`** (default: `false`)
- **`allowed_procs`** (default: `"gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter"`)
- **`require_tools`** (default: `""`) — empty = warn only
- **`nccl_socket_ifname`** / **`gloo_socket_ifname`** (default: `""`)
- **`nccl_ib_hca`** (default: `""`) — defaults to comma-joined `node_check.rdma_interfaces`
- **`nccl_ib_gid_index`** (default: `null`) — defaults to `node_check.gid_index`
- **`ssh_timeout`** (default: `300`)
- **`extra_args`** (default: `[]`) — additional flags forwarded to primus-cli

#### Node Smoke Tier 2 perf sanity (`node_smoke_tier1.tier2_perf`) — optional

When `tier2_perf` is `true`, preflight forwards `--tier2-perf` to Primus `node_smoke`, enabling all three Tier 2 checks on each node (same as `launch_nodesmoke_ssh.sh -- --tier2-perf`):

1. **Large GEMM TFLOPS floor** — 8192³ bf16 `torch.matmul`; FAIL below `gemm_tflops_min` (default 600)
2. **HBM D2D bandwidth** — 512 MB device-to-device copy; FAIL below `hbm_gbs_min` (default 2000 GB/s)
3. **Local multi-GPU RCCL all-reduce** — node-local only; FAIL below `rccl_gbs_min` (default 100 GB/s)

Set `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, and `NCCL_IB_GID_INDEX` (via `node_smoke_tier1` config or cluster `env_vars`) before enabling Node Smoke Tier 2 — RCCL init enumerates every transport even though the all-reduce is local-only.

- **`tier2_perf`** (default: `false`) — master switch; maps to `--tier2-perf`
- **`gemm_tflops_min`** (default: `600`) — `--gemm-tflops-min`
- **`hbm_gbs_min`** (default: `2000`) — `--hbm-gbs-min`
- **`rccl_gbs_min`** (default: `100`) — `--rccl-gbs-min`
- **`rccl_size_mb`** (default: `64`) — `--rccl-size-mb`
- **`rccl_timeout_sec`** (default: `120`) — `--rccl-timeout-sec`

Tier 2 runs need a longer SSH budget; when `tier2_perf` is enabled the effective timeout is at least 600 seconds even if `ssh_timeout` is lower.

#### Node Smoke Tier 3 (`node_smoke_tier3`) — opt-in

Runs Node Smoke Tier 3 (`primus-cli direct -- preflight --host --gpu --network`) across the cluster with a distributed rendezvous. Independent of Tier 1 — enabling `node_smoke_tier1` does not enable Tier 3.

- **`connectivity_mode`** (default: `"skip"`) — `"run"` or `"skip"`
- **`auto_setup`** (default: `true`) — clone/update Primus and create venv before Tier 3 (falls back to Tier 1 paths)
- **`primus_dir`** / **`venv_activate`** — optional; empty inherits from `node_smoke_tier1`
- **`gpus_per_node`** (default: `8`) — GPUs per node for torchrun
- **`master_port`** (default: `1234`) — `MASTER_PORT` for the distributed env
- **`dump_path`** — empty uses `<reporting.artifacts_root_dir>/node_smoke_tier3`
- **`report_file_name`** (default: `"node_smoke_tier3"`) — base name for Primus markdown/PDF reports
- **`dist_timeout_sec`** (default: `120`) — timeout for `torch.distributed` init while aggregating the report
- **`save_pdf`** (default: `false`) — also emit a PDF report when true
- **`nccl_socket_ifname`** / **`gloo_socket_ifname`** / **`nccl_ib_hca`** / **`nccl_ib_gid_index`** — NCCL transport overrides (same semantics as Tier 1)
- **`ssh_timeout`** (default: `600`) — SSH timeout for the parallel cluster run
- **`extra_args`** (default: `[]`) — additional flags forwarded to `primus-cli`

Legacy config key `tier3_info` is accepted as an alias for `node_smoke_tier3`.

### Reporting Settings (`reporting`)

- **`generate_html_report`** (default: `true`)
  - Whether to generate detailed HTML report
  - Set to `false` to disable HTML report generation

- **`artifacts_root_dir`** (default: `"/home/{user-id}/preflight"`)
  - Root directory where preflight artifacts are saved
  - Includes HTML reports and RDMA full_mesh workspace logs under `rdma_connectivity_workspace/`
  - Must be writable by the user running the tests

- **`generate_rdma_pairs_csv`** (default: `true`)
  - Whether to generate CSV file with failed RDMA pairs alongside HTML report
  - Set to `false` to disable CSV generation

## Usage Examples

### Basic 8-Node Cluster Check

```json
{
  "preflight": {
    "node_check": {
      "enabled": true,
      "gpus_per_node": 8,
      "expected_rocm_version": "6.2.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "basic",
        "gid_index": "3",
        "interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
        "ibv_test_timeout": 90,
        "ibv_test_port_range": "10000-50000"
      }
    }
  }
}
```

### Comprehensive Full-Mesh Testing

```json
{
  "preflight": {
    "node_check": {
      "enabled": true,
      "gpus_per_node": 8,
      "expected_rocm_version": "6.2.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "full_mesh",
        "gid_index": "3",
        "interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
        "ibv_test_timeout": 120,
        "ibv_test_port_range": "10000-50000"
      }
    }
  }
}
```

### Configuration-Only Validation (Skip Connectivity)

```json
{
  "preflight": {
    "node_check": {
      "enabled": true,
      "gpus_per_node": 8,
      "expected_rocm_version": "6.2.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "skip"
      }
    }
  }
}
```

### Enable Primus Node Smoke with Tier 2 perf

```json
{
  "preflight": {
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.4.2",
      "rdma_interfaces": ["rdma0", "rdma1", "rdma2", "rdma3", "rdma4", "rdma5", "rdma6", "rdma7"]
    },
    "node_smoke_tier1": {
      "connectivity_mode": "run",
      "auto_setup": true,
      "shared_install": true,
      "primus_dir": "/home/{user-id}/INSTALL/Primus",
      "venv_activate": "/home/{user-id}/envs/preflight/.venv/bin/activate",
      "gpus_per_node": 8,
      "tier2_perf": true,
      "gemm_tflops_min": 700,
      "hbm_gbs_min": 4500,
      "rccl_gbs_min": 180,
      "nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
      "nccl_ib_gid_index": 3,
      "ssh_timeout": 600
    }
  }
}
```

### Enable Node Smoke Tier 1 and Tier 3

```json
{
  "preflight": {
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.4.2",
      "rdma_interfaces": ["rdma0", "rdma1", "rdma2", "rdma3", "rdma4", "rdma5", "rdma6", "rdma7"]
    },
    "node_smoke_tier1": {
      "connectivity_mode": "run",
      "auto_setup": true,
      "shared_install": true,
      "primus_dir": "/home/{user-id}/INSTALL/Primus",
      "venv_activate": "/home/{user-id}/envs/preflight/.venv/bin/activate",
      "gpus_per_node": 8
    },
    "node_smoke_tier3": {
      "connectivity_mode": "run",
      "ssh_timeout": 600
    }
  }
}
```

### Enable Primus Node Smoke

```json
{
  "preflight": {
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.4.2",
      "rdma_interfaces": ["rdma0", "rdma1", "rdma2", "rdma3", "rdma4", "rdma5", "rdma6", "rdma7"]
    },
    "node_smoke_tier1": {
      "connectivity_mode": "run",
      "auto_setup": true,
      "shared_install": true,
      "primus_dir": "/home/{user-id}/INSTALL/Primus",
      "venv_activate": "/home/{user-id}/envs/preflight/.venv/bin/activate",
      "gpus_per_node": 8
    }
  }
}
```

### Advanced Configuration with Debug and Tuning

```json
{
  "preflight": {
    "debug": {
      "scriptlet": true
    },
    "node_check": {
      "enabled": true,
      "gpus_per_node": 8,
      "expected_rocm_version": "7.2.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "full_mesh",
        "gid_index": "3",
        "interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0", "rocep158s0", "rocep190s0", "rocep206s0", "rocep222s0"],
        "nodes_per_full_mesh_group": 32,
        "ibv_test_timeout": 180,
        "ibv_test_port_range": "15000-20000",
        "inter_full_mesh_group_pairs_per_wave": 4,
        "prune_failure_threshold": 0.3,
        "port_retry_max": 5,
        "port_retry_gap": 2000
      }
    },
    "reporting": {
      "generate_html_report": true,
      "artifacts_root_dir": "/home/{user-id}/preflight",
      "generate_rdma_pairs_csv": true
    }
  }
}
```

## Running Preflight Checks

```bash
# Basic usage with default config
cvs run preflight_checks --cluster_file cluster.json --config_file preflight_config.json

# Run only Node Smoke Tier 1
cvs run preflight_checks test_node_smoke_tier1 \
  --cluster_file cluster.json \
  --config_file preflight_config.json

# Run only Node Smoke Tier 3
cvs run preflight_checks test_node_smoke_tier3 \
  --cluster_file cluster.json \
  --config_file preflight_config.json

# With custom HTML output
cvs run preflight_checks \
  --cluster_file cluster.json \
  --config_file preflight_config.json \
  --html /path/to/preflight_report.html \
  --self-contained-html
```

## Troubleshooting

### Common Issues

1. **GID Check Failures**
   - Ensure RDMA drivers are loaded: `lsmod | grep rdma`
   - Check interface status: `rdma link show`
   - Verify GID entries: `cat /sys/class/infiniband/*/ports/1/gids/3`

2. **RDMA Connectivity Failures**
   - Check firewall settings: `sudo ufw status`
   - Verify ibv_rc_pingpong is available: `which ibv_rc_pingpong`
   - Enable debug mode: set `"scriptlet": true` for detailed logs
   - Check port conflicts in the specified `ibv_test_port_range`
   - Test manual connectivity: `ibv_rc_pingpong -d <device> -g <gid_index>`

3. **ROCm Version Mismatches**
   - Check ROCm installation: `amd-smi version`
   - Verify consistent installation across nodes
   - Update expected_rocm_version in config

4. **Missing RDMA Interfaces**
   - List interfaces: `ls /sys/class/infiniband/`
   - Update `connectivity_check.rdma.interfaces` to match your cluster setup
   - Ensure all expected interfaces are present on each node

5. **Node-Health Failures**
   - Compare `gpus_per_node` with `amd-smi list`
   - Verify AMDGPU and KFD are loaded and inspect kernel errors in `dmesg`
   - Confirm `expected_rocm_version` matches `amd-smi version`

6. **IFoE Fabric or L2 Failures**
   - Confirm AIFM/AFM services and the in-band node agent are healthy
   - Inspect the HTML report for vPOD, station-mask, down-port, and coverage errors
   - Verify `afmctl` is installed and available through the cluster environment

7. **TransferBench Failures**
   - Review the captured TransferBench output and exit status in the report
   - Confirm all admitted nodes resolve to one consistent vPOD
   - Reduce to `scope: "node"` to isolate a failing host before retrying cluster scope

8. **Node Smoke Failures**
   - Set `node_smoke_tier1.connectivity_mode` to `"run"` (default is `"skip"`)
   - Verify `primus_dir` and `venv_activate`, or enable `auto_setup: true`
   - On shared NFS home, use `shared_install: true` to avoid parallel clone races
   - Match `torch_pip_index_url` to your ROCm version
   - Review per-node fail reasons in the preflight HTML report

9. **Node Smoke Tier 3 Failures**
   - Set `node_smoke_tier3.connectivity_mode` to `"run"` (independent of Tier 1)
   - Ensure NCCL transport env vars are set when validating RDMA/RCCL inventory findings
   - Review `<artifacts_root_dir>/node_smoke_tier3/node_smoke_tier3.md` on the leader node
   - Increase `ssh_timeout` or `dist_timeout_sec` on large or slow clusters

### Performance Considerations

**RDMA Connectivity Testing Times:**
- **Basic mode**: ~30 seconds for 8 nodes
- **Full mesh mode**: ~5-10 minutes for 8 nodes
- **Skip mode**: fastest path when validating only node-local checks

**Node Smoke Testing Times:**
- **First run with auto_setup**: several minutes per node (clone + ROCm PyTorch install)
- **Tier 1 subsequent runs**: ~30–60 seconds per node
- **Tier 2 with tier2_perf**: several minutes per node (GEMM + HBM + local RCCL)
- **Tier 3 cluster run**: ~2–10 minutes depending on cluster size and `ssh_timeout`

**Parallel Processing Impact:**
- **Small nodes_per_full_mesh_group (16-32)**: More rounds, less resource usage per node, better for resource-constrained environments
- **Large nodes_per_full_mesh_group (128+)**: Fewer rounds, more resource usage per node, faster overall completion
- **Debug mode impact**: `scriptlet=true` adds 10-30% overhead due to strace logging

**Advanced Parameter Tuning:**
- **inter_full_mesh_group_pairs_per_wave**: Lower values (2-4) reduce memory/CPU load but increase test time
- **prune_failure_threshold**: Lower values (0.2-0.3) are more aggressive at removing problematic nodes
- **Port retry settings**: Higher retry counts help in congested network environments but increase test time

### Network Requirements

- **Ports**: Ensure ibv_test_port_range is not blocked by firewalls
- **RDMA**: InfiniBand or RoCE interfaces must be active
- **SSH**: Passwordless SSH access to all cluster nodes
- **Privileges**: Some checks may require sudo access for system information

## Integration with Performance Tests

The preflight checks are designed to run before:

1. **IB Performance Tests** (`ib_perf_bw_test`)
2. **RCCL Training Tests** (`rccl_multinode_*`)
3. **Inference Workloads** (PyTorch, JAX, etc.)

A typical workflow:

```bash
# 1. Run preflight checks
cvs run preflight_checks --cluster_file cluster.json --config_file preflight_config.json

# 2. If preflight passes, run performance tests
cvs run ib_perf_bw_test --cluster_file cluster.json --config_file ib_config.json

# 3. Run RCCL training tests
cvs run rccl_multinode_default_cvs --cluster_file cluster.json --config_file rccl_config.json
```

This ensures your cluster is healthy before running resource-intensive performance tests.

Within preflight, the mandatory order is node health and optional scale-up
fabric admission, then l2ping, then TransferBench, then Node Smoke Tier 1/2 and
Tier 3 (when enabled), followed by RDMA checks. Setting
`connectivity_check.rdma.connectivity_mode` to `"skip"` skips RDMA without
disabling the independent IFoE gates or Node Smoke tiers.
