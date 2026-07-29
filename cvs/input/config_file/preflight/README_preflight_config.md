# Preflight Configuration Guide

This document explains how to configure the GPU cluster preflight checks system.

## Overview

The preflight checks system validates essential cluster health before running performance tests like IB performance tests, RCCL training, and inference workloads. It performs the following validations:

1. **GID Consistency** - Ensures RDMA interfaces have valid GID entries
2. **RDMA Connectivity** - Tests node-to-node RDMA communication using ibv_rc_pingpong
3. **ROCm Version Consistency** - Verifies consistent ROCm versions across nodes
4. **Interface Name Consistency** - Validates RDMA interface naming patterns
5. **Node-Health Admission (mandatory when enabled)** - Always validates
   AMDGPU/KFD, GPU visibility, and current-boot kernel health. With
   `fabric_checks: true`, it also validates the MI4XX AIFM/AFM/vPOD and IFoE
   station/port state.
6. **IFoE L2 Connectivity (AIMVT-180; opt-in)** - Runs `afmctl test ping`
   on each node and enforces per-port and Summary pass/fail accounting
7. **IFoE TransferBench Smoketest (AIMVT-181; opt-in)** - Runs the
   TransferBench candidate-branch `smoketest` preset to validate IFoE
   scale-up data-path one layer above L2 (using the MI4XX AFM admission
   result, not `amd-smi fabric --topology`, when fabric checks are enabled)

## Configuration File Structure

The preflight configuration file follows this structure:

```json
{
  "preflight": {
    "node_health": {
      "enabled": true,
      "gpus_per_node": 4,
      "fabric_checks": true
    },
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
    },
    "debug": {
      "scriptlet": false
    },
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "basic",
        "nodes_per_full_mesh_group": 128,
        "ibv_test_timeout": 90,
        "ibv_test_port_range": "10000-50000",
        "inter_full_mesh_group_pairs_per_wave": "auto"
      }
    },
    "reporting": {
      "generate_html_report": "true",
      "artifacts_root_dir": "/tmp/{user-id}/preflight",
      "generate_rdma_pairs_csv": "true"
    }
  }
}
```

## Configuration Structure

The preflight configuration uses a **nested structure organized by execution phase** for better organization and future extensibility:

### Structure Overview

```
preflight/
├── debug/                # Debug and troubleshooting options  
├── node_health/          # Generic GPU health plus optional MI4XX fabric admission
├── l2ping/               # Strict IFoE L2 connectivity admission
├── transferbench/        # IFoE data-path validation admission
├── node_check/           # Individual node validation parameters
├── connectivity_check/   # Inter-node connectivity tests
│   └── rdma/             # RDMA-specific parameters (including nodes_per_full_mesh_group)
└── reporting/           # Output and report generation
```

### Execution Flow
1. **node_health** - Validate GPU health after SSH reachability, optionally including MI4XX fabric admission
2. **Node-health admission** - When enabled, block IFoE, TransferBench, and RDMA checks until every declared node is admitted
3. **connectivity_check.rdma.nodes_per_full_mesh_group** - Configure RDMA batching resources
4. **connectivity_check** - Test inter-node connectivity by protocol
5. **reporting** - Generate reports and outputs

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

- **`gid_index`** (default: "3")
  - GID index to check on all RDMA interfaces
  - Typically "3" for RoCE (RDMA over Converged Ethernet)
  - Must be a valid GID index for your InfiniBand/RoCE setup

- **`expected_rocm_version`** (default: "6.2.0")
  - Expected ROCm version across all cluster nodes
  - Must match the output of `amd-smi version` on all nodes
  - Format: "major.minor.patch" (e.g., "6.2.0", "5.7.1")

- **`rdma_interfaces`** (default: ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"])
  - List of specific RDMA interface names that should be present on all cluster nodes
  - Examples:
    - `["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]` - Standard 4-interface setup
    - `["mlx5_0", "mlx5_1"]` - Mellanox 2-interface setup
    - `["ib0", "ib1", "ib2", "ib3"]` - Generic InfiniBand setup

### Connectivity Check Settings (`connectivity_check`)

#### RDMA Settings (`connectivity_check.rdma`)

- **`connectivity_mode`** (default: "basic")
  - **"basic"**: Test adjacent node pairs (fast, ~14% coverage for 8 nodes)
  - **"full_mesh"**: Test all possible node pairs (comprehensive, 100% coverage)
  - **"skip"**: Skip RDMA connectivity testing entirely

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

#### Node-Health Admission (`node_health`)

Node health has three customer-facing options:

```json
"node_health": {
  "enabled": true,
  "gpus_per_node": 4,
  "fabric_checks": true
}
```

- **`enabled`** runs the mandatory read-only node-health gate. A failure is
  recorded in the report and blocks downstream IFoE, TransferBench, and RDMA
  checks. CVS never attempts platform recovery.
- **`gpus_per_node`** is the GPU count required from `amd-smi list --json` on
  every node.
- **`fabric_checks`** enables the complete MI4XX fabric bundle: the IFoE
  module, AIFM agent, one AFM device per expected GPU, AFM `ACTIVE` bare-metal
  vPOD state, 36-port device inventory, station masks, and admitted UP ports.
  When false, only generation-independent AMDGPU/KFD, GPU visibility, and
  current-boot kernel-health checks run.

Tool names, JSON transport, privileges, readiness polling, AFM counts, and
station policy are CVS-owned implementation details rather than customer
configuration.
- **Rendezvous/firewall coverage**
  - CVS does not add a generic port scanner. Keep the existing RDMA
    `full_mesh` test enabled with an `ibv_test_port_range` that is permitted
    between all nodes; that directly validates the TCP/RDMA path used by
    RCCL-style scale-up rendezvous. TransferBench `scope: "cluster"` likewise
    requires CVS-owned TCP port `31337` between all selected nodes.

#### L2 Ping Settings (`l2ping`) — opt-in (AIMVT-180)

Runs `afmctl test ping` on every reachable node as a strict IFoE fabric gate.
CVS discovers each node's AFM BDFs, vPOD peers, and UP ports, then tests every
ordered non-self accelerator pair. When node-health fabric admission is
available, physical UP ports are intersected with its station-mask-admitted
ports.

- **`enabled`** (default: `false`)
  - Run l2ping before TransferBench and RDMA.
  - When enabled, packet loss, incomplete coverage, malformed discovery, or a
    missing required node fails the preflight gate.
- **`pings_per_port`** (default: `3`)
  - Number of samples sent through each selected port pair.
  - Higher values increase confidence and runtime.

CVS owns the command path, JSON transport, full-mesh topology discovery,
UP-port selection, three traffic types, zero-loss policy, strict coverage,
timeouts, and gate behavior. These are intentionally not customer options.

#### TransferBench Settings (`transferbench`) — opt-in (AIMVT-181)

Runs the TransferBench candidate-branch **`smoketest`** preset on every reachable
node to validate IFoE scale-up data-path connectivity (one layer above the
AIMVT-180 L2 ping). With node-health fabric checks enabled, it consumes the mandatory
`afmctl show device --json` admission result and never queries unreliable
`amd-smi fabric --topology`. Generic non-MI4XX profiles retain the `amd-smi`
precondition. The binary's exit code is reconciled with per-cell
`[PASS]/[FAIL]/[SKIP]` markers. A failed enabled run is a mandatory preflight
gate; skip-budget warnings remain non-fatal.

- **`enabled`** (default: `false`) — run the TransferBench gate.
- **`scope`** (default: `"node"`) — `"node"` runs independently on every
  node; `"cluster"` runs one multi-rank job across all reachable nodes.
- **`profile`** (default: `"smoketest"`) — CVS-supported validation profile.
  Only `smoketest` is supported in this iteration because its output and
  verdict contract are explicitly parsed by CVS.
- **`message_sizes`** (default: `["1K", "16M"]`) — message sizes exercised
  by every smoketest cell.
- **`iterations`** (default: `2`) — validated iterations per test and size.
- **`warmup_iterations`** (default: `0`) — warmups before validated work.

CVS owns executable discovery, privilege handling, runtime environment,
validation and parallelism flags, single-vPOD enforcement, deterministic
master selection, skip policy, and timeout derivation. Cluster scope uses the
fixed coordination port `31337`, which must be open between selected nodes.

### Reporting Settings (`reporting`)

- **`generate_html_report`** (default: "true")
  - Whether to generate detailed HTML report
  - Set to "false" to disable HTML report generation

- **`artifacts_root_dir`** (default: "/tmp/{user-id}/preflight")
  - Root directory where preflight artifacts are saved
  - Includes HTML reports and RDMA full_mesh workspace logs under `rdma_connectivity_workspace/`
  - Must be writable by the user running the tests

- **`generate_rdma_pairs_csv`** (default: "true")
  - Whether to generate CSV file with failed RDMA pairs alongside HTML report
  - Set to "false" to disable CSV generation

## Usage Examples

### Basic 8-Node Cluster Check

```json
{
  "preflight": {
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "basic",
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
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "full_mesh",
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
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "skip"
      }
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
      "gid_index": "3",
      "expected_rocm_version": "7.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0", "rocep158s0", "rocep190s0", "rocep206s0", "rocep222s0"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "full_mesh",
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
      "generate_html_report": "true",
      "artifacts_root_dir": "/tmp/{user-id}/preflight",
      "generate_rdma_pairs_csv": "true"
    }
  }
}
```

## Running Preflight Checks

```bash
# Basic usage with default config
cvs run preflight_checks --cluster_file cluster.json --config_file preflight_config.json

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
   - Update rdma_interfaces list to match your cluster setup
   - Ensure all expected interfaces are present on each node

### Performance Considerations

**RDMA Connectivity Testing Times:**
- **Basic mode**: ~30 seconds for 8 nodes
- **Full mesh mode**: ~5-10 minutes for 8 nodes
- **Skip mode**: fastest path when validating only node-local checks

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
