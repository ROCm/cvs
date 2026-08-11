# Preflight Configuration Guide

This document explains how to configure the GPU cluster preflight checks system.

## Overview

The preflight checks system validates essential cluster health before running performance tests like IB performance tests, RCCL training, and inference workloads. It performs the following validations:

1. **Node Health** - Checks GPU visibility, AMDGPU/KFD, kernel health, and ROCm consistency
2. **MI4XX Scale-up Fabric Admission** - Optionally validates AIFM/AFM/vPOD membership, station masks, and IFoE port state
3. **IFoE L2 Connectivity** - Optionally runs strict `afmctl test ping` coverage before TransferBench and RDMA
4. **TransferBench** - Optionally validates the IFoE data path per node or with a multi-rank cluster run
5. **GID and Interface Consistency** - Ensures configured RDMA interfaces and GID entries are present and consistent
6. **RDMA Connectivity** - Tests node-to-node RDMA communication using `ibv_rc_pingpong`

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
        "fabric_checks": true,
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
    "reporting": {
      "generate_html_report": true,
      "artifacts_root_dir": "/tmp/{user-id}/preflight",
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
├── reporting/                 # Output and report generation
└── debug/                     # Debug and troubleshooting options
```

### Execution Flow
1. **node_check** - Validate individual nodes and, when enabled, MI4XX scale-up fabric admission
2. **connectivity_check.ifoe.l2ping** - Enforce strict L2 connectivity
3. **connectivity_check.ifoe.transferbench** - Validate the IFoE data path
4. **connectivity_check.rdma** - Validate RDMA inventory, GIDs, and connectivity unless skipped
5. **reporting** - Generate reports and outputs

## Configuration Parameters

Every parameter -- with its type, real default, constraints and an example --
is documented by `cvs man`:

```bash
cvs man preflight_checks              # every parameter
cvs man preflight_checks gid_index    # a single parameter
cvs man preflight_checks --json       # machine-readable
```

That reference is generated from `PreflightConfigFile` in
`cvs/parsers/schemas.py`, so it cannot drift from the code the way a
hand-written table does. All parameters are optional and have sensible
defaults.

### Important Update: RDMA Connectivity Testing

**As of this version, preflight checks now use `ibv_rc_pingpong` instead of `rping` for RDMA connectivity testing.**

**Why this change?**
- `rping` uses RDMA Connection Manager (`rdma_cm`) which is more forgiving of network issues
- `ibv_rc_pingpong` uses direct InfiniBand verbs (same as RCCL) which is more strict
- This change allows preflight checks to detect the same connectivity issues that cause RCCL failures
- **Result**: Better correlation between preflight results and actual RCCL performance

**Updated parameter names**: Configuration parameters now use accurate names (`ibv_test_timeout`, `ibv_test_port_range`) that reflect the use of `ibv_rc_pingpong` for testing.

### Legacy RDMA paths — deprecated

Existing RDMA users may temporarily retain `node_check.gid_index` and
`node_check.rdma_interfaces`. CVS normalizes them to
`connectivity_check.rdma.gid_index` and `connectivity_check.rdma.interfaces`
and emits a deprecation warning. If a legacy and canonical value are both
present, they must match. New configurations should use the canonical RDMA
paths; the compatibility paths will be removed in a future release.

### IFoE settings that are no longer configurable

IFoE validation (`connectivity_check.ifoe`) used to expose implementation
details -- fabric discovery, privilege handling, port selection, timeouts --
directly as config keys. Those keys are no longer read; CVS now derives all
of this automatically and follows a fixed policy:

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
      "artifacts_root_dir": "/tmp/{user-id}/preflight",
      "generate_rdma_pairs_csv": true
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

Within preflight, the mandatory order is node health and optional scale-up
fabric admission, then l2ping, then TransferBench, followed by RDMA checks.
Setting `connectivity_check.rdma.connectivity_mode` to `"skip"` skips RDMA
without disabling the independent IFoE gates.
