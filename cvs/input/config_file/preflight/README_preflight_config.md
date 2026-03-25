# Preflight Configuration Guide

This document explains how to configure the GPU cluster preflight checks system.

## Overview

The preflight checks system validates essential cluster health before running performance tests like IB performance tests, RCCL training, and inference workloads. It performs four key validations:

1. **GID Consistency** - Ensures RDMA interfaces have valid GID entries
2. **RDMA Connectivity** - Tests node-to-node RDMA communication using rping
3. **ROCm Version Consistency** - Verifies consistent ROCm versions across nodes
4. **Interface Name Consistency** - Validates RDMA interface naming patterns

## Configuration File Structure

The preflight configuration file follows this structure:

```json
{
  "preflight": {
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "basic",
    "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
    "rping_timeout": "10",
    "rping_port_range": "9000-9999",
    "generate_html_report": "true",
    "report_output_dir": "/tmp/preflight_reports"
  }
}
```

## Configuration Parameters

### Important Update: RDMA Connectivity Testing

**As of this version, preflight checks now use `ibv_rc_pingpong` instead of `rping` for RDMA connectivity testing.**

**Why this change?**
- `rping` uses RDMA Connection Manager (`rdma_cm`) which is more forgiving of network issues
- `ibv_rc_pingpong` uses direct InfiniBand verbs (same as RCCL) which is more strict
- This change allows preflight checks to detect the same connectivity issues that cause RCCL failures
- **Result**: Better correlation between preflight results and actual RCCL performance

**Backward compatibility**: Configuration parameter names remain the same (`rping_timeout`, `rping_port_range`) but now apply to `ibv_rc_pingpong`.

### GID Configuration

- **`gid_index`** (default: "3")
  - GID index to check on all RDMA interfaces
  - Typically "3" for RoCE (RDMA over Converged Ethernet)
  - Must be a valid GID index for your InfiniBand/RoCE setup

### ROCm Version

- **`expected_rocm_version`** (default: "6.2.0")
  - Expected ROCm version across all cluster nodes
  - Must match the output of `amd-smi version` on all nodes
  - Format: "major.minor.patch" (e.g., "6.2.0", "5.7.1")

### RDMA Connectivity Testing

- **`rdma_connectivity_check`** (default: "basic")
  - **"basic"**: Test adjacent node pairs (fast, ~14% coverage for 8 nodes)
  - **"full_mesh"**: Test all possible node pairs (comprehensive, 100% coverage)
  - **"sample"**: Test random 20% of node pairs (balanced speed/coverage)
  - **"skip"**: Skip RDMA connectivity testing entirely

- **`rping_timeout`** (default: "10")
  - Timeout in seconds for each ibv_rc_pingpong connectivity test
  - **Note**: Now uses `ibv_rc_pingpong` (direct IB verbs) instead of `rping` for RCCL-compatible testing
  - Increase for slower networks or high-latency connections

- **`rping_port_range`** (default: "9000-9999")
  - Port range for ibv_rc_pingpong tests to avoid conflicts
  - **Note**: Parameter name kept for backward compatibility, but now applies to `ibv_rc_pingpong`
  - Format: "start-end" (e.g., "9000-9999", "10000-10999")
  - Ensure ports are not blocked by firewalls

### RDMA Interfaces

- **`rdma_interfaces`** (default: ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"])
  - List of specific RDMA interface names that should be present on all cluster nodes
  - Examples:
    - `["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]` - Standard 4-interface setup
    - `["mlx5_0", "mlx5_1"]` - Mellanox 2-interface setup
    - `["ib0", "ib1", "ib2", "ib3"]` - Generic InfiniBand setup

### Reporting

- **`generate_html_report`** (default: "true")
  - Whether to generate detailed HTML report
  - Set to "false" to disable HTML report generation

- **`report_output_dir`** (default: "/tmp/preflight_reports")
  - Directory where HTML reports will be saved
  - Must be writable by the user running the tests

## Usage Examples

### Basic 8-Node Cluster Check

```json
{
  "preflight": {
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "basic",
    "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
    "rping_timeout": "10",
    "rping_port_range": "9000-9999"
  }
}
```

### Comprehensive Full-Mesh Testing

```json
{
  "preflight": {
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "full_mesh",
    "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
    "rping_timeout": "15",
    "rping_port_range": "9000-9999"
  }
}
```

### Quick Sample-Based Validation

```json
{
  "preflight": {
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "sample",
    "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
    "rping_timeout": "5",
    "rping_port_range": "9000-9999"
  }
}
```

### Configuration-Only Validation (Skip Connectivity)

```json
{
  "preflight": {
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "skip",
    "expected_interface_pattern": "rocep*s0"
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
   - Verify rping is available: `which rping`
   - Test manual connectivity: `rping -s -p 9000` and `rping -c -a <node> -p 9000`

3. **ROCm Version Mismatches**
   - Check ROCm installation: `amd-smi version`
   - Verify consistent installation across nodes
   - Update expected_rocm_version in config

4. **Missing RDMA Interfaces**
   - List interfaces: `ls /sys/class/infiniband/`
   - Update rdma_interfaces list to match your cluster setup
   - Ensure all expected interfaces are present on each node

### Performance Considerations

- **Basic mode**: ~30 seconds for 8 nodes
- **Sample mode**: ~1-2 minutes for 8 nodes  
- **Full mesh mode**: ~5-10 minutes for 8 nodes
- **Large clusters**: Consider sample mode for initial validation, then full mesh for critical paths

### Network Requirements

- **Ports**: Ensure rping_port_range is not blocked by firewalls
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