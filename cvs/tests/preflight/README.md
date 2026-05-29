# GPU Cluster Preflight Checks

A comprehensive validation system for GPU clusters before running performance tests, training workloads, and inference tasks.

## Overview

The preflight checks system validates essential cluster health and configuration consistency across all nodes. It performs the following validations:

1. **GID Consistency** - Ensures RDMA interfaces have valid Global Identifier entries
2. **RDMA Interface Presence** - Validates that expected RDMA interfaces are present and link-up
3. **ROCm Version Consistency** - Verifies consistent ROCm versions across all nodes
4. **IFoE L2 Connectivity** - Validates L2 reachability of IFoE links via `afmctl test ping` *(AIMVT-180; opt-in)*
5. **IFoE TransferBench Smoketest** - Runs the TransferBench candidate-branch `smoketest`
   preset to validate IFoE scale-up data-path (after a single-vPod precondition via
   `amd-smi fabric --topology --json`) *(AIMVT-181; opt-in)*
6. **RDMA Connectivity** - Tests node-to-node RDMA communication using `ibv_rc_pingpong`

## Quick Start

### Basic Usage

```bash
# Run preflight checks with default configuration
cvs run preflight_checks \
  --cluster_file cluster.json \
  --config_file cvs/input/config_file/preflight/preflight_config.json
```

### With Custom HTML Report

```bash
# Generate HTML report in specific location
cvs run preflight_checks \
  --cluster_file cluster.json \
  --config_file preflight_config.json \
  --html /path/to/preflight_report.html \
  --self-contained-html
```

## Test Modes

### Basic Mode (Default)
- Tests adjacent node pairs (like current IB performance tests)
- Fast execution (~30 seconds for 8 nodes)
- 14.3% coverage for 8-node cluster (4 out of 28 possible pairs)
- Good for quick validation

### Full Mesh Mode
- Tests all possible node pair combinations
- Comprehensive coverage (100% of all pairs)
- Longer execution (~5-10 minutes for 8 nodes)
- Uses batched approach to maximize parallelism
- Recommended for thorough validation

### Sample Mode
- Tests random 20% of all possible node pairs
- Balanced speed vs coverage (~1-2 minutes for 8 nodes)
- Good for regular health checks

## Configuration

### Default Configuration File
Located at: `cvs/input/config_file/preflight/preflight_config.json`

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
        "ibv_test_timeout": "10",
        "ibv_test_port_range": "10000-50000"
      }
    },
    "reporting": {
      "generate_html_report": "true",
      "artifacts_root_dir": "/tmp/preflight"
    }
  }
}
```

### Key Parameters

- **`connectivity_check.rdma.connectivity_mode`**: `"basic"`, `"full_mesh"`, or `"skip"`
- **`connectivity_check.ifoe.connectivity_mode`**: `"run"` or `"skip"` (default)
- **`connectivity_check.transferbench.connectivity_mode`**: `"run"` or `"skip"` (default)
- **`node_check.expected_rocm_version`**: ROCm version expected across all nodes
- **`node_check.rdma_interfaces`**: List of expected RDMA interface names
- **`connectivity_check.rdma.ibv_test_timeout`**: Timeout in seconds for ibv_rc_pingpong tests
- **`connectivity_check.rdma.ibv_test_port_range`**: Port range for parallel ibv_rc_pingpong tests

## IFoE L2 Connectivity (AIMVT-180)

Validates L2 reachability of IFoE links by invoking `afmctl test ping` on
each reachable node and parsing the per-port pass/fail counts and the
aggregate `Summary:` section from `afmctl`'s output.

Each invocation issues:

```
<afmctl_path> test ping -b <bdf> -c <pings_per_port> [-p <ports>] \
    --dst-accelerator <accel_id> [-t <per_ping_timeout>] [--traffic-type ...]
```

The check is **opt-in**: it defaults to `connectivity_mode: "skip"` so it
will not run unless explicitly enabled. Enable it by setting
`preflight.connectivity_check.ifoe.connectivity_mode` to `"run"` once the
IFoE driver and `afmctl` are available on every node.

When `bdfs` is empty and `bdf_discovery` is `"auto"` (default), preflight
runs `afmctl show device` on every reachable node and uses the BDFs it
reports. A node is marked **FAIL** if any enabled traffic type
(`ifoe_req`, `ifoe_resp`, `non_ifoe`) exceeds `loss_threshold_pct` (default
`0.0`), or if any per-port table entry reports `FAIL`.

### Example IFoE config block

```json
"connectivity_check": {
  "ifoe": {
    "connectivity_mode": "run",
    "afmctl_path": "/usr/local/bin/afmctl",
    "use_sudo": true,
    "bdf_discovery": "auto",
    "dst_accelerators": [0, 1],
    "ports": "all",
    "pings_per_port": 5,
    "traffic_types": ["ifoe_req", "ifoe_resp", "non_ifoe"],
    "loss_threshold_pct": 0.0,
    "ssh_timeout": 240
  }
}
```

## IFoE TransferBench Smoketest (AIMVT-181)

Validates IFoE scale-up data-path one layer above L2 by invoking the
TransferBench candidate-branch **`smoketest`** preset on every reachable
node and reconciling the binary's exit code with per-cell
`[PASS] / [FAIL] / [SKIP]` markers in its output.

Before TransferBench runs the check enforces a single-vPod precondition by
querying:

```
sudo amd-smi fabric --topology --json
```

on every reachable node and verifying that every node reports exactly one
`vpod_id` and all nodes share the same `vpod_id`. The TransferBench
candidate-branch smoketest preset exits with `ERR_FATAL` (exit code `2`)
when ranks span multiple virtual pods, so this precondition lets us surface
the cause clearly rather than blaming the smoketest for an environment
issue.

Each TransferBench invocation issues:

```
[sudo] [PATH=<rocm>/bin:... LD_LIBRARY_PATH=<rocm>/lib:...] \
  NUM_ITERATIONS=<n> NUM_WARMUPS=<n> ALWAYS_VALIDATE=1 RUN_PARALLEL=1 \
  [TB_NUM_RANKS=<n> TB_RANK=<r> TB_MASTER_ADDR=<rank0> TB_MASTER_PORT=<port>] \
  <tb_binary> smoketest <size_list...>
```

with a `__TB_SMOKE_EXIT__=$?` sentinel appended so we can recover the
binary's exit code from stdout even when the parallel SSH transport
discards process exit codes.

### Orchestration modes

- **`per_node`** (default) — one independent single-rank TransferBench per
  reachable node. Exercises intra-node AID↔MID IFoE only, but needs no
  cross-node coordination and is the safest first run when bringing up a
  cluster.
- **`multi_rank`** — every reachable node is wired into one socket-comm
  cluster (`TB_NUM_RANKS=N`, `TB_RANK=0..N-1`, `TB_MASTER_ADDR=<rank0>`).
  This is the closest thing to a full-mesh IFoE scale-up test the candidate
  branch ships today; it traverses the rack IFoE switch end-to-end. Requires
  bidirectional IP reachability on `socket_master_port` between every pair
  of nodes, and at least two reachable hosts (otherwise we degrade to
  `per_node` automatically and log a warning).

### Verdict logic

Per-node verdict is derived as:

1. No `__TB_SMOKE_EXIT__` sentinel observed → **FAIL** (orchestration broke)
2. Exit code `2` → **FAIL** (TransferBench precondition fired; usually a
   pod-membership or executor-symmetry issue inside the preset)
3. Any non-zero exit code → **FAIL**
4. Any parsed `FAIL` marker or fatal-keyword line in stdout → **FAIL**
5. `num_skip / num_tests > max_skip_pct%` → **WARNING**
6. Otherwise → **PASS**

### Example TransferBench config block

```json
"connectivity_check": {
  "transferbench": {
    "connectivity_mode": "run",
    "tb_binary": "/opt/rocm/bin/TransferBench",
    "rocm_path": "/opt/rocm",
    "amd_smi_path": "amd-smi",
    "use_sudo": true,
    "preset": "smoketest",
    "size_list": ["1K", "16M"],
    "num_iterations": 2,
    "num_warmups": 0,
    "always_validate": true,
    "run_parallel": true,
    "force_single_pod": true,
    "rank_mode": "per_node",
    "max_skip_pct": 25.0,
    "ssh_timeout": 600
  }
}
```

To exercise the rack IFoE switch end-to-end, flip `rank_mode` to
`"multi_rank"` and open `socket_master_port` (default `31337`) in the
firewall between every node pair.

## Output and Reporting

### Console Output
```
✅ GID Consistency: PASS (64/64 interfaces have GID index 3)
⚪ RDMA Connectivity: SKIPPED (Test skipped by configuration)  
✅ ROCm Versions: PASS (All nodes running 6.2.0)
✅ Interface Names: PASS (All interfaces match rocep*s0 pattern)

Overall Status: PASS - Cluster ready for performance testing (connectivity not tested)
```

### HTML Report
Comprehensive HTML report includes:
- Executive summary with pass/fail status
- Detailed per-node results for each check
- RDMA connectivity matrix showing all tested pairs
- Configuration details and recommendations
- Error details for failed checks

## Integration with Performance Tests

### Recommended Workflow

```bash
# 1. Run preflight checks first
cvs run preflight_checks \
  --cluster_file cluster.json \
  --config_file preflight_config.json

# 2. If preflight passes, run IB performance tests
cvs run ib_perf_bw_test \
  --cluster_file cluster.json \
  --config_file ib_config.json

# 3. Run RCCL training tests
cvs run rccl_multinode_default_cvs \
  --cluster_file cluster.json \
  --config_file rccl_config.json

# 4. Run inference workloads
cvs run pytorch_xdit_wan \
  --cluster_file cluster.json \
  --config_file inference_config.json
```

### Benefits of Preflight Validation

- **Early Problem Detection**: Catch configuration issues before expensive performance tests
- **Time Savings**: Avoid running long performance tests on misconfigured clusters
- **Clear Diagnostics**: Detailed error reporting for quick issue resolution
- **Comprehensive Coverage**: Validates all critical cluster health aspects

## Architecture

### Test Execution Flow

```
1. Load and validate cluster + preflight configurations
2. Test SSH connectivity to all nodes
3. Run parallel preflight checks:
   - GID consistency across all RDMA interfaces
   - RDMA connectivity using rping (mode-dependent)
   - ROCm version consistency via amd-smi
   - Interface naming pattern compliance
4. Generate comprehensive summary and HTML report
5. Return overall PASS/FAIL status
```

### Parallel Execution

- **SSH Operations**: Parallel execution across all cluster nodes
- **RDMA Connectivity**: Batched parallel testing to maximize throughput
- **Error Collection**: Continues all checks even if some fail (collect-all mode)

## Troubleshooting

### Common Issues

#### GID Check Failures
```bash
# Check RDMA driver status
lsmod | grep rdma

# Verify interface status  
rdma link show

# Check GID entries manually
cat /sys/class/infiniband/*/ports/1/gids/3
```

#### RDMA Connectivity Failures
```bash
# Verify rping availability
which rping

# Check firewall status
sudo ufw status

# Test manual connectivity
rping -s -p 9000  # Server
rping -c -a <node> -p 9000  # Client
```

#### ROCm Version Mismatches
```bash
# Check ROCm version on each node
amd-smi version

# Verify consistent installation
ls -la /opt/rocm/
```

#### Interface Name Inconsistencies
```bash
# List RDMA interfaces
ls /sys/class/infiniband/

# Check interface details
ibv_devinfo
```

### Performance Considerations

| Mode | 8 Nodes | 16 Nodes | 32 Nodes |
|------|---------|----------|----------|
| Basic | ~30s | ~45s | ~60s |
| Full Mesh | ~5-10min | ~20-30min | ~2-3hrs |
| Skip | ~5s | ~5s | ~5s |

### Network Requirements

- **SSH Access**: Passwordless SSH to all cluster nodes
- **RDMA Interfaces**: Active InfiniBand or RoCE interfaces
- **Port Access**: ibv_test_port_range must not be blocked by firewalls
- **Privileges**: Some checks may require sudo access

## Files and Structure

```
cvs/cvs/tests/preflight/
├── __init__.py
├── preflight_checks.py          # Main test module
└── README.md                    # This file

cvs/cvs/lib/
└── preflight_lib.py             # Core preflight functions

cvs/cvs/input/config_file/preflight/
├── preflight_config.json        # Default configuration
└── README_preflight_config.md   # Configuration guide
```

## Advanced Usage

### Custom Configuration

```json
{
  "preflight": {
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["mlx5_0", "mlx5_1"]
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "full_mesh",
        "ibv_test_timeout": "15",
        "ibv_test_port_range": "10000-10999"
      }
    },
    "reporting": {
      "generate_html_report": "true",
      "artifacts_root_dir": "/shared/preflight_reports"
    }
  }
}
```

### Scripted Validation

```bash
#!/bin/bash
# Automated cluster validation script

CLUSTER_FILE="production_cluster.json"
PREFLIGHT_CONFIG="production_preflight.json"

echo "Running preflight checks..."
if cvs run preflight_checks \
  --cluster_file "$CLUSTER_FILE" \
  --config_file "$PREFLIGHT_CONFIG"; then
  
  echo "✅ Preflight checks passed - proceeding with performance tests"
  
  # Run performance tests
  cvs run ib_perf_bw_test --cluster_file "$CLUSTER_FILE" --config_file ib_config.json
  cvs run rccl_multinode_default_cvs --cluster_file "$CLUSTER_FILE" --config_file rccl_config.json
  
else
  echo "❌ Preflight checks failed - fix issues before running performance tests"
  exit 1
fi
```

### Large Cluster Considerations

For clusters with 100+ nodes:
- Use `"basic"` mode for regular health checks
- Use `"full_mesh"` mode only for critical validation
- Consider hierarchical testing (rack-by-rack)
- Increase `ibv_test_timeout` for high-latency networks
- Monitor network bandwidth during full mesh tests

## Contributing

When adding new preflight checks:

1. Add the check function to `preflight_lib.py`
2. Add the test function to `preflight_checks.py`
3. Update the HTML report generation
4. Add configuration parameters if needed
5. Update documentation

### Example: Adding a New Check

```python
# In preflight_lib.py
def check_gpu_health(phdl):
    """Check GPU health across all nodes."""
    cmd = "rocm-smi --showtemp --showpower"
    # Implementation here
    return results

# In preflight_checks.py  
def test_gpu_health(phdl, config_dict):
    """Test GPU health across cluster nodes."""
    results = preflight_lib.check_gpu_health(phdl)
    # Validation and reporting here
```

This system provides a solid foundation for ensuring cluster health before running critical workloads.