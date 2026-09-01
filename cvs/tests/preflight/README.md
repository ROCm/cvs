# GPU Cluster Preflight Checks

A comprehensive validation system for GPU clusters before running performance tests, training workloads, and inference tasks.

## Overview

The preflight checks system validates essential cluster health and configuration consistency across all nodes. It performs the following validations:

1. **Node Health** - Validates AMDGPU/KFD, GPU visibility, and kernel health, with optional MI4XX fabric admission
2. **GID Consistency** - Ensures RDMA interfaces have valid Global Identifier entries
3. **RDMA Interface Presence** - Validates that expected RDMA interfaces are present and link-up
4. **ROCm Version Consistency** - Verifies consistent ROCm versions across all nodes
5. **IFoE L2 Connectivity** - Validates L2 reachability of IFoE links via `afmctl test ping` *(opt-in)*
6. **IFoE TransferBench Smoketest** - Runs the TransferBench candidate-branch `smoketest`
   preset to validate the IFoE scale-up data path (using MI4XX AFM admission or,
   for generic profiles, an `amd-smi fabric --json` single-vPod precondition)
   *(AIMVT-181; opt-in)*
7. **Node Smoke Tier 1 (opt-in)** - Per-node GPU/RDMA operational roll-call via Primus `node_smoke`
8. **Node Smoke Tier 2 (optional)** - Per-node perf sanity when `tier2_perf` is enabled (GEMM, HBM, local RCCL)
9. **Node Smoke Tier 3 (opt-in)** - Cluster-wide Host/GPU/Network inventory via Primus `preflight --host --gpu --network`
10. **RDMA Connectivity** - Tests node-to-node RDMA communication using `ibv_rc_pingpong`

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
      "enabled": true,
      "gpus_per_node": 4,
      "expected_rocm_version": "7.15.0"
    },
    "connectivity_check": {
      "rdma": {
        "connectivity_mode": "skip"
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
    "reporting": {
      "generate_html_report": true,
      "artifacts_root_dir": "/home/{user-id}/preflight",
      "generate_rdma_pairs_csv": false
    }
  }
}
```

### Key Parameters

- **`node_check.enabled`**: Run GPU node-health and ROCm validation
- **`node_check.gpus_per_node`**: Exact GPU count expected on every node
- **`connectivity_check.ifoe.fabric_checks`**: Opt in to MI4XX AIFM/AFM/vPOD and IFoE station/port checks; disabled by default
- **`connectivity_check.rdma.connectivity_mode`**: `"basic"`, `"full_mesh"`, or `"skip"`
- **`connectivity_check.ifoe.l2ping.enabled`**: Run the strict IFoE L2 connectivity gate
- **`connectivity_check.ifoe.l2ping.pings_per_port`**: Samples per discovered UP port pair
- **`connectivity_check.ifoe.transferbench.enabled`**: Run the TransferBench data-path gate
- **`connectivity_check.ifoe.transferbench.scope`**: `"node"` or `"cluster"`
- **`connectivity_check.ifoe.transferbench.profile`**: CVS-supported profile (`"smoketest"`)
- **`connectivity_check.ifoe.transferbench.message_sizes`**, **`iterations`**, **`warmup_iterations`**: Workload intensity
- **`node_check.expected_rocm_version`**: ROCm version expected across all nodes
- **`connectivity_check.rdma.interfaces`**: List of expected RDMA interface names
- **`connectivity_check.rdma.gid_index`**: GID index validated on those interfaces
- **`connectivity_check.rdma.ibv_test_timeout`**: Timeout in seconds for ibv_rc_pingpong tests
- **`connectivity_check.rdma.ibv_test_port_range`**: Port range for parallel ibv_rc_pingpong tests

Legacy RDMA configurations may temporarily use `node_check.gid_index` and
`node_check.rdma_interfaces`. CVS maps them to
`connectivity_check.rdma.gid_index` and `connectivity_check.rdma.interfaces`
and emits a deprecation warning. The legacy paths will be removed in a future
release. If legacy and canonical values are both supplied, they must match.
New configurations should use only the canonical paths.

## IFoE L2 Connectivity

Validates L2 reachability of IFoE links by invoking `afmctl test ping` on
each reachable node and parsing the per-port pass/fail counts and the
aggregate `Summary:` section from `afmctl`'s output.

Each invocation issues:

```
<afmctl_path> test ping -b <bdf> -c <pings_per_port> [-p <ports>] \
    --dst-accelerator <accel_id> [-t <per_ping_timeout>] [--traffic-type ...]
```

The check is opt-in through `preflight.connectivity_check.ifoe.l2ping.enabled`. When enabled, CVS
discovers each node's source BDFs, vPOD peers, and operational ports, then
tests every ordered non-self accelerator pair. It validates IFoE request,
IFoE response, and non-IFoE traffic with a zero-loss policy and requires
complete coverage. Discovery or connectivity failures fail the preflight gate.

### Example IFoE config block

```json
"ifoe": {
  "l2ping": {
    "enabled": true,
    "pings_per_port": 3
  }
}
```

## IFoE TransferBench Smoketest (AIMVT-181)

Validates IFoE scale-up data-path one layer above L2 by invoking the
TransferBench candidate-branch **`smoketest`** preset on every reachable
node and reconciling the binary's exit code with per-cell
`[PASS] / [FAIL] / [SKIP]` markers in its output.

With node-health `fabric_checks` enabled, TransferBench consumes its mandatory
`afmctl show device --json` vPOD admission and does not run a second AMD SMI
topology query. Generic profiles enforce the single-vPod precondition with
the CVS-managed `amd-smi` command:

```
sudo <resolved-amd-smi-binary> fabric --json
```

on every reachable node and verifying that every node reports exactly one
`vpod_id` and all nodes share the same `vpod_id`. The TransferBench
candidate-branch smoketest preset exits with `ERR_FATAL` (exit code `2`)
when ranks span multiple virtual pods, so this precondition lets us surface
the cause clearly rather than blaming the smoketest for an environment
issue.

Each TransferBench invocation issues:

```
[sudo] SIZE_LIST=<size1>,<size2> NUM_ITERATIONS=<n> NUM_WARMUPS=<n> ALWAYS_VALIDATE=1 RUN_PARALLEL=1 \
  [TB_NUM_RANKS=<n> TB_RANK=<r> TB_MASTER_ADDR=<rank0> TB_MASTER_PORT=<port>] \
  <resolved-tb_binary> smoketest
```

with a `__TB_SMOKE_EXIT__=$?` sentinel appended so we can recover the
binary's exit code from stdout even when the parallel SSH transport
discards process exit codes.

CVS owns executable resolution, privilege handling, environment setup,
validation flags, skip policy, and timeout derivation. The customer selects
the workload through `profile`, `message_sizes`, `iterations`, and
`warmup_iterations`.

### Scopes

- **`node`** (default) — one independent single-rank TransferBench per
  reachable node, exercising intra-node AID↔MID IFoE.
- **`cluster`** — every reachable node is wired into one socket-comm cluster
  (`TB_NUM_RANKS=N`, `TB_RANK=0..N-1`, `TB_MASTER_ADDR=<rank0>`).
  This is the closest thing to a full-mesh IFoE scale-up test the candidate
  branch ships today; it traverses the rack IFoE switch end-to-end. Requires
  bidirectional IP reachability on CVS-owned port `31337` between every pair
  of nodes, and at least two reachable hosts (otherwise we degrade to
  node scope automatically and log a warning).

### Verdict logic

Per-node verdict is derived as:

1. No `__TB_SMOKE_EXIT__` sentinel observed → **FAIL** (orchestration broke)
2. Exit code `2` → **FAIL** (TransferBench precondition fired; usually a
   pod-membership or executor-symmetry issue inside the preset)
3. Any non-zero exit code → **FAIL**
4. Any parsed `FAIL` marker or fatal-keyword line in stdout → **FAIL**
5. More than the CVS-owned skip budget → **WARNING**
6. Otherwise → **PASS**

### Example TransferBench config block

```json
"ifoe": {
  "transferbench": {
    "enabled": true,
    "scope": "node",
    "profile": "smoketest",
    "message_sizes": ["1K", "16M"],
    "iterations": 2,
    "warmup_iterations": 0
  }
}
```

To exercise the rack IFoE switch end-to-end, set `scope` to `"cluster"` and
allow TCP port `31337` between every selected node.

## Node Smoke tiers (Primus)

Node Smoke checks are **opt-in** and configured under `node_smoke_tier1` and
`node_smoke_tier3` in the preflight config. Legacy keys `node_smoke` and
`tier3_info` are still accepted.

| Tier | Config | Scope | Default count (8 GPU) |
|------|--------|-------|------------------------|
| **Tier 1** | `node_smoke_tier1.connectivity_mode: "run"` | Per node | 39 tests run per node |
| **Tier 2** | `node_smoke_tier1.tier2_perf: true` | Per node | 17 tests run per node |
| **Tier 3** | `node_smoke_tier3.connectivity_mode: "run"` | Cluster-wide | 27 tests run cluster-wide |

Tier 1 runs `primus-cli direct --single -- node_smoke` on each node. Tier 2 adds
`--tier2-perf` (large GEMM TFLOPS, HBM D2D bandwidth, local multi-GPU RCCL).
Tier 3 runs `primus-cli direct -- preflight --host --gpu --network` with a
distributed rendezvous across the cluster. Tier 3 is independent of Tier 1.

Preflight reports each tier separately in the console and HTML summary, for example:

```
✅ Node Smoke Tier 1: PASS - 2/2 nodes passed Node Smoke Tier 1; 39 tests run per node
✅ Node Smoke Tier 2: PASS - 2/2 nodes passed Node Smoke Tier 2; 17 tests run per node
✅ Node Smoke Tier 3: PASS - 2/2 nodes passed Node Smoke Tier 3; 27 tests run cluster-wide
```

See [README_preflight_config.md](../../input/config_file/preflight/README_preflight_config.md)
for the full parameter reference, check catalog, and configuration examples.

### Run individual Node Smoke tests

```bash
# Tier 1 only
cvs run preflight_checks test_node_smoke_tier1 \
  --cluster_file cluster.json \
  --config_file preflight_config.json

# Tier 3 only
cvs run preflight_checks test_node_smoke_tier3 \
  --cluster_file cluster.json \
  --config_file preflight_config.json
```

## Output and Reporting

### Console Output
```
✅ Node Health: PASS (2/2 nodes healthy)
✅ GID Consistency: PASS (64/64 interfaces have GID index 3)
⚪ RDMA Connectivity: SKIPPED (Test skipped by configuration)
✅ ROCm Versions: PASS (All nodes running 6.2.0)
✅ Node Smoke Tier 1: PASS - 2/2 nodes passed Node Smoke Tier 1; 39 tests run per node
✅ Node Smoke Tier 2: PASS - 2/2 nodes passed Node Smoke Tier 2; 17 tests run per node
✅ Node Smoke Tier 3: PASS - 2/2 nodes passed Node Smoke Tier 3; 27 tests run cluster-wide

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
3. Run generic GPU node health and optional MI4XX AFM/vPOD admission when enabled
4. Validate ROCm version consistency
5. Run IFoE checks before RDMA-specific eligibility pruning:
   - L2 connectivity using `afmctl test ping` (opt-in)
   - TransferBench scale-up data-path smoketest (opt-in)
6. Run Node Smoke Tier 1 (and Tier 2 when tier2_perf is enabled) per node (opt-in)
7. Run Node Smoke Tier 3 cluster inventory (opt-in)
8. Run RDMA checks:
   - Interface naming and presence
   - GID consistency
   - RDMA connectivity using `ibv_rc_pingpong` (mode-dependent)
9. Generate the comprehensive summary and HTML report
10. Return overall PASS/FAIL status
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
cvs/tests/preflight/
├── __init__.py
├── preflight_checks.py          # Main test module
└── README.md                    # This file

cvs/lib/preflight/
├── gid_consistency.py           # GID validation
├── interface_consistency.py     # RDMA interface checks
├── node_smoke.py                # Node Smoke Tier 1/2
├── node_smoke_counts.py         # Tier test-count catalog
├── tier3_info.py                # Node Smoke Tier 3
├── report.py                    # HTML report generation
└── ...                          # Other check modules

cvs/input/config_file/preflight/
├── preflight_config.json        # Default configuration
└── README_preflight_config.md   # Configuration guide
```

## Advanced Usage

### Custom Configuration

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
        "interfaces": ["mlx5_0", "mlx5_1"],
        "ibv_test_timeout": 15,
        "ibv_test_port_range": "10000-10999"
      },
      "ifoe": {
        "fabric_checks": false
      }
    },
    "reporting": {
      "generate_html_report": true,
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

1. Add the check class or function under `cvs/lib/preflight/`
2. Add the test function to `preflight_checks.py`
3. Update `report.py` summary and HTML generation
4. Add configuration parameters in `cvs/schema/config_file/preflight/config.py` and `preflight_config.json`
5. Add unit tests under the module's `unittests/` directory
6. Update documentation in this README and `README_preflight_config.md`

### Example: Adding a New Check

```python
# In cvs/lib/preflight/my_check.py
class MyCheck:
    def run(self, phdl, config_dict):
        """Check something across all nodes."""
        # Implementation here
        return results

# In preflight_checks.py
def test_my_check(phdl, config_dict):
    """Test my check across cluster nodes."""
    check = MyCheck()
    results = check.run(phdl, config_dict)
    # Validation and reporting here
```

This system provides a solid foundation for ensuring cluster health before running critical workloads.
