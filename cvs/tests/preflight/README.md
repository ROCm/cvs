# GPU Cluster Preflight Checks

A comprehensive validation system for GPU clusters before running performance tests, training workloads, and inference tasks.

## Overview

The preflight checks system validates essential cluster health and configuration consistency across all nodes. It performs four critical validations:

1. **GID Consistency** - Ensures RDMA interfaces have valid Global Identifier entries
2. **RDMA Connectivity** - Tests node-to-node RDMA communication using `rping`
3. **ROCm Version Consistency** - Verifies consistent ROCm versions across all nodes
4. **RDMA Interface Presence** - Validates that expected RDMA interfaces are present

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

### Key Parameters

- **`rdma_connectivity_check`**: `"basic"`, `"full_mesh"`, `"sample"`, or `"skip"`
- **`expected_rocm_version`**: ROCm version expected across all nodes
- **`rdma_interfaces`**: List of expected RDMA interface names
- **`rping_timeout`**: Timeout in seconds for connectivity tests
- **`rping_port_range`**: Port range for parallel rping tests

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
| Sample | ~1-2min | ~3-5min | ~8-12min |
| Full Mesh | ~5-10min | ~20-30min | ~2-3hrs |

### Network Requirements

- **SSH Access**: Passwordless SSH to all cluster nodes
- **RDMA Interfaces**: Active InfiniBand or RoCE interfaces
- **Port Access**: rping_port_range must not be blocked by firewalls
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
    "gid_index": "3",
    "expected_rocm_version": "6.2.0",
    "rdma_connectivity_check": "full_mesh",
    "rdma_interfaces": ["mlx5_0", "mlx5_1"],
    "rping_timeout": "15",
    "rping_port_range": "10000-10999",
    "generate_html_report": "true",
    "report_output_dir": "/shared/preflight_reports"
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
- Use `"sample"` mode for regular health checks
- Use `"full_mesh"` mode only for critical validation
- Consider hierarchical testing (rack-by-rack)
- Increase `rping_timeout` for high-latency networks
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