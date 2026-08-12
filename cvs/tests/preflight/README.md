# GPU Cluster Preflight Checks

A comprehensive validation system for GPU clusters before running performance tests, training workloads, and inference tasks.

## Overview

The preflight checks system validates essential cluster health and configuration consistency across all nodes. It performs the following validations:

1. **Ping Reachability** - ICMP `ping` from the CVS driver host to every cluster node *(opt-in, diagnostic only)*
2. **SSH Mesh Connectivity** - All-pairs passwordless SSH reachability across reachable nodes *(opt-in)*
3. **Node Uptime** - Informational `uptime` collection across nodes *(opt-in, never gates status)*
4. **/etc/hosts Consistency** - Validates `/etc/hosts` entries against the cluster inventory *(opt-in)*
5. **limits.conf Validation** - Validates `/etc/security/limits.conf` required lines *(opt-in, blocking)*
6. **NIC Firmware/Host-Software** - Per-vendor (AINIC/Broadcom/Mellanox, selected via `nic_type`) device count and firmware/host-software version validation *(opt-in)*
7. **NIC Driver Version** - Per-vendor (AINIC/Broadcom/Mellanox, selected via `nic_type`) driver/package version validation (Broadcom via `niccli`); nodes without the selected vendor's hardware are SKIPPED *(opt-in)*
8. **Node Health** - Validates AMDGPU/KFD, GPU visibility, and kernel health, with optional MI4XX fabric admission
9. **GID Consistency** - Ensures RDMA interfaces have valid Global Identifier entries
10. **RDMA Interface Presence** - Validates that expected RDMA interfaces are present and link-up
11. **ROCm Version Consistency** - Verifies consistent ROCm versions across all nodes
12. **IFoE L2 Connectivity** - Validates L2 reachability of IFoE links via `afmctl test ping` *(opt-in)*
13. **IFoE TransferBench Smoketest** - Runs the TransferBench candidate-branch `smoketest`
    preset to validate the IFoE scale-up data path (using MI4XX AFM admission or,
    for generic profiles, an `amd-smi fabric --json` single-vPod precondition)
    *(AIMVT-181; opt-in)*
14. **AINIC PFC/QoS/DCQCN** - Validates AINIC PFC, QoS, and DCQCN control-plane configuration via `nicctl`
    *(opt-in, blocking gate run alongside the IFoE L2/TransferBench gates)*
15. **RDMA Connectivity** - Tests node-to-node RDMA communication using `ibv_rc_pingpong`

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
      "expected_rocm_version": "7.15.0",
      "ping_check": { "enabled": false, "count": 4, "timeout_sec": 1 },
      "uptime_check": { "enabled": false },
      "etc_hosts": { "enabled": false, "extra_entries": [] },
      "limits_conf": { "enabled": false },
      "nic_driver_version": {
        "enabled": false,
        "nic_type": ["broadcom"],
        "ainic": { "expected_fw_version": "1.117.5-a-56" },
        "broadcom": { "expected_package_version": "<changeme>" },
        "mellanox": { "expected_mlx5_core_version": "<changeme>", "expected_ofed_version": "<changeme>" }
      }
    },
    "connectivity_check": {
      "ssh_mesh": { "enabled": false, "ssh_timeout_sec": 10 },
      "rdma": {
        "connectivity_mode": "skip"
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
        },
        "nic_firmware": {
          "enabled": false,
          "nic_type": ["ainic"],
          "ainic": { "expected_nic_count": 8, "expected_fw_version": "1.117.5-a-56", "expected_host_version": "1.117.5-a-56" },
          "broadcom": { "expected_nic_count": 2, "expected_fw_version": "<changeme>" },
          "mellanox": { "expected_nic_count": 8, "expected_fw_version": "<changeme>" }
        },
        "pfc_qos_dcqcn": {
          "enabled": false
        }
      }
    },
    "reporting": {
      "generate_html_report": true,
      "artifacts_root_dir": "/tmp/{user-id}/preflight",
      "generate_rdma_pairs_csv": false
    }
  }
}
```

### Key Parameters

- **`node_check.enabled`**: Run GPU node-health and ROCm validation
- **`node_check.gpus_per_node`**: Exact GPU count expected on every node
- **`node_check.ping_check.enabled`**: Run diagnostic ICMP ping reachability from the CVS driver host to every node
- **`node_check.uptime_check.enabled`**: Collect informational `uptime` output across nodes (never gates status)
- **`node_check.etc_hosts.enabled`**: Validate `/etc/hosts` against the cluster inventory (WARNING at worst)
- **`node_check.etc_hosts.extra_entries`**: Additional `{hostname, ip}` pairs that must match exactly
- **`node_check.limits_conf.enabled`**: Validate `/etc/security/limits.conf` required lines (blocking FAIL)
- **`node_check.limits_conf.required_lines`**: Exact lines that must be present on every node. Required (non-empty) when `limits_conf.enabled` is `true` — the check raises rather than silently passing every node if this is left empty
- **`node_check.nic_driver_version.enabled`**: Validate driver/package version for the vendor(s) selected via `nic_type`; SKIPPED on nodes without the selected vendor's hardware
- **`node_check.nic_driver_version.nic_type`**: List of vendor(s) to activate — one or more of `ainic`, `broadcom`, `mellanox` (default `["broadcom"]`)
- **`node_check.nic_driver_version.ainic`** / **`broadcom`** / **`mellanox`**: Per-vendor golden-value sub-blocks. `ainic.expected_fw_version` is checked via `nicctl show version firmware` (the CLI tool installed alongside the AINIC driver); `broadcom.expected_package_version` is checked via `niccli` (the CLI tool installed alongside the Broadcom driver); `mellanox` uses `modinfo`-reported module versions (`expected_*_version`). The `mellanox` sub-block is new and unvalidated against real Mellanox hardware — its `<changeme>` defaults must be filled in before enabling
- **`connectivity_check.ssh_mesh.enabled`**: Validate all-pairs passwordless SSH reachability (WARNING at worst)
- **`connectivity_check.ssh_mesh.ssh_timeout_sec`**: Per-connection SSH timeout for each node-to-node probe
- **`connectivity_check.ifoe.fabric_checks`**: Add MI4XX AIFM/AFM/vPOD and IFoE station/port checks
- **`connectivity_check.rdma.connectivity_mode`**: `"basic"`, `"full_mesh"`, or `"skip"`
- **`connectivity_check.ifoe.l2ping.enabled`**: Run the strict IFoE L2 connectivity gate
- **`connectivity_check.ifoe.l2ping.pings_per_port`**: Samples per discovered UP port pair
- **`connectivity_check.ifoe.transferbench.enabled`**: Run the TransferBench data-path gate
- **`connectivity_check.ifoe.transferbench.scope`**: `"node"` or `"cluster"`
- **`connectivity_check.ifoe.transferbench.profile`**: CVS-supported profile (`"smoketest"`)
- **`connectivity_check.ifoe.transferbench.message_sizes`**, **`iterations`**, **`warmup_iterations`**: Workload intensity
- **`connectivity_check.ifoe.nic_firmware.enabled`**: Validate device count and firmware/host-software versions for the vendor(s) selected via `nic_type`
- **`connectivity_check.ifoe.nic_firmware.nic_type`**: List of vendor(s) to activate — one or more of `ainic`, `broadcom`, `mellanox` (default `["ainic"]`)
- **`connectivity_check.ifoe.nic_firmware.ainic`** / **`broadcom`** / **`mellanox`**: Per-vendor `expected_nic_count` (FAIL on mismatch) and `expected_fw_version` / `expected_host_version` (WARNING on mismatch, AINIC only has host-software); `broadcom.expected_fw_version` is checked against the `FwVersion` column of `niccli --list`; the `mellanox` sub-block is new and unvalidated against real hardware — its `<changeme>` defaults must be filled in before enabling
- **`connectivity_check.ifoe.pfc_qos_dcqcn.enabled`**: Run the AINIC PFC/QoS/DCQCN control-plane blocking gate
- **`connectivity_check.ifoe.pfc_qos_dcqcn.pfc`**, **`qos`**, **`dcqcn`**: Golden-value overrides for each control-plane facet (generic AINIC defaults, fully overridable)
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

## Diagnostic Node and Connectivity Checks

These checks are opt-in (disabled by default) and independent of RDMA/IFoE
eligibility pruning. Each reports a simple per-node `{status, errors}` result.

- **Ping Reachability** (`node_check.ping_check`) — ICMP `ping -c <count> -W
  <timeout_sec>` from the CVS driver host to every cluster node. Diagnostic
  only: a node-level FAIL is demoted to WARNING in the summary and never
  gates overall preflight status.
- **SSH Mesh Connectivity** (`connectivity_check.ssh_mesh`) — all-pairs SSH
  reachability probe between every reachable node (not just driver-host to
  node). WARNING at worst; never prunes nodes.
- **Node Uptime** (`node_check.uptime_check`) — collects `uptime` output per
  node for the HTML report. Purely informational; never affects status.
- **/etc/hosts Consistency** (`node_check.etc_hosts`) — validates that every
  cluster node address has an `/etc/hosts` entry, plus any explicit
  `extra_entries` `{hostname, ip}` pairs. WARNING at worst.
- **limits.conf Validation** (`node_check.limits_conf`) — validates that all
  `required_lines` are present (whitespace-insensitive, any order) in
  `/etc/security/limits.conf` on every node. Blocking FAIL when enabled.
- **NIC Firmware/Host-Software** (`connectivity_check.ifoe.nic_firmware`) —
  per-vendor check, activated via `nic_type` (one or more of `ainic`,
  `broadcom`, `mellanox`). AINIC validates device count via `ibv_devices`
  (FAIL on mismatch) and firmware/host-software version via `nicctl show
  version {firmware,host-software}` (WARNING on mismatch); Broadcom
  validates both device count and firmware version from a single `niccli
  --list` command (the CLI tool installed alongside the Broadcom driver;
  count mismatch FAILs, firmware mismatch WARNs); Mellanox validates device
  count via `ibv_devices` and firmware version via `ethtool -i` (WARNING on
  mismatch). A node without a configured vendor's hardware is SKIPPED for
  that vendor rather than FAILed. The entire Mellanox sub-block is new and
  unvalidated against real hardware.
- **NIC Driver Version** (`node_check.nic_driver_version`) — per-vendor
  check, activated via `nic_type` (one or more of `ainic`, `broadcom`,
  `mellanox`). AINIC validates the per-NIC firmware version via `nicctl
  show version firmware` (`Uboot-A`/`Firmware-A` fields, the CLI tool
  installed alongside the AINIC driver); Broadcom validates the NIC
  package version via `niccli` (`niccli -i <idx> show --pkg_ver`'s "Active
  Package Version", the CLI tool installed alongside the Broadcom driver --
  it does not report kernel module/`modinfo` version at all); Mellanox
  validates `mlx5_core` module version and the MLNX_OFED stack version via
  `ofed_info -s`. Nodes without a configured
  vendor's hardware (e.g. AINIC/Pollara fleets when only `broadcom` is
  selected) report SKIPPED rather than FAIL, so this check is safe to leave
  enabled cluster-wide on mixed fleets. The Mellanox sub-block is new and
  unvalidated against real Mellanox hardware.

### Example config block

```json
"node_check": {
  "ping_check": { "enabled": true, "count": 4, "timeout_sec": 1 },
  "uptime_check": { "enabled": true },
  "etc_hosts": { "enabled": true, "extra_entries": [] },
  "limits_conf": {
    "enabled": true,
    "required_lines": [
      "* soft memlock unlimited",
      "* hard memlock unlimited",
      "* soft nofile 1048576",
      "* hard nofile 1048576"
    ]
  },
  "nic_driver_version": {
    "enabled": true,
    "nic_type": ["broadcom"],
    "broadcom": { "expected_package_version": "<changeme>" }
  }
},
"connectivity_check": {
  "ssh_mesh": { "enabled": true, "ssh_timeout_sec": 10 },
  "ifoe": {
    "nic_firmware": {
      "enabled": true,
      "nic_type": ["ainic"],
      "ainic": { "expected_nic_count": 8, "expected_fw_version": "1.117.5-a-56", "expected_host_version": "1.117.5-a-56" }
    }
  }
}
```

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

## AINIC PFC/QoS/DCQCN

Validates AINIC PFC, QoS, and DCQCN control-plane configuration via `nicctl`
on every reachable node, run alongside the IFoE L2/TransferBench gates
(after the node-health admission gate, before RDMA connectivity). This is a
blocking FAIL gate when enabled through
`preflight.connectivity_check.ifoe.pfc_qos_dcqcn.enabled`.

This check is AINIC-specific (it drives `nicctl`) and is additionally
guarded by the sibling `connectivity_check.ifoe.nic_firmware.nic_type`
selector (which defaults to `["ainic"]`): if `nic_type` does not include
`"ainic"`, the check is SKIPPED regardless of `pfc_qos_dcqcn.enabled`, since
running AINIC-only tooling against a Broadcom/Mellanox fleet would otherwise
produce a false failure rather than a meaningful result.

Three independent validators run per node and are combined into a single
per-node verdict (FAIL if any sub-check fails):

- **PFC** — validates the number of AINIC cards reporting PFC state and the
  pause type (`pfc.expected_card_count`, `pfc.expected_pause_type`).
- **QoS** — validates DSCP-to-priority mappings, the PFC-enabled priority
  bitmap and no-drop priorities, and per-priority scheduling
  (algorithm|weight|rate-limit) for priorities 0, 3, and 6.
- **DCQCN** — validates the active DCQCN profile ID, enablement status, and
  the full set of congestion-control tuning parameters (AI rate, byte count,
  alpha gain, monitor period, token bucket, CNP DSCP marking, etc.).

All golden values under `pfc`, `qos`, and `dcqcn` are generic AINIC defaults
and fully overridable per deployment; none are hardcoded to a specific
customer environment.

### Example config block

```json
"ifoe": {
  "pfc_qos_dcqcn": {
    "enabled": true,
    "pfc": { "expected_card_count": 8, "expected_pause_type": "PFC" },
    "qos": { "expected_card_count": 8, "dscp24_priority": "3", "dscp46_priority": "6" },
    "dcqcn": { "expected_device_count": 8, "profile_id": 1, "status": "Enabled" }
  }
}
```

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
2. Elimination tier: opt-in diagnostic ping reachability (never prunes),
   SSH echo reachability (prunes unreachable nodes), opt-in all-pairs SSH
   mesh check (diagnostic, never prunes), opt-in uptime (informational)
3. Static config tier: opt-in /etc/hosts consistency, opt-in limits.conf
   (blocking FAIL when enabled), opt-in per-vendor NIC firmware/device-count
   validation, opt-in per-vendor NIC driver version (SKIPPED on nodes
   without the selected vendor's hardware) — none of these prune
4. Run generic GPU node health and optional MI4XX AFM/vPOD admission when
   enabled (mandatory FAIL gate)
5. Validate ROCm version consistency
6. Run IFoE/functional checks, each BLOCKED (not run) if node-health
   admission failed:
   - L2 connectivity using `afmctl test ping` (opt-in)
   - AINIC PFC/QoS/DCQCN control-plane validation using `nicctl` (opt-in)
   - TransferBench scale-up data-path smoketest (opt-in)
7. Run RDMA tier:
   - Interface naming and presence (prunes)
   - GID consistency (prunes)
   - RDMA connectivity using `ibv_rc_pingpong` (mode-dependent, BLOCKED if
     node-health admission failed)
8. Generate the comprehensive summary and HTML report
9. Return overall PASS/FAIL status
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

#### limits.conf Failures
```bash
# Inspect current limits.conf
cat /etc/security/limits.conf

# Verify effective limits for the running shell
ulimit -a
```

#### NIC Firmware/Host-Software Mismatches
```bash
# Count RDMA devices (AINIC/Mellanox)
ibv_devices

# AINIC: check firmware/host-software versions
nicctl show version

# Broadcom: check NIC count and per-NIC firmware version in one command
niccli --list

# Mellanox: check per-interface firmware version
ethtool -i <iface>
```

#### NIC Driver Version Mismatches
```bash
# Broadcom: check NIC package version via niccli
niccli --list
niccli -i <idx> show --pkg_ver

# AINIC: check per-NIC firmware version via nicctl
nicctl show version firmware

# Mellanox: check mlx5_core module version and OFED stack version
modinfo mlx5_core
ofed_info -s
```

#### PFC/QoS/DCQCN Failures
```bash
# Inspect AINIC control-plane state
nicctl show pfc
nicctl show qos
nicctl show dcqcn
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
