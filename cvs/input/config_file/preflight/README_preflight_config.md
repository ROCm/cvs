# Preflight Configuration Guide

This document explains how to configure the GPU cluster preflight checks system.

## Overview

The preflight checks system validates essential cluster health before running performance tests like IB performance tests, RCCL training, and inference workloads. It performs the following validations:

1. **GID Consistency** - Ensures RDMA interfaces have valid GID entries
2. **RDMA Connectivity** - Tests node-to-node RDMA communication using ibv_rc_pingpong
3. **ROCm Version Consistency** - Verifies consistent ROCm versions across nodes
4. **Interface Name Consistency** - Validates RDMA interface naming patterns
5. **MI4XX Node-Health Admission (mandatory when enabled)** - Validates
   AMDGPU/KFD, AIFM node agent, four GPU/AFM devices, AFM `ACTIVE`
   bare-metal vPOD membership, and IFoE station-mask/AFM-UP-port coherence
6. **IFoE L2 Connectivity (AIMVT-180; opt-in)** - Runs `afmctl test ping`
   on each node and enforces per-port and Summary pass/fail accounting
7. **IFoE TransferBench Smoketest (AIMVT-181; opt-in)** - Runs the
   TransferBench candidate-branch `smoketest` preset to validate IFoE
   scale-up data-path one layer above L2 (using the MI4XX AFM admission
   result, not `amd-smi fabric --topology`, when MI4XX is enabled)

## Configuration File Structure

The preflight configuration file follows this structure:

```json
{
  "preflight": {
    "debug": {
      "scriptlet": false
    },
    "node_check": {
      "gid_index": "3",
      "expected_rocm_version": "6.2.0",
      "rdma_interfaces": ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
      "mi4xx_node_health": {"enabled": false, "failure_mode": "gate"}
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
├── node_check/           # Individual node validation parameters
│   └── mi4xx_node_health/ # Mandatory MI4XX driver/AIFM/AFM/vPOD admission when enabled
├── connectivity_check/   # Inter-node connectivity tests
│   ├── rdma/             # RDMA-specific parameters (including nodes_per_full_mesh_group)
│   ├── ifoe/             # IFoE L2 ping parameters (AIMVT-180; opt-in)
│   └── transferbench/    # IFoE TransferBench smoketest parameters (AIMVT-181; opt-in)
└── reporting/           # Output and report generation
```

### Execution Flow
1. **node_check** - Validate individual nodes in parallel; MI4XX admission runs after SSH reachability
2. **MI4XX admission** - When enabled, block IFoE, TransferBench, and RDMA scale-up tests until every declared node is admitted
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

#### MI4XX Node-Health Admission (`node_check.mi4xx_node_health`)

Set `enabled: true` for every MI4XX rack scale-up preflight. This is a
mandatory **read-only** gate: it records platform state and never attempts to
load `amdgpu`, start an AIFM agent, change a vPOD, change a station mask, or
reboot a node. If it fails, CVS writes the report, marks IFoE L2,
TransferBench, and RDMA scale-up checks as `BLOCKED`, then fails the preflight
command after report generation.

- **`failure_mode`** (required: `"gate"` when enabled)
  - MI4XX admission cannot be downgraded to report-only behavior.
- **Expected inventory** (`expected_gpus_per_node: 4`,
  `expected_ifoe_devices_per_node: 4`, `expected_network_ports_per_device: 36`)
  - Requires four GPUs from `amd-smi list --json`, four AFM devices from
    `afmctl show device --json`, and 36 AFM network ports per device.
- **Tool paths and privileges**
  - Set `afmctl_path`, `amd_smi_path`, and `use_sudo` to match the selected
    rack image. CVS does not assume the tools are on a benchmark user's
    `PATH`.
- **AMDGPU and AIFM state** (`required_ifoe_modules`, `agent_process_name`,
  `agent_slot_ids`)
  - Requires loaded `amdgpu`, `/dev/kfd`, required IFoE module(s), a running
    AIFM node agent, and optionally a configured per-node slot ID. Current-boot
    kernel initialization failure signatures also fail admission.
- **AFM/vPOD state**
  - Every AFM device must be `ACTIVE`, `bare-metal`, have an accelerator ID
    (including valid ID `0`), and report non-empty vPOD accelerator membership.
    The membership must agree across all devices and all declared benchmark
    nodes. CVS polls read-only AFM state for `readiness_timeout_seconds`
    (default 600 seconds); it does not perform recovery.
- **Station-mask policy**
  - `lane_en_bitmap` contains 18 station nibbles per GPU (36 network ports).
    `f` means fully enabled, `0` means intentionally masked/disabled, and
    partial `c`/`3` is rejected. The default permits valid `f`/`0` mixes; it
    does **not** impose an all-ports-enabled policy.
  - CVS uses AFM `spec.station_id` to require both ports of each `f` station
    to be physically UP. Physical links reported for a `0`-masked station are
    intentionally ignored: a station mask is an IFoE enablement policy, not a
    promise that its associated cables are electrically down. Set `min_up_ports_per_gpu`,
    `require_uniform_station_mask`, or `expected_station_masks` only for a
    deployment that explicitly requires that policy.
- **Rendezvous/firewall coverage**
  - CVS does not add a generic port scanner. Keep the existing RDMA
    `full_mesh` test enabled with an `ibv_test_port_range` that is permitted
    between all nodes; that directly validates the TCP/RDMA path used by
    RCCL-style scale-up rendezvous. TransferBench `multi_rank` likewise
    requires `socket_master_port` to be reachable between all selected nodes.

#### IFoE Settings (`connectivity_check.ifoe`) — opt-in (AIMVT-180)

Runs `afmctl test ping` on each reachable node and validates its aggregate
`Summary:` block. AFM's `--skip-pass` output retains failed port rows for
diagnostics while omitting successful rows; CVS verifies Summary totals against
the selected-port count.

- **`connectivity_mode`** (default: `"skip"`)
  - `"run"` — execute the L2 ping on every reachable node
  - `"skip"` — preflight records a SKIPPED result and does not invoke afmctl
- **`afmctl_path`** (default: `"afmctl"`)
  - Absolute path or PATH-resolved binary name on each node
- **`skip_pass`** (default: `true`)
  - Passes `--skip-pass` to `afmctl test ping`, reducing output to failed port
    rows and the Summary block.
  - A passing summary still proves complete selected-port coverage because CVS
    requires each traffic type's `total` to equal `selected ports × -c`.
  - Set it to `false` only when complete per-port PASS output is needed for a
    diagnostic artifact.
- **`use_sudo`** (default: `false`)
  - Prepend `sudo` to the afmctl invocation when the cluster image requires root
- **`json_args`** (default: `["--json"]`) and **`allow_text_fallback`** (default: `false`)
  - CVS requests JSON for `afmctl show device` and `show port -b <BDF>`, and
    uses those JSON parsers for topology and port discovery. It writes each
    port inventory to a `umask 077` remote `/tmp` artifact, retrieves that
    single-host artifact through SFTP, and removes it before continuing. This
    avoids parsing AFM JSON through SSH stdout; command, SFTP, cleanup, and
    parse errors all fail closed.
  - This AFM release does not support `--json` for `afmctl test ping`; CVS
    parses its documented text Summary and any failed-port rows. Discovery text
    parsing remains an explicit compatibility diagnostic only.
- **`bdf_discovery`** (default: `"auto"`)
  - `"auto"` — run `afmctl show device` on each node and use the reported BDFs
  - `"config"` — use only the `bdfs` list below; nodes with no matching BDFs FAIL
- **`bdfs`** (default: `[]`)
  - Optional explicit list of accelerator BDFs to test on every node
  - Example: `["0001:01:00.1"]`
- **`dst_accelerators`** (default: `[0]`)
  - Compatibility-only destination list for `mesh_mode: "config"`
  - Do not use a shared static list for a rack with node-specific global accelerator IDs
- **`mesh_mode`** (default: `"config"`; benchmark recommendation: `"full_mesh"`)
  - `"config"` — preserve the legacy `(source BDF, dst_accelerator)` combinations
  - `"full_mesh"` — discover source accelerator IDs and vPOD membership, then test every ordered non-self source-to-peer pair
  - A full mesh excludes `source == destination`; a self-ping is an invalid coverage cell, not a fabric result
  - With `strict_discovery: true`, a missing vPOD membership list fails the gate rather than silently substituting a local-only mesh. Set strict discovery to `false` only for a diagnostic run while collecting the missing hardware fixture.
- **`ports`** (default: `"all"`; benchmark recommendation: `"up"`)
  - `"up"` writes `afmctl show port -b <BDF> --json` to a private remote file,
    retrieves it over SFTP, and supplies the discovered UP ports explicitly
    through `-p`
  - When MI4XX node-health admission is enabled, `"up"` is further intersected
    with the port IDs from its `f` stations. This excludes physical links that
    AFM reports as UP but the rack intentionally station-masks out.
  - `"all"` omits `-p`, a string such as `"0-7"` or `"0,1,2"`, or a list `[0, 1, 2]`
  - `"all"` can include intentionally down or unwired ports and therefore must not be used for a strict benchmark gate
- **`port_discovery`** (default: `"auto"`)
  - Used with `ports: "up"`; CVS parses the SFTP-retrieved JSON artifact and
    fails closed if AFM command execution, SFTP, artifact cleanup, or port
    state parsing fails
  - If the hardware's output is unknown or malformed, strict discovery fails closed rather than falling back to all ports
- **`pings_per_port`** (default: `1`; benchmark recommendation: `3`)
  - Passed to afmctl as `-c <count>`
- **`per_ping_timeout`** (default: `null`)
  - Optional afmctl `-t <minutes>` value; omitted when `null`
- **`traffic_types`** (default: `["ifoe_req", "ifoe_resp", "non_ifoe"]`)
  - Determines which afmctl traffic categories are required to pass
  - When all three are selected, `--traffic-type` is omitted so afmctl runs them all
- **`loss_threshold_pct`** (default: `0.0`)
  - Maximum tolerated loss percentage per traffic type (Summary line)
- **`ssh_timeout`** (default: `180`)
  - Per-invocation SSH timeout (seconds); raise for high `pings_per_port`
- **`require_complete_coverage`** (default: `true`)
  - Require every planned source, non-self destination, selected UP port, and invocation result to be present
  - Required nodes lost to an earlier preflight prune are reported as missing coverage, rather than silently reducing the mesh
- **`strict_discovery`** (default: `true`)
  - Treat missing/malformed topology, vPOD membership, or port state as a failure
- **`failure_mode`** (default: `"report"`; benchmark recommendation: `"gate"`)
  - `"report"` preserves diagnostic-only preflight behavior
  - `"gate"` records the detailed report and then fails pytest/CLI if L2 results or coverage fail

> **MI4XX benchmark profile.** Use `mesh_mode: "full_mesh"`, `ports: "up"`,
> `strict_discovery: true`, and `failure_mode: "gate"`. This validates every
> active IFoE path while allowing stations intentionally masked by the rack
> configuration.

#### TransferBench Settings (`connectivity_check.transferbench`) — opt-in (AIMVT-181)

Runs the TransferBench candidate-branch **`smoketest`** preset on every reachable
node to validate IFoE scale-up data-path connectivity (one layer above the
AIMVT-180 L2 ping). With MI4XX node health enabled, it consumes the mandatory
`afmctl show device --json` admission result and never queries unreliable
`amd-smi fabric --topology`. Generic non-MI4XX profiles retain the `amd-smi`
precondition. The binary's exit code is reconciled with per-cell
`[PASS]/[FAIL]/[SKIP]` markers.

> **PATH / LD_LIBRARY_PATH.** This check resolves `TransferBench` and `amd-smi`
> from each node's `PATH`. Cluster-wide tool location (e.g. a non-default ROCm
> install root) should be set via the cluster file's top-level `env_vars` block
> (see [`cvs/input/cluster_file/README.md`](../../cluster_file/README.md)), not
> duplicated here.

- **`connectivity_mode`** (default: `"skip"`)
  - `"run"` — execute the smoketest on every reachable node
  - `"skip"` — preflight records a SKIPPED result and does not invoke TransferBench
- **`tb_binary`** (default: `"TransferBench"`)
  - TransferBench binary name; PATH-resolved on each node. Override with an
    absolute path here only when this single preflight check needs to point at
    a different binary than the rest of the cluster's tooling. The
    pod-membership precondition uses `amd-smi` resolved from `PATH` and has no
    per-check override (use cluster file `env_vars` to point at a non-default
    install).
- **`use_sudo`** (default: `true`)
  - Prepend `sudo` to TransferBench and generic-mode amd-smi calls (typically
    required on production cluster images for IFoE access)
- **`preset`** (default: `"smoketest"`)
  - TransferBench preset name. Change only when a TransferBench build ships
    a renamed preset with the same semantics.
- **`size_list`** (default: `["1K", "16M"]`)
  - Transfer sizes passed positionally to TransferBench. Kept short for
    preflight; default covers both small/latency and large/bandwidth regimes.
- **`num_iterations`** (default: `2`)
  - `NUM_ITERATIONS` env var. Two iterations is enough to surface intermittent
    failures without blowing up runtime.
- **`num_warmups`** (default: `0`)
  - `NUM_WARMUPS` env var. Preflight is a connectivity gate, not a benchmark.
- **`always_validate`** (default: `true`)
  - `ALWAYS_VALIDATE=1` so every iteration is data-validated. Required for
    silent-corruption detection.
- **`run_parallel`** (default: `true`)
  - `RUN_PARALLEL=1` so tests with disjoint executors run concurrently.
- **`use_bdma`** (default: `false`)
  - `USE_BDMA=1` to prefer the BDMA path on supported hardware.
- **`force_single_pod`** (default: `true`)
  - `FORCE_SINGLE_POD=1` — defense in depth alongside the AFM admission on
    MI4XX or the amd-smi vPod precondition in generic mode.
- **`rank_mode`** (default: `"per_node"`)
  - `"per_node"` — each reachable node runs an independent single-rank
    TransferBench (exercises intra-node AID↔MID IFoE only)
  - `"multi_rank"` — every reachable node is wired into one socket-comm
    cluster (`TB_NUM_RANKS=N`, `TB_RANK=0..N-1`, `TB_MASTER_ADDR=<rank0>`)
    so the smoketest traverses the rack IFoE switch end-to-end. Requires
    bidirectional IP reachability on `socket_master_port` between every pair
    of nodes.
- **`socket_master_port`** (default: `31337`)
  - `TB_MASTER_PORT` used in `multi_rank` mode. Must be free and open in
    the firewall on every node.
- **`master_node`** (default: `""`)
  - Optional hostname / IP forced to rank 0; defaults to the lexicographically
    smallest reachable host.
- **`max_skip_pct`** (default: `25.0`)
  - Maximum percentage of smoketest cells allowed to be `SKIP` before the
    node is downgraded to `WARNING`. Set to `0.0` for the strictest gate.
- **`ssh_timeout`** (default: `600`)
  - Per-invocation SSH timeout (seconds) for each TransferBench run
- **`skip_pod_check`** (default: `false`)
  - Compatibility-only bypass of the generic amd-smi pod-membership check.
    It is ignored for MI4XX: AFM/vPOD admission is mandatory and cannot be
    skipped.

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
