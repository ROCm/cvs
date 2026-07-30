# Preflight Configuration Guide

CVS preflight validates node-local GPU health, ROCm consistency, optional
MI4XX IFoE fabric health, and optional RDMA connectivity before benchmarks run.
The configuration is grouped by subsystem so that settings live beside the
check that consumes them.

## MI450 example

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
      "artifacts_root_dir": "/tmp/preflight",
      "generate_rdma_pairs_csv": false
    },
    "debug": {
      "scriptlet": false
    }
  }
}
```

Replace `expected_rocm_version` with the release installed on the cluster.

## Node checks

`preflight.node_check` contains generation-independent node validation:

- `enabled` enables GPU visibility, AMDGPU/KFD, kernel-health, and ROCm
  validation. Set it to `false` to skip those checks.
- `gpus_per_node` is the exact AMD GPU count expected on every node.
- `expected_rocm_version` is the ROCm version required on every node.

## IFoE checks

`preflight.connectivity_check.ifoe` contains the MI4XX fabric checks:

- `fabric_checks` adds AIFM/AFM/vPOD, station-mask, and IFoE port admission to
  node health. It requires `node_check.enabled: true`.
- `l2ping.enabled` enables strict `afmctl test ping` validation before
  TransferBench and RDMA.
- `l2ping.pings_per_port` controls how many ping samples are sent per selected
  port pair.
- `transferbench.enabled` enables the TransferBench data-path gate.
- `transferbench.scope` is `"node"` for independent node runs or `"cluster"`
  for one multi-rank run.
- `transferbench.profile` currently supports `"smoketest"`.
- `message_sizes`, `iterations`, and `warmup_iterations` control the workload.

CVS owns executable discovery, privilege handling, topology discovery, strict
coverage policy, traffic types, timeout derivation, and result parsing.

## RDMA checks

`preflight.connectivity_check.rdma` owns RDMA inventory and connectivity:

```json
"rdma": {
  "connectivity_mode": "full_mesh",
  "gid_index": "3",
  "interfaces": ["enp4s0np0"],
  "nodes_per_full_mesh_group": 128,
  "ibv_test_timeout": 90,
  "ibv_test_port_range": "10000-50000"
}
```

- `connectivity_mode` is `"basic"`, `"full_mesh"`, or `"skip"`.
- `gid_index` is the GID index validated on each configured interface.
- `interfaces` lists the RDMA devices expected on every node.
- `nodes_per_full_mesh_group`, `ibv_test_timeout`, and
  `ibv_test_port_range` tune full-mesh execution when needed.

When `connectivity_mode` is `"skip"`, CVS skips RDMA interface presence, GID
validation, and pairwise RDMA connectivity. IFoE checks are independent and
can still run.

### Legacy RDMA configuration — deprecated

Existing RDMA preflight configurations may temporarily keep these two legacy
paths:

| Deprecated path | Canonical path |
|---|---|
| `preflight.node_check.gid_index` | `preflight.connectivity_check.rdma.gid_index` |
| `preflight.node_check.rdma_interfaces` | `preflight.connectivity_check.rdma.interfaces` |

CVS normalizes the legacy values before validation and prints a deprecation
warning. These paths will be removed in a future release; new configurations
must use the canonical `connectivity_check.rdma` shape shown above.

If a deprecated and canonical path are both present, their values must match.
Conflicting values are rejected rather than silently selecting a different GID
or interface inventory. This compatibility exception does not apply to the
retired `node_health`, `l2ping`, or `transferbench` blocks.

## Reporting and debug

- `reporting.generate_html_report` and `generate_rdma_pairs_csv` are JSON
  booleans.
- `reporting.artifacts_root_dir` selects the report and RDMA artifact root.
- `debug.scriptlet` preserves RDMA ScriptLet diagnostics and enables detailed
  tracing; leave it disabled unless troubleshooting.

The retired flat `node_health`, `l2ping`, and `transferbench` blocks are not
accepted. Use `node_check` and `connectivity_check.ifoe` as shown above.
