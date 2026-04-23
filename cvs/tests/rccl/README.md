# RCCL Performance Tests

## Library entry points (for automation)

- `cvs.lib.rccl_lib.run_rccl` — stages the env script on the cluster, sets `env_source_script`, then runs the selected runner (`single`, `cluster`, or `cluster_default`).
- `cvs.lib.rccl_lib.run_rccl_from_context` — same with a `RcclRunContext` (handles + runner) and a separate `engine_params` dict for message sizes, `rccl_result_file`, `mpi_oob_port`, etc.
- `cvs.lib.rccl_lib.rccl_perf` / `rccl_regression` — call `run_rccl` for perf or multi-case regression.
- Low-level `rccl_cluster_test` / `rccl_cluster_test_default` / `rccl_single_node_test` remain for legacy calls; new code should prefer `run_rccl`.

RCCL tests in CVS are split into a small set of focused workflows:

1. `rccl_perf`  
   User-facing performance suite. It runs the configured collectives and stages one or more env scripts to every node before launch.

2. `rccl_regression`  
   Regression suite with Cartesian product sweep. Uses `regression` object in JSON for NCCL/RCCL env variable combinations, or internal defaults.

3. `rccl_singlenode_cvs`  
   Single-node RCCL suite for local collective checks.

4. `heatmap`  
   Standalone result-comparison suite. It generates a heatmap from two result JSON files and is reusable beyond RCCL-only flows.

All RCCL execution suites still collect host/network info and validate firewall state before performance runs.

## Prerequisites

1. Provide a valid cluster file, for example `input/cluster_file/cluster.json`.
2. Make sure your env script exports `RCCL_TESTS_BUILD_DIR` for the `*_perf` binaries.
3. Make sure your env script exports `MPI_HOME` for the Open MPI install used by `mpirun`.
4. Put RCCL/NCCL/UCX tuning into env scripts. CVS now stages and sources those env files instead of building `mpirun -x ...` lists from JSON.
5. Update `results` thresholds for your hardware and cluster size before relying on pass/fail.

## How to run

```bash
cvs list rccl_perf
cvs list rccl_regression
cvs list rccl_singlenode_cvs
cvs list heatmap
```

### Performance

```bash
cvs run rccl_perf \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_perf.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_perf.log -vvv -s
```

`rccl_perf` accepts either:

- `env_files`: list of env scripts to stage and run (supports multiple env files)
- `env_source_script`: single env script fallback

Starter scripts are shipped under `input/config_file/rccl/`:

- `ainic_env_script.sh`
- `thor2_env_script.sh`
- `cx7_env_script.sh`

Most clusters should list only the one that matches their NIC in `env_files`, or use multiple for comparison.

### Regression

```bash
cvs run rccl_regression \
  --cluster_file input/cluster_file/cluster.json \
  --config_file /path/to/rccl_regression.json \
  --html=/var/www/html/cvs/rccl_regression.html --capture=tee-sys --self-configured-html \
  --log-file=/tmp/rccl_regression.log -vvv -s
```

`rccl_regression` uses a `regression` object in the config JSON:

```json
"regression": {
  "NCCL_ALGO": ["ring", "tree"],
  "NCCL_PROTO": ["Simple"], 
  "NCCL_IB_QPS_PER_CONNECTION": ["1", "2"],
  "NCCL_PXN_DISABLE": ["0", "1"]
}
```

- **Keys** are real NCCL/RCCL environment variable names
- **Values** are lists; CVS builds the Cartesian product
- **Tree + collective rule**: `tree` is skipped for collectives other than `all_reduce_perf` (same as main's old `rccl_multinode_cvs`)
- Missing `regression` or `{}` → single default case

Example internal config: [`rccl_regression_internal.json`](rccl_regression_internal.json).

Full design notes and a **from `main` rebuild** checklist: [RCCL_HANDOFF_FROM_MAIN.md](RCCL_HANDOFF_FROM_MAIN.md).

### Single-node

```bash
cvs run rccl_singlenode_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/single_node_mi355_rccl.json \
  --html=/var/www/html/cvs/rccl_singlenode.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_singlenode.log -vvv -s
```

### Heatmap

```bash
cvs run heatmap \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/heatmap/heatmap_config.json \
  --html=/var/www/html/cvs/heatmap.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/heatmap.log -vvv -s
```

Heatmap config requires:
- `actual_json_file`: Path to test results JSON
- `reference_json_file`: Path to golden/reference results JSON
- `heatmap_output_file`: Output HTML path (optional)

## Configuration

The RCCL config keeps benchmark intent and validation settings, while RCCL/NCCL/UCX tuning moves into env scripts.

### Env files vs JSON scope

- **Env scripts**: `NCCL_*`, `RCCL_*`, `UCX_*`, `MPI_HOME`, `RCCL_TESTS_BUILD_DIR`, and runtime paths
- **JSON config**: collectives list, message sizes, thresholds (`results`), `mpi_pml`, cluster orchestration

### Performance config

```json
{
  "rccl": {
    "env_files": ["/root/ainic_env_script.sh", "/root/thor2_env_script.sh"],
    "rccl_collective": ["all_reduce_perf", "all_gather_perf"],
    "rccl_result_file": "/tmp/rccl_perf_result.json",
    "start_msg_size": "1024",
    "end_msg_size": "16g",
    "results": { }
  }
}
```

### Regression config

```json
{
  "rccl": {
    "env_source_script": "/root/thor2_env_script.sh", 
    "rccl_collective": ["all_reduce_perf"],
    "regression": {
      "NCCL_ALGO": ["ring", "tree"],
      "NCCL_PROTO": ["Simple"],
      "NCCL_IB_QPS_PER_CONNECTION": ["1", "2"]
    },
    "start_msg_size": "1024",
    "end_msg_size": "16g",
    "results": { }
  }
}
```

## Implementation details

- **Env staging**: `run_rccl` stages env scripts to `/tmp/cvs_rccl_env/` on all nodes with optional per-case overrides
- **Regression cases**: Each combination in the Cartesian product gets its own result file suffix
- **Single env dump**: Only `test_print_env_once` prints the environment; other tests focus on performance
- **Tree filter**: Matches main's `rccl_multinode_cvs.py` behavior for algorithm restrictions