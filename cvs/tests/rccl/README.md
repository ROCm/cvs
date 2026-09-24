# RCCL Performance Tests

## Library entry points (for automation)

- `cvs.lib.rccl_lib.RcclJob.from_config(...)` — composes the rccl-tests command, `OpenMPI`, and either `MpiRun` or `Srun`. Call `run_perf()` or `run_regression()`.

RCCL tests in CVS are split into a small set of focused workflows:

1. `rccl_perf`  
   User-facing performance suite. It runs nine fixed collectives and stages one or more env scripts to every node before launch.

2. `rccl_regression`  
   Regression suite with Cartesian product sweep. Uses `regression` object in JSON for NCCL/RCCL env variable combinations, or internal defaults.

**Single-node testing:** Configure a single-node cluster in your cluster JSON file and use the main suites above for single-node testing.

4. `heatmap`  
   Standalone result-comparison suite. It generates a heatmap from two result JSON files and is reusable beyond RCCL-only flows.

All RCCL execution suites still collect host/network info and validate firewall state before performance runs.

## Run Deck reports

With `--html`, `rccl_perf`, `rccl_regression`, and `rccl_pairwise` generate
`rccl_run_deck.html` and `rccl_run_deck.json` in the `<suite>_html` directory
beside the pytest report. Both artifacts are linked from the pytest report
and included in its zip bundle. Perf and regression also retain their
existing amCharts reports.

The deck includes bus bandwidth, algorithm bandwidth, and time curves by
message size, a results table, and a run card. Collectives and message sizes
come from collected results, including when only part of a suite runs.
Pairwise series retain their Phase 0/1/2 labels, and its run card lists the
node and MPI rank counts used across those runs.

Reporting does not change qualification checks. The deck uses the existing
graph conversion: when several rows share a series and message size, the
last row wins across in-place/out-of-place, data type, and cycle dimensions.
The Thresholds row reflects the configured `verify_*` switches; it does not
confirm that a bus-bandwidth threshold was found or applied. Existing
threshold configuration mismatches still require separate correction.
`rccl_perf` still uses its fixed collective parametrization, while regression
uses top-level `rccl.rccl_collective` or defaults to `all_reduce_perf`.

Chart x-axis labels are humanized (`1K`, `1M`, `1G`); the results table and
`rccl_run_deck.json` keep the raw byte size so external tooling can still
sort or filter on it numerically. On large clusters, `rccl_pairwise` can
produce one series per node pair — each chart card renders at most 40 series
(configurable per card via the profile's `max_series`) and shows a "Showing N
of M" banner when truncated; the full set always remains in the results
table and JSON export.

## Prerequisites

1. Provide a valid cluster file, for example `input/cluster_file/cluster.json`.
2. Make sure your env script exports `RCCL_TESTS_BUILD_DIR` for the `*_perf` binaries.
3. Make sure `mpi_params.mpi_dir` points at the Open MPI prefix used by `mpirun` / PMIx.
4. Put RCCL/NCCL/HSA tuning into env scripts. MPI/PMIx/ORTE (`OMPI_MCA_*`, `OPAL_PREFIX`, Spur tmpdir) are set by `rccl_lib` from `mpi_params`, including on nested `spur run`/`srun --mpi=pmix`.
5. Update `results` thresholds for your hardware and cluster size before relying on pass/fail.

## How to run

```bash
cvs list rccl_perf
cvs list rccl_regression
# Single-node testing via cluster config
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
- **Tree + collective rule**: `tree` is skipped for collectives other than `all_reduce_perf`
- Missing `regression` or `{}` → single default case

Example internal config: [`rccl_regression_internal.json`](rccl_regression_internal.json).

Full design notes and a **from `main` rebuild** checklist: [RCCL_HANDOFF_FROM_MAIN.md](RCCL_HANDOFF_FROM_MAIN.md).

### Single-node testing

Single-node RCCL testing is achieved by using either `rccl_perf` or `rccl_regression` with a cluster configuration that contains only one node. The tests automatically adapt to single-node execution.

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
- **Tree filter**: Algorithm restrictions apply to maintain compatibility
