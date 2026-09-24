# RCCL Performance Tests

## Library entry points (for automation)

- `cvs.lib.rccl_lib.RcclJob.from_config(...)` — composes the rccl-tests command, `OpenMPI`, and either `MpiRun` or `Srun`. Call `run_perf()` or `run_regression()`.

RCCL tests in CVS are split into a small set of focused workflows:

1. `rccl_perf`  
   User-facing performance suite. It runs the configured collectives and stages one or more env scripts to every node before launch.

2. `rccl_regression`  
   Regression suite with Cartesian product sweep. Uses `regression` object in JSON for NCCL/RCCL env variable combinations, or internal defaults.

**Single-node testing:** Configure a single-node cluster in your cluster JSON file and use the main suites above for single-node testing.

4. `heatmap`  
   Standalone result-comparison suite. It generates a heatmap from two result JSON files and is reusable beyond RCCL-only flows.

All RCCL execution suites still collect host/network info and validate firewall state before performance runs.

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

### Reported vs requested topology

CVS compares the `nodes`, `ranks`, `ranksPerNode`, and `gpusPerRank` in rccl-tests
JSON with the topology requested by the launch command. This check runs for every
performance data type and every regression case, including pairwise performance
runs. It also detects topology changes between rows in the same result file.

Set `rccl.cvs_params.topology_check` in your config:

| Mode | Behaviour |
| --- | --- |
| `warn` (default) | Log a mismatch banner and record the differences; topology mismatches alone do not fail the test. |
| `strict` | Record the differences and fail the test through the normal CVS result checks. Result collection continues, and pairwise runs are marked unclean. |
| `off` | Disable the requested-vs-reported comparison. Result parsing and schema validation remain active, and aggregation still rejects rows whose topology disagrees. The audit file contains the mode and an empty check list. |

Older configs that omit the setting use `warn`. Mode names are case insensitive;
unrecognised values produce a warning and fall back to `warn`.

For Slurm/Spur, requested topology comes from `-N`, `-n`, and `--ntasks-per-node`.
For bare-metal `mpirun`, it comes from `-np`, the cluster node list, and hostfile
slots. If the MPI ranks cannot be divided evenly across those nodes, the check is
recorded as `skipped` with a reason because this comparison supports uniform
launches. An empty result set, or a result set in which every row omits all topology
fields, is also recorded as `skipped`, including in `strict` mode. Partial topology
blocks and topology that appears or disappears between rows still produce
mismatches. Malformed JSON result structures fail result loading independently of
the topology-check mode; CVS expects an array of result objects.

The existing `rccl_test_params.threads_per_gpu` key has two meanings:

- Performance runs pass it as `-g`, the number of GPUs per thread, with the default
  of one thread per MPI rank.
- Regression runs pass it as `-t`, the number of threads per MPI rank, with the
  default of one GPU per thread because they omit `-g`.

The rccl-tests reporter sets `gpusPerRank` to `nThreads * nGpus`, as shown in
[`common.cu`](https://github.com/ROCm/rocm-systems/blob/3ae53f79fc26ba4d79c3dfa16ab71065760608e8/projects/rccl-tests/src/common.cu#L1348).
The expected total is therefore `threads_per_gpu` on both paths. For example,
`threads_per_gpu: "8"` means `gpusPerRank=8` for both `-g 8` and `-t 8`.

CVS writes `<result_file_stem>_topology_check.json` beside the result files on the
head node. For example, `/results/rccl.json` produces
`/results/rccl_topology_check.json`. Performance runs write one audit file containing
one check entry per data type. Each `run_regression()` invocation writes one audit
file beside the configured result file. The regression suite currently reuses
`rccl_result_file` across cases, so later cases overwrite the earlier result and
audit files. Callers using `RcclJob` directly can set a distinct `rccl_result_file`
per case to retain every audit. Each check entry contains
`label`, `requested`, `reported` (the first row's topology), `mismatches`, `mode`,
and `verdict` (`pass`, `mismatch`, or `skipped`). Skipped entries include a `reason`;
differences in later rows appear in `mismatches` with their zero-based row index.
Checks collected before an aborted run are still saved.

For a launch requesting 2 nodes, 1 rank per node, and 8 GPUs per rank, affected
rccl-tests builds report `nodes=1`, `ranks=2`, `ranksPerNode=2`, `gpusPerRank=8`.
The audit records `nodes: requested 2, reported 1` and
`ranksPerNode: requested 1, reported 2`. CVS preserves the producer's topology
values in the raw, combined, and aggregated results; the audit carries the
requested values separately. Single-node output without the topology block is
accepted by the result schema, and the topology comparison is recorded as
`skipped` when the block is absent from every row.

The probable producer cause is in rccl-tests `src/common.cu`: the reporter receives
the global MPI communicator size (`args->nProcs`) as `ranksPerNode`, then derives
`nodes` from that value. The identity `ranks == nodes * ranksPerNode` still holds,
so schema validation alone cannot catch it. The upstream fix must pass the
separately computed `localSize`. This is tracked separately from the CVS change;
see [AIMVT-334](https://amd-hub.atlassian.net/browse/AIMVT-334). Use `strict` once
your installed rccl-tests build reports per-node rank counts correctly. Changing
the CVS default and documenting a minimum fixed version are follow-up work.

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
- **Regression cases**: Each combination in the Cartesian product runs separately and reuses the configured result-file path
- **Single env dump**: Only `test_print_env_once` prints the environment; other tests focus on performance
- **Tree filter**: Algorithm restrictions apply to maintain compatibility
