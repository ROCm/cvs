# RCCL validation

CVS provides three RCCL suites:

- `rccl_perf` runs the performance collectives listed in the suite, across `rccl_test_params.data_types`, and generates a performance report.
- `rccl_regression` runs the collectives in `rccl_test_params.rccl_collective` across the configured `regression` environment combinations. The older `rccl.rccl_collective` key remains a fallback.
- `rccl_pairwise` checks the reference node, tests each reference/candidate pair, then builds an incremental group from the successful candidates. Phase 1 requires a clean run; Phase 2 also applies `cvs_params.pairwise_min_bw` when greater than zero.

Host and network diagnostics use the host OS. Perf and regression retain their time-bounded dmesg checks when passwordless sudo is available. RCCL execution and cleanup use the orchestrator's execution environment.

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

Chart x-axis labels are humanized (`1K`, `1M`, `1G`); the results table and
`rccl_run_deck.json` keep the raw byte size so external tooling can still
sort or filter on it numerically. On large clusters, `rccl_pairwise` can
produce one series per node pair — each chart card renders at most 40 series
(configurable per card via the profile's `max_series`) and shows a "Showing N
of M" banner when truncated; the full set always remains in the results
table and JSON export.

### Reported vs requested topology

CVS compares the `nodes`, `ranks`, `ranksPerNode`, and `gpusPerRank` in rccl-tests
JSON with the topology requested by the launch command. This check runs for every
performance data type and every regression case, including pairwise performance
runs. It also detects topology changes between rows in the same result file.

Set `rccl.cvs_params.topology_check` in your config:

| Mode | Behaviour |
| --- | --- |
| `warn` (default) | Log a mismatch banner and record the differences; topology mismatches alone do not fail the test. |
| `strict` | Record the differences and call `fail_test` directly for that test case. Result collection continues, and pairwise runs are marked unclean. |
| `off` | Disable the requested-vs-reported comparison. Result parsing and schema validation remain active, and aggregation still rejects rows whose topology disagrees. The audit file contains the mode and an empty check list. |

Older configs that omit the setting use `warn`. Mode names are case insensitive;
unrecognised values produce a warning and fall back to `warn`.

For Slurm/Spur, requested topology comes from `-N`, `-n`, and `--ntasks-per-node`.
For bare-metal `mpirun`, it comes from `-np`, the cluster node list, and hostfile
slots. If the MPI ranks cannot be divided evenly across those nodes, the check is
recorded as `skipped` with a reason because this comparison supports uniform
launches. An empty result set, or a result set in which every row omits all topology
fields, is also recorded as `skipped`, including in `strict` mode. A row that omits
topology entirely is not itself a mismatch as long as another row reports topology
matching the request; fields absent from every row still mismatch against what was
requested, and a field present in both the baseline row and a later row with a
different value produces a mismatch. Rows are only compared against the baseline,
not against each other, so two non-baseline rows can still disagree on a field the
baseline never reported. Malformed JSON result structures fail result loading
independently of the topology-check mode; CVS expects an array of result objects.

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
`label`, `requested`, `reported` (the first row that contains any topology field, not
necessarily row 0), `mismatches`, `mode`, and `verdict` (`pass`, `mismatch`, or
`skipped`). Skipped entries include a `reason`; differences in later rows appear in
`mismatches` with their zero-based row index.
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

`heatmap` is a standalone result-comparison suite, reusable beyond RCCL-only
flows. Discover it with `cvs list heatmap`, then compare two result JSONs:

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

Start from [`rccl_config.json`](../../input/config_file/rccl/rccl_config.json). Replace any `<changeme>` values and configure paths, interfaces, node counts, and reference bandwidths for your cluster.

- `mpi_params.no_of_nodes` and `no_of_local_ranks` set the MPI topology explicitly. For a single-node run, set `no_of_nodes` to `1` and provide a one-node cluster or allocation.
- `mpi_params.mpi_dir` names the Open MPI installation prefix; a trailing `/bin` is accepted. `mpi_pml`, `mpi_oob_port`, `net_dev_list`, and `ucx_tls` control Open MPI/UCX discovery and initialization.
- `rccl_test_params.rccl_tests_dir` names the directory containing the collective binaries. Message sizes, iteration counts, and optional `rccl_timeout`/`output_algo_proto_channels` flags belong in this object.
- `env_source_script` names one existing script, available at the same path on every participating node or container. Use `/dev/null` or `"none"` when no script is needed. Regression overrides are applied after sourcing it.
- `cvs_params.cvs_exec_timeout` caps the outer collective launch (default 2400 seconds).

Starter environment scripts are in [`cvs/input/env_file/rccl/`](../../input/env_file/rccl/): `ainic_env_script.sh`, `thor2_env_script.sh`, and `cx7_env_script.sh`. Adapt the appropriate script to your fabric. Put RCCL/NCCL/HSA/plugin tuning in it. CVS supplies MPI/PMIx/ORTE settings from `mpi_params`, including on managed runs.

### Expected results

Place `results` directly under `rccl`, alongside `cvs_params`. Both perf and regression read it. Older `cvs_params.results` is accepted only when the top-level key is absent; an explicit `results: {}` disables references.

For thresholds that depend on NIC, data types, and MPI rank count, use this format:

```json
"results": {
  "thor": {
    "all_reduce_perf-float-16": {
      "8589934592": {"bus_bw": "<changeme>"},
      "17179869184": {"bus_bw": "<changeme>"}
    }
  }
}
```

NIC keys are `ainic`, `thor`, and `connectx`, selected from `cvs_params.nic_model`. The result key is `<collective>-<data_types joined with underscores>-<total MPI ranks>`. Perf uses its configured data types in order; regression uses `float`, the rccl-tests default for its command. Pairwise uses the rank count of the current subset. A NIC/rank reference is used only when its full key matches.

Both suites also accept the shipped legacy flat format:

```json
"results": {
  "all_reduce_perf": {
    "bus_bw": {"8589934592": "<changeme>", "17179869184": "<changeme>"}
  }
}
```

A matching NIC/rank reference takes precedence over a flat reference. Flat references apply regardless of data type or rank count, so calibrate them for every topology you test. The sample numbers are examples for two nodes, not qualification targets for other clusters. Pairwise/incremental sub-runs drop flat references entirely, since a threshold calibrated for the full cluster does not apply at pairwise node counts; only NIC/rank-keyed references, already scoped by rank count, are honored there.

Set `cvs_params.verify_bus_bw` to `"True"` to enforce bandwidth thresholds. Measurements below 95% of the configured reference fail. The existing bandwidth/latency dip checks use reference message sizes and their existing `verify_bw_dip`/`verify_lat_dip` switches. With no matching reference, these checks have no reference data to validate.

### Regression combinations

A non-empty `regression` object is required for parametrization:

```json
"regression": {
  "NCCL_ALGO": ["Ring", "Tree"],
  "NCCL_PROTO": ["Simple"],
  "NCCL_IB_QPS_PER_CONNECTION": ["1", "2"],
  "NCCL_MIN_NCHANNELS": ["8", "16"],
  "NCCL_MAX_NCHANNELS": ["8", "16"]
}
```

CVS takes the Cartesian product of the environment axes and configured collectives. `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` must both be present or both absent, and their lists must have equal lengths: channel values are paired by position. Choose algorithm/collective combinations supported by your RCCL build.

### Result storage

`cvs_params.rccl_result_file` defaults to `{run_dir}/rccl_result_file.json`. CVS resolves `{run_dir}` under `<workspace>/cvs_runs/<run_id>/`; managed jobs use the scheduler job ID. Perf also writes files with data-type and `_aggregated` suffixes. Regression writes to the configured result path. Successive collective runs reuse these names, so preserve JSON files separately when comparing individual cases. Pairwise runs one `all_reduce_perf` job per phase/candidate pair against this same path, so each sub-run's combined/aggregated JSON overwrites the previous one — only the last phase's files survive on disk. Pass/fail is unaffected, since pairwise verifies each sub-run's in-memory result immediately after it runs; copy the files out between sub-runs if you need every phase's artifacts.

Every run — managed or not, bare metal or container — requires the result directory to be writable and reachable through the same head-host transfer interface RCCL uses to save results. During preparation, CVS writes a unique sentinel through the head-node orchestrator, then reads it back with `download_from_head` and checks the round trip. An unavailable directory fails before launching RCCL and names `--workspace`/`CVS_WORKSPACE` in the error. This checks the result path between CVS and the head; it does not certify every mount on every allocated node.

Under the container backend, bind-mount the result directory into the container **at the same absolute path as on the host**. The sentinel command runs inside the head container, so preparation rejects a container path that is not visible at the matching host path. RCCL launches inside the execution environment, while result upload/download uses the orchestrator's head-host transfer interface. Managed HTTP transfers are local shared-filesystem copies. Combined or aggregated result-save failures are reported as test failures.

## Running on bare metal

Provide a cluster file with SSH keys and reachable MPI addresses, and a configured RCCL JSON:

```bash
cvs run rccl_perf \
  --cluster_file /path/to/cluster.json \
  --config_file /path/to/rccl_config.json \
  --workspace /path/to/cvs-workspace \
  --html /path/to/reports/rccl_perf.html --self-contained-html \
  --log-file /path/to/reports/rccl_perf.log --capture=tee-sys -vvv -s
```

Repeat with `rccl_regression` and `rccl_pairwise`. The launcher uses `mpirun`. The container backend launches it inside the head container and supplies the orchestrator's SSH port (2224 for containers); passwordless container-to-container SSH must be available. MPI, RCCL binaries, and the environment script must be usable in that environment.

## Running under SPUR / Slurm

Run inside a scheduler allocation with **one CVS task per node**. An allocation shell alone does not start the CVS agents. Rank 0 runs pytest; other ranks serve command execution through HTTP agents. CVS launches each RCCL workload as a nested PMIx step with the configured GPU/MPI rank count.

For SPUR, inside an allocation:

```bash
spur run --mpi=none -N 2 --ntasks-per-node 1 -- \
  cvs run rccl_perf \
    --config_file /shared/user/rccl_config.json \
    --workspace /shared/user/cvs \
    --html /shared/user/reports/rccl_perf.html --self-contained-html \
    --log-file /shared/user/reports/rccl_perf.log --capture=tee-sys -vvv -s
```

For Slurm, use the same CVS arguments with:

```bash
srun --mpi=none -N 2 --ntasks-per-node 1 -- \
  cvs run rccl_perf \
    --config_file /shared/user/rccl_config.json \
    --workspace /shared/user/cvs \
    --html /shared/user/reports/rccl_perf.html --self-contained-html \
    --log-file /shared/user/reports/rccl_perf.log --capture=tee-sys -vvv -s
```

Use paths and node counts appropriate to your allocation. Repeat for `rccl_regression` and `rccl_pairwise`, with distinct HTML/log names. Configure `SPUR_CONTROLLER_ADDR` as required by your cluster's Spur setup.

- `--workspace` or `CVS_WORKSPACE` must name writable shared storage on all participating nodes. Specify it explicitly inside the CVS container image, where the default venv parent is not shared. Mount both the workspace and result path consistently.
- `--cluster_file` is optional for a managed run: CVS synthesizes a job-owned cluster file with each node's hostname as its `vpc_ip`. Supply an appropriate cluster config when selecting the container orchestrator.
- Scheduler detection checks SPUR before Slurm. Override it with `CVS_SCHEDULER=spur` or `CVS_SCHEDULER=slurm` if needed; an override does not replace the job-step requirement.
- Nested launches use `spur run --overlap --mpi=pmix` or `srun --overlap --mpi=pmix`. Managed container setup skips sshd because the scheduler uses PMIx.
- The SPUR pairwise/incremental guard remains enabled: Spur 0.11 ignores `--nodelist` on nested steps, preventing reliable subset execution. Both pairwise workload tests skip with this reason; host/network collection still runs. Check the deployed Spur version and actual subset behavior before changing the guard. Slurm subset steps use `-w`.

## Reports and validation artifacts

Perf and regression add their charts to the pytest HTML report. CVS writes a timestamped suite ZIP beside the HTML report. Pairwise also records phase membership in `cvs_params.pairwise_results_file`; choose a persistent path if needed.

For cluster validation, retain before/after bare-metal result JSONs and HTML bundles, record each managed suite's pass/fail/skip outcome, and check that results can be read from shared storage. SPUR pairwise validation must record the observed version and subset behavior. Upload report bundles outside CVS, for example:

```bash
aws s3 cp /path/to/rccl_perf_timestamp.zip s3://<bucket>/AIMVT-344/
```

## Library entry point

```python
from cvs.lib.rccl_lib import RcclJob

job = RcclJob.from_config(
    orch, collective, rccl_config, node_list, vpc_node_list,
    env_overrides={"NCCL_ALGO": "Ring"},
)
results = job.run_perf()
```

Pass the inner `rccl` config object after resolving placeholders. `run_regression()` is the alternate workflow. The API now takes a single orchestrator; callers of the previous two-handle signature must migrate. `OpenMPI.prepare`, `MpiRun`, and `Srun` also consume that orchestrator.
