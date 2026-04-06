# RCCL Performance Tests

RCCL in CVS uses one suite, `rccl_cvs`, with a strict nested config (`rccl.run`, `rccl.validation`, `rccl.artifacts`) and a canonical run-directory artifact (`run.json`). Benchmark intent stays in JSON; **all install paths** (`RCCL_TESTS_BUILD_DIR`, `ROCM_HOME`, `MPI_HOME`, `RCCL_HOME`, etc.) are set only via **required** `rccl.run.env_script`. NCCL, UCX, plugin, and site-specific tuning belong there too.

The JSON file must have **only** the top-level key `rccl`. Legacy fields (`mode`, `results` / `results_file`, flat shapes) and unknown keys anywhere in the nested schema are rejected. Optional `matrix` is valid only as omitted, `null`, or `{}` until matrix expansion is implemented in this runner.

## Supported suite

Use `rccl_cvs` for both single-node and multi-node runs. Topology is **inferred** from `num_ranks` and `ranks_per_node` (there is no `mode` field).

## Collective tests covered

Select any of these RCCL binaries through `rccl.run.collectives`:

- `all_reduce_perf`
- `all_gather_perf`
- `scatter_perf`
- `gather_perf`
- `reduce_scatter_perf`
- `sendrecv_perf`
- `alltoall_perf`
- `alltoallv_perf`
- `broadcast_perf`

## Prerequisites

1. Provide a valid cluster file such as `input/cluster_file/cluster.json`.
2. Install `rccl-tests` and point `RCCL_TESTS_BUILD_DIR` in `rccl.run.env_script` at the build directory containing the `*_perf` binaries.
3. For multi-node runs (`num_ranks / ranks_per_node > 1`), export `MPI_HOME` in `env_script` (the runner uses `${MPI_HOME}/bin/mpirun`).
4. For `validation.profile` `thresholds` or `strict`, maintain `validation.thresholds` or `validation.thresholds_file` for your platform.

## How to run with CVS

Run from the CVS repo root:

```bash
cvs list rccl_cvs
```

Example:

```bash
cvs run rccl_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl.log -vvv -s
```

Use `num_ranks` equal to `ranks_per_node` for a single-node launch; use a larger `num_ranks` (and enough cluster nodes) for multi-node.

## Config quick reference

Edit `input/config_file/rccl/rccl_config.json` before running:

- `rccl.run`: **required** `env_script`; ranks (`num_ranks`, `ranks_per_node` as integers); benchmark args (`collectives`, `datatype`, size sweep fields, `warmups`, `iterations`, `cycles`). No JSON path fields.
- `rccl.validation`: `profile` (`none` | `smoke` | `thresholds` | `strict`), optional `thresholds` or `thresholds_file`
- `rccl.artifacts`: `output_dir`, `export_raw`

Placeholder handling still works in RCCL config paths, including `{user-id}`.

Example inline thresholds (numeric values; message sizes as decimal strings matching rccl-tests `size`):

```json
"thresholds": {
  "all_reduce_perf": {
    "bus_bw": {
      "8589934592": 330.0,
      "17179869184": 350.0
    }
  }
}
```

## Optional AINIC/ANP setup

If using AINIC + ANP:

1. Ensure ANP is installed on all target nodes.
2. Edit `input/config_file/rccl/ainic_env_script.sh` with your ANP path.
3. Point `rccl.run.env_script` at that file.

## Output artifact

`rccl_cvs` writes `{output_dir}/{run_id}/run.json`. **`run.json` is the only normative machine-readable result** (`schema_version` `rccl_cvs.run.v1`); there is no separate `summary.json`. The layout is described in `docs/reference/configuration-files/rccl.rst` (Result artifact section). Contents include topology, cluster nodes, echoed config, filtered env vars after `env_script`, one `cases[]` entry per collective (no-matrix path uses `case_id` values `c0_<collective>`, `c1_<collective>`, …), `metrics.rows`, and `summary`. If `export_raw` is true, optional raw rccl-tests JSON is stored per case under `{output_dir}/{run_id}/raw/<case_id>.json` for debugging only.
