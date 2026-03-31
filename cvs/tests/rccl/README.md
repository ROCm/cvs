# RCCL Performance Tests

RCCL in CVS now uses one suite, `rccl_cvs`, with one small config surface and one consolidated JSON artifact. The RCCL config keeps benchmark intent only; NCCL, UCX, plugin, and site-specific environment tuning should live in the user-provided `env_script`.

## Supported suite

Use `rccl_cvs` for both single-node and multi-node runs.

## Collective tests covered

Select any of these RCCL binaries through `collectives`:

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
2. Install `rccl-tests` and set `rccl_tests_dir` in the config.
3. For multi-node runs, set either `mpi_root` or `mpirun_path`.
4. Keep runtime tuning in `env_script` so the benchmark command stays minimal.
5. Update `results` thresholds for your platform before using them for pass/fail decisions.

## How to run with CVS

Run from the CVS repo root:

```bash
cvs list rccl_cvs
```

Multi-node example:

```bash
cvs run rccl_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl.log -vvv -s
```

Single-node example:

```bash
cvs run rccl_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_single.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_single.log -vvv -s
```

Switch between the two paths by changing `mode` in the config.

## Config quick reference

Edit `input/config_file/rccl/rccl_config.json` before running:

- Install paths: `rccl_tests_dir`, `mpi_root` or `mpirun_path`, `rocm_path`, `env_script`
- Mode and scale: `mode`, `num_ranks`, `ranks_per_node`
- Benchmark args: `collectives`, `datatype`, `start_size`, `end_size`, `step_factor`, `warmups`, `iterations`, `cycles`
- Validation: `verify_bus_bw`, `verify_bw_dip`, `verify_lat_dip`, `results` or `results_file`
- Artifact: `output_json`

Placeholder handling still works in RCCL config paths, including `{user-id}`.

Example threshold format:

```json
"results": {
  "all_reduce_perf": {
    "bus_bw": {
      "8589934592": "330.00",
      "17179869184": "350.00"
    }
  }
}
```

## Optional AINIC/ANP setup

If using AINIC + ANP:

1. Ensure ANP is installed on all target nodes.
2. Edit `input/config_file/rccl/ainic_env_script.sh` with your ANP path.
3. Point `env_script` at that file.

## Output artifact

`rccl_cvs` writes one JSON artifact to `output_json`. It contains run metadata, one entry per collective, parsed rccl-tests rows, and the validation summary for each collective.
