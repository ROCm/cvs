# RCCL Performance Tests

RCCL (ROCm Communication Collectives Library) tests validate distributed GPU communication performance across AMD GPU clusters. These tests run RCCL collectives through CVS, compare results against optional thresholds, and generate HTML artifacts for review.

## Supported test suites

CVS currently provides these RCCL suites:

1. `rccl_multinode_cvs`  
   Multinode sweep across collective, algorithm, protocol, queue-pair scaling (`qp_scale`), and PXN toggle (`nccl_pxn_disable`).

2. `rccl_multinode_default_cvs`  
   Multinode collectives using RCCL default algorithm/protocol behavior.

3. `rccl_singlenode_cvs`  
   Single-node collective performance and threshold checks.

4. `rccl_heatmap_cvs`  
   Multinode matrix sweep over collective, `gpu_count_list`, `data_type_list`, and `channel_config_list`, then builds a heatmap against a golden reference JSON.

All suites also collect host/network info and validate that firewall services are not blocking the run.

## Collective tests covered

These RCCL binaries can be selected through `rccl_collective`:

- `all_reduce_perf`: all ranks reduce and receive the reduced value.
- `all_gather_perf`: each rank gathers data from all other ranks.
- `scatter_perf`: one rank scatters data to all ranks.
- `gather_perf`: all ranks gather data to one root rank.
- `reduce_scatter_perf`: reduction followed by shard distribution.
- `sendrecv_perf`: point-to-point send/receive performance.
- `alltoall_perf`: all ranks exchange equal-sized payloads.
- `alltoallv_perf`: all ranks exchange variable-sized payloads.
- `broadcast_perf`: one rank broadcasts data to all ranks.

## Prerequisites

1. **Cluster file**: Provide a valid cluster file (for example `input/cluster_file/cluster.json`) with node definitions, username, key path, and reachable node IPs.
2. **RCCL test binaries**: Install `rccl-tests` and ensure paths in config are correct (`rccl_dir`, `rccl_tests_dir`, `rccl_path_var`).
3. **MPI/ROCm paths**: Ensure `mpi_dir`, `mpi_path_var`, and `rocm_path_var` in config match your environment.
4. **Passwordless SSH**: Nodes should be able to run distributed MPI launch commands without interactive prompts.
5. **Expected results**: Update `results` thresholds for your hardware and cluster size (defaults are sample values only).

## How to run with CVS

Run from the CVS repo root (directory containing `cvs` and `input`):

### List available RCCL suites

```bash
cvs list rccl_multinode_cvs
cvs list rccl_multinode_default_cvs
cvs list rccl_singlenode_cvs
cvs list rccl_heatmap_cvs
```

### Multinode parameter sweep

```bash
cvs run rccl_multinode_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_multinode.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_multinode.log -vvv -s
```

### Multinode (RCCL defaults)

```bash
cvs run rccl_multinode_default_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_multinode_default.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_multinode_default.log -vvv -s
```

### Single-node run

```bash
cvs run rccl_singlenode_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/single_node_mi355_rccl.json \
  --html=/var/www/html/cvs/rccl_singlenode.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_singlenode.log -vvv -s
```

### Heatmap run

```bash
cvs run rccl_heatmap_cvs \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_heatmap.html --capture=tee-sys --self-contained-html \
  --log-file=/tmp/rccl_heatmap.log -vvv -s
```

## Parameter quick reference

Edit `input/config_file/rccl/rccl_config.json` (and `single_node_mi355_rccl.json` for single-node baseline) before running:

- **Scale and topology**: `no_of_nodes`, `no_of_global_ranks`, `no_of_local_ranks`, `ranks_per_node`.
- **Sweep controls**:
  - `rccl_collective`, `rccl_algo`, `rccl_protocol`, `qp_scale`, `nccl_pxn_disable`
  - `gpu_count_list`, `data_type_list`, `channel_config_list` (used by `rccl_heatmap_cvs`)
- **Message sweep**: `start_msg_size`, `end_msg_size`, `step_function`, `warmup_iterations`, `no_of_iterations`, `no_of_cycles`.
- **Network and transport**: `ib_hca_list`, `net_dev_list`, `oob_port`, `gid_index`, `nccl_socket_ifname`, `ucx_tls`, `mpi_pml`.
- **Validation controls**: `verify_bus_bw`, `verify_bw_dip`, `verify_lat_dip`, `results`.
- **Artifacts/reference**: `rccl_result_file`, `golden_reference_json_file`, `output_dir`, `heatmap_title`.

Additional tuning parameters used in multinode runs:

- **Runtime and verbosity**: `threads_per_gpu`, `debug_level`, `cluster_snapshot_debug`.
- **IB/UCX tuning**: `nccl_ib_timeout`, `ib_rx_queue_len`, `ucx_tls`, `hcoll_enable_mcast_all`.
- **RCCL/NCCL behavior**: `nccl_cumem_enable`, `nccl_ib_sl`, `nccl_ib_tc`, `nccl_ib_split_data_on_qps`, `nccl_net_plugin`.
- **Heatmap-specific validation**: `nic_model` and `golden_reference_json_file`.

Placeholder handling:

- `{user-id}` and other supported placeholders in cluster/config files are resolved at runtime by CVS.
- You can keep placeholders or replace them with explicit absolute paths.

Expected-results format (`results`):

- Thresholds are keyed by collective and message size in bytes.
- Typical structure:

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

## Optional AINIC/ANP net-plugin setup

If using AINIC + ANP plugin:

1. Ensure AMD ANP is installed and accessible on all target nodes.
2. Edit `input/config_file/rccl/ainic_env_script.sh` and set `ANP_HOME_DIR` to your absolute ANP install path.
3. Set `env_source_script` in your RCCL JSON config to that script path.
4. Run RCCL with `cvs run ...`; CVS sources the script automatically before test execution.

If you are not using AINIC/ANP, set `env_source_script` to your standard environment setup script (or `"None"` to skip script sourcing).

## Expected output artifacts

- Pytest HTML summary from `--html`.
- RCCL perf graph HTML under `/tmp/rccl_perf_report_*.html` or `/tmp/rccl_singlenode_perf_report_*.html`.
- Heatmap output under `/tmp/rccl_heatmap_*.html` plus JSON result files.
- Optional copied heatmap artifacts under `output_dir` (from config) for easier sharing.
- Graph/heatmap files under `/tmp` can be copied to your web server path (for example `/var/www/html/cvs`) for browser viewing.

The run passes when command execution succeeds and configured validations (`results`, dip checks, and optional bus bandwidth checks) are met.
