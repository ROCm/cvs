# Aorta benchmark

Aorta runs through CVS's container orchestrator. Use `aorta_single` with exactly one node or
`aorta_distributed` with two or more nodes. Both suites expose container launch, repository
verification, optional RCCL build, benchmark execution, trace collection, optional analysis,
parsing, threshold validation, report generation and teardown as separate pytest tests.
The distributed suite also verifies torchrun and the configured RDMA devices.

## Configure and run

Copy the JSON variants and their sibling threshold file:

```bash
cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_single.json --output ./aorta_single.json
cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_distributed.json --output ./aorta_distributed.json
cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_threshold.json --output ./mi3xx_aorta_profile_overlap_2gpu_threshold.json
```

Replace every `<changeme>` in your chosen variant. Set `paths.shared_fs` to storage available
at the same path on each node, and `gpus_per_node` to the number of GPUs used per node. Local
storage is sufficient; the path does not have to be an NFS filesystem. For distributed runs,
set `multi_node.extra_env.NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` to your fabric interfaces, or
remove these entries when using transport defaults. Keep GPU counts and profiling settings
consistent with Aorta's `config/profile_overlap_2gpu.yaml`, the proof variant migrated from
the previous sample. Adjust that base YAML for your distributed workload as necessary.

Place the Aorta repository at `aorta_path` on every node, or set `aorta_auto_clone: true` and
supply `aorta_clone_url`. The matching writable container mount is explicit in
`container.runtime.args.volumes`. Aorta runs as the container's default user; use storage it
can write. Root-squashed NFS may prevent a root container from writing to the repository.

```bash
cvs run aorta_single --cluster_file cluster-single.json --config_file aorta_single.json
cvs run aorta_distributed --cluster_file cluster-distributed.json --config_file aorta_distributed.json
```

`multi_node.master_launch_mode` defaults to `auto`: one node uses `experiment_script`,
multiple nodes use per-node torchrun processes. Explicit `torchrun` is also supported on one
node through `aorta_single`; `script` requires one node. Rendezvous uses `master_addr` when
set, otherwise the first cluster node's `vpc_ip`, otherwise its node identifier. Omit
`master_port` or set it to zero to choose an available port on that node. Omit
`nproc_per_node` or set it to zero to use `gpus_per_node`.

The shared loader resolves `{user-id}`, references within `paths` such as `{shared_fs}`, and
cross-block references such as `{paths.shared_fs}`. Unknown placeholders and `<changeme>`
markers fail before container launch. The common schema includes `model`, `models_dir` and
`hf_token_file`; this suite does not download models or require a Hugging Face token.

## Outputs and validation

Artifacts are downloaded through the orchestrator's transport to `output_dir/<run-id>/` on
the machine running CVS. Shared storage between that machine and the cluster is unnecessary.
Distributed traces retain the parser layout
`combined_traces/node_<rank>/<original-output>/torch_profiler/`. Single-node collection keeps
the newest fresh profiler tree. Files older than each node's recorded benchmark start are
excluded, as are prior `combined_traces` and CVS working directories. Setting
`multi_node.collect_traces: false` collects only the head's latest tree, retaining the legacy
opt-out; use the default `true` for complete distributed metrics.

Optional TraceLens and GEMM analysis runs in the head container against its original trace
tree. A missing analysis dependency or failed script produces a warning and leaves raw-trace
parsing available. Multi-node metrics always come from the collected raw traces. Single-node
runs try generated Excel reports first and fall back to raw traces when reports contain no
metrics. `analysis.gemm_script` and `analysis.skip_if_exists` are honored independently.

The sibling threshold JSON contains `expected_results` with `max_avg_iteration_ms`,
`min_compute_ratio`, `min_overlap_ratio`, and `max_time_variance_ratio`. The sample preserves
the previous gfx942 starting values; calibrate them for your hardware and workload. Set
`enforce_thresholds: false` for measurement without threshold assertions. A run with no usable
metrics fails parsing.

`aorta_benchmark_report.json` retains cluster, configuration, performance, and per-rank
summary fields. It also records execution status, validation status and trace-collection
errors. Benchmark and RCCL logs are downloaded alongside the report. Surviving nodes' traces
and reports remain available after a distributed failure; threshold checks are skipped once
an earlier stage fails. Each node's kernel journal is checked between its benchmark start
and end using the shared CVS error patterns. Passwordless access to `journalctl -k` through
`sudo -n` is required.

The job stops its own process groups and restores repository ownership before the fixture
handles container teardown. `container.lifetime` retains the shared `per_run`, `persistent`
and `no_launch` semantics. Remote working scripts and logs remain under
`aorta_path/.cvs-aorta/<run-id>/` for diagnostics.

## Migration from the YAML runner

Replace the old Aorta YAML with a JSON variant and sibling threshold file. Move Docker
settings to `container`, environment values to `container.env`, and `expected_results` to
the threshold file. Use `container.runtime.args.ipc: host`; `shm_size` is no longer used.
`skip_rccl_build` still skips the configured build script. When enabled, that script runs on
every node; `rccl_path`, `RCCL_CLONE_URL` and `RCCL_BRANCH` are available in its environment.
The script remains responsible for applying its RCCL checkout/build options.

Execution now follows the shared runtime and transport support. This checkout's Enroot
runtime remains unimplemented; migrating Aorta does not implement that backend.
