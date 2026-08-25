# Worked transform examples

All examples assume the user has provided a HuggingFace model repo ID. Replace `<user-hf-model>` with that value in model fields and derive output filenames from it.

## PyTorch XDit WAN I2V: single → distributed

**Start from:** closest shipped WAN single template under `cvs/input/config_file/inference/pytorch_xdit/` (match GPU arch).

**Output name:** `{gpu}_pytorch_xdit_{user-model-slug}_distributed.json`

### `config` changes

| Field | single | distributed |
|-------|--------|-------------|
| `container_name` | workload-specific | add `-dist` suffix |
| `model_repo` | `<user-hf-model>` | `<user-hf-model>` or local path on all nodes |
| `model_rev` | commit hash if needed | `""` when using local path |
| `nnodes` | omit | `2` |
| `master_addr` | omit | `"<changeme>"` |
| `master_port` | omit | `29500` |
| `nccl_ib_hca` | omit | `"<changeme>"` |
| `nccl_socket_ifname` | omit | `"<changeme>"` |
| `gloo_socket_ifname` | omit | `"<changeme>"` |
| `nccl_ib_gid_index` | omit | `3` |
| `nccl_debug` | omit | `"INFO"` |
| `container_config.env_dict` | `{}` | `{"NCCL_PROTO": "Simple"}` |

### `benchmark_params.wan22_i2v_a14b` changes

| Field | single | distributed (2 nodes) |
|-------|--------|----------------------|
| `ulysses_size` | 8 (default) | `8` |
| `ring_size` | 1 (default) | `2` |
| `expected_results.auto.max_avg_total_time_s` | baseline | relax ~20% |
| `expected_results.<gpu>.max_avg_total_time_s` | baseline | relax ~20% |

Preserve: `prompt`, `size`, `frame_num`, `num_benchmark_steps`, `compile`, `torchrun_nproc`.

**Run:**
```bash
cvs run pytorch_xdit_wan22_14b_distributed \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/{gpu}_pytorch_xdit_{user-model-slug}_distributed.json
```

---

## PyTorch XDit Flux T2I: single → distributed

**Start from:** closest shipped Flux single template (match GPU arch).

### `config` changes

Same NCCL block as WAN distributed (see above). Set `model_repo` to `<user-hf-model>`.

### `benchmark_params.flux1_dev_t2i` changes

| Field | single | distributed (2 nodes) |
|-------|--------|----------------------|
| `ring_degree` | `1` | `2` |
| `ulysses_degree` | `8` | `8` |
| `expected_results.*.max_avg_pipe_time_s` | lower | raise ~20–25% |

**Run:**
```bash
cvs run pytorch_xdit_flux1_dev_distributed \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/{gpu}_pytorch_xdit_{user-model-slug}_distributed.json
```

---

## vLLM: single → distributed (reference)

**Start from:** closest shipped single template for the target GPU arch and precision.

1. Copy to `{gpu}_vllm_{user-model-slug}_{prec}_distributed_config.json` and matching threshold sibling.
2. Set `threshold_json` to the new threshold filename.
3. Replace model path/id with `<user-hf-model>`.
4. `params.pipeline_parallel_size`: `"1"` → `"2"`
5. `params.nnodes`: `"1"` → `"2"`
6. Add `master_addr`, `master_port`, `roles.server.ib_*`.
7. Threshold keys: append `,PP=2`.

---

## SGLang: distributed (reference)

**Start from:** closest shipped distributed template for the target GPU arch (e.g. `mi30x_sglang_*_distributed.json` or `mi35x_sglang_distributed.json` for ainic/MI355X).

**Output:** config + threshold siblings, e.g.:
- `{gpu}_sglang_{user-model-slug}_distributed.json`
- `{gpu}_sglang_{user-model-slug}_distributed_threshold.json`

Set `benchmark_params.<workload>.model` to `<user-hf-model>`. Set `benchmark_params.<name>.threshold_file` to the threshold filename. Adapt threshold keys from the source template's threshold file structure.

---

## SGLang: single → disaggregated (reference)

**Start from:** closest shipped single template for the target GPU arch.

**Reference:** matching disaggregated template for the same GPU arch.

Set `benchmark_params.<workload>.model` to `<user-hf-model>`. Add to `config`:
- `prefill_node_list`, `decode_node_list`
- `proxy_router_node`, `benchmark_serv_node`
- `prefill_coordinator_addr`, `decode_coordinator_addr`
- prefill/decode ports and `pipeline_parallelism: "2"` in benchmark params

Disaggregated is SGLang-only in CVS today.
