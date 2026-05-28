# TorchTitan Training Test Guide

## Overview

TorchTitan is PyTorch's native distributed training framework integrated into AMD's Primus ecosystem.
This guide covers everything needed to run, configure, and extend the TorchTitan CVS training tests.

### Key Differences from Megatron-LM

| Aspect | Megatron-LM | TorchTitan |
|--------|-------------|------------|
| Container image | `rocm/megatron-lm:*` | `rocm/primus:*` |
| Working directory | `/workspace/Megatron-LM` | `/workspace/torchtitan` |
| Config format | CLI arguments | TOML file via `$CONFIG_FILE` env var |
| Launch command | `bash train_llama3.sh` | `torchrun ... -m torchtitan.train` |
| Parallelism config | `tensor_parallelism`, `fsdp` | `data_parallel_shard_degree`, `tensor_parallel_degree` |
| Result metrics | `throughput_per_gpu`, `tokens/GPU/s` | `tok/s` (aggregate), `loss` |
| Rendezvous port | dynamic | fixed `29500` |
| Rdzv ID | not set | `101` |

---

## File Structure

```
cvs/
├── tests/training/torchtitan/
│   ├── __init__.py
│   ├── torchtitan_llama3_1_8b_single.py       # Single-node Llama 3.1 8B test
│   ├── torchtitan_llama3_1_70b_single.py      # Single-node Llama 3.1 70B test
│   ├── torchtitan_llama3_1_8b_distributed.py  # Multi-node Llama 3.1 8B test
│   ├── torchtitan_llama3_1_70b_distributed.py # Multi-node Llama 3.1 70B test
│   └── TORCHTITAN_TEST_GUIDE.md               # This file
│
├── lib/
│   └── torchtitan_training_lib.py             # TorchTitanTrainingJob class
│
└── input/config_file/training/torchtitan/
    ├── mi3xx_torchtitan_llama_single.json      # Config template: single-node
    └── mi3xx_torchtitan_llama_distributed.json # Config template: multi-node
```

---

## Configuration

### Cluster File

Reuse the standard cluster file used by other training tests. It defines:
- `username` — SSH username
- `priv_key_file` — path to SSH private key
- `node_dict` — map of node hostnames/IPs

### Config File — Single Node (`mi3xx_torchtitan_llama_single.json`)

| Field | Description | Required |
|-------|-------------|----------|
| `container_image` | Docker image, e.g. `rocm/primus:v26.2` | Yes |
| `container_name` | Name for the running container | Yes |
| `nnodes` | Number of nodes (set to `1` for single-node) | Yes |
| `training_iterations` | Number of training steps | Yes |
| `hf_token_file` | Path to file containing HuggingFace token | Yes |
| `data_cache_dir` | Cache directory (local path is fine for single-node) | Yes |
| `log_dir` | Where training logs are written | Yes |
| `scripts_dir` | Where wrapper scripts are created | Yes |
| `rocm_dir` | ROCm path — leave empty `""` for auto-detection | No |
| `verify_network_errors` | `"True"` or `"False"` — check RDMA/NIC counters after training | No |
| `container_config.device_list` | GPU devices to pass through: `["/dev/dri", "/dev/kfd"]` | Yes |
| `container_config.volume_dict` | Bind mounts into the container | Yes |

**Model params** (`model_params.single_node.<model_name>.<gpu_type>`):

| Field | Description | Default |
|-------|-------------|---------|
| `tokenizer_path` | HuggingFace model ID for tokenizer download | `meta-llama/Llama-3.1-70B` |
| `model_size` | Model size string, e.g. `"8b"`, `"70b"` | `"70b"` |
| `global_batch_size` | Total batch size across all GPUs | `"128"` |
| `micro_batch_size` | Per-GPU micro batch size | `"2"` |
| `sequence_length` | Token sequence length | `"8192"` |
| `data_parallel_shard_degree` | FSDP2 sharding degree | `"8"` |
| `tensor_parallel_degree` | Tensor parallelism degree | `"1"` |
| `pipeline_parallel_degree` | Pipeline parallelism degree | `"1"` |
| `context_parallel_degree` | Context parallelism degree | `"1"` |
| `expert_parallel_degree` | Expert parallelism (MoE models) | `"1"` |
| `activation_checkpointing` | `"selective"` or `"full"` | `"selective"` |
| `compile` | Enable `torch.compile` (`"true"`/`"false"`) | `"false"` |
| `enable_float8` | Enable Float8 training (`"true"`/`"false"`) | `"true"` |
| `result_dict.tokens_per_sec` | Minimum acceptable `tok/s` threshold | Yes |
| `result_dict.loss` | Minimum acceptable loss threshold | Yes |

### Config File — Multi-Node (`mi3xx_torchtitan_llama_distributed.json`)

Contains all the fields above plus these additional network fields:

| Field | Example | Description |
|-------|---------|-------------|
| `master_address` | `"10.0.0.1"` | Head node IP for rendezvous |
| `nic_type` | `"thor2"`, `"cx7"`, `"broadcom"` | NIC vendor type for setup workarounds |
| `nccl_ib_hca_list` | `"bnxt_re0,bnxt_re1,..."` | RDMA devices for NCCL |
| `nccl_ib_hca` | same as above | Alternate NCCL HCA env var |
| `nccl_socket_ifname` | `"ens51f1np1"` | Network interface for NCCL |
| `gloo_socket_ifname` | `"ens51f1np1"` | Network interface for Gloo |
| `nccl_ib_gid_index` | `"3"` | GID index (auto-set to `3` for Broadcom) |
| `nccl_debug` | `"ERROR"` | NCCL log level |
| `verify_network_errors` | `"True"` | Compare RDMA/ethtool counters before/after training |

Multi-node `container_config` must also include InfiniBand devices:
```json
"device_list": ["/dev/dri", "/dev/kfd", "/dev/infiniband/rdma_cm"],
"volume_dict": {
    "/dev/infiniband": "/dev/infiniband",
    "/usr/local/lib/libbnxt_re-rdmav34.so": "/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host"
}
```

---

## How to Run

### Single-Node Tests
```bash
# Llama 3.1 8B single-node
cvs run torchtitan_llama3_1_8b_single \
    --cluster_file /path/to/cluster.json \
    --config_file /path/to/mi3xx_torchtitan_llama_single.json

# Llama 3.1 70B single-node
cvs run torchtitan_llama3_1_70b_single \
    --cluster_file /path/to/cluster.json \
    --config_file /path/to/mi3xx_torchtitan_llama_single.json
```

### Multi-Node Tests
```bash
# Llama 3.1 8B distributed
cvs run torchtitan_llama3_1_8b_distributed \
    --cluster_file /path/to/cluster.json \
    --config_file /path/to/mi3xx_torchtitan_llama_distributed.json

# Llama 3.1 70B distributed
cvs run torchtitan_llama3_1_70b_distributed \
    --cluster_file /path/to/cluster.json \
    --config_file /path/to/mi3xx_torchtitan_llama_distributed.json
```

Copy sample config templates locally first:
```bash
cvs copy-config
```

---

## Test Execution Flow

### Single-Node Test Steps
```
test_cleanup_stale_containers
    └── Kill existing container by name
    └── Delete all containers and volumes

test_launch_torchtitan_containers
    └── docker run with device_list and volume_dict
    └── shm_size = 128G, timeout = 20 min

test_llama_3_1_<size>_single_node
    └── TorchTitanTrainingJob.__init__()
        └── Resolve all config defaults
        └── Detect ROCm path (auto or from config)
        └── Determine tt_module + tt_config from model name/size
        └── Recreate scripts_dir on nodes
    └── exec_nic_setup_scripts()   [no-op for single-node]
    └── build_training_job_cmd()
        └── Set env vars: HF_TOKEN, HSA_FORCE_FINE_GRAIN_PCIE, CONFIG_FILE
        └── Download tokenizer via download_hf_assets.py
        └── Build single-node torchrun command (see below)
        └── Write to single_node_wrapper_script.sh
    └── start_training_job()
        └── mkdir log dirs inside container
        └── Execute wrapper script inside container
        └── sleep 50s
    └── poll_for_training_completion()
        └── sleep 80s initial
        └── Poll loop (iterations + 10 cycles):
            └── scan_for_training_errors() — checks NCCL/GPU/torch errors
            └── grep log for "step: {iterations}"
            └── On completion: check for NaN/Inf, extract results
    └── verify_training_results()
        └── Check results dict not empty
        └── Check for NaN/Inf in all metrics
        └── verify_dmesg_for_errors()
        └── Compare tok/s and loss against thresholds
```

### Multi-Node Additional Steps
- `test_disable_firewall` runs **first** (before cleanup) — stops `ufw` on all nodes
- `exec_nic_setup_scripts()` — for Broadcom/Thor NICs, copies libbnxt_re and validates `ibv_devinfo`
- Per-node wrapper scripts are created: `distributed_wrapper_script_{i}.sh`
- If `verify_network_errors = "True"`: RDMA and ethtool counters are collected before and after training and compared

---

## Generated torchrun Commands

### Single Node
```bash
cd /workspace/torchtitan
export HF_TOKEN=<token>
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export CONFIG_FILE="./torchtitan/models/<tt_module>/train_configs/<tt_config>"
python scripts/download_hf_assets.py --repo_id <tokenizer_repo> --assets --all tokenizer --hf_token=<token>
torchrun --nnodes 1 --node_rank=0 --nproc_per_node 8 \
    --rdzv_id 101 --rdzv_backend c10d \
    --rdzv_endpoint "127.0.0.1:29500" \
    --role rank --tee 3 \
    -m torchtitan.train --job.config_file $CONFIG_FILE \
    > <log_dir>/torchtitan-logs/out-node0/training.log 2>&1 &
```

### Multi-Node (per node `i` of `N`)
```bash
cd /workspace/torchtitan
export HF_TOKEN=<token>
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export CONFIG_FILE="./torchtitan/models/<tt_module>/train_configs/<tt_config>"
export NCCL_IB_HCA=<nccl_ib_hca_list>
export NCCL_SOCKET_IFNAME=<nccl_socket_ifname>
export GLOO_SOCKET_IFNAME=<gloo_socket_ifname>
export NCCL_DEBUG=ERROR
export NCCL_IB_GID_INDEX=<nccl_ib_gid_index>
python scripts/download_hf_assets.py --repo_id <tokenizer_repo> --assets --all tokenizer --hf_token=<token>
torchrun --nnodes N --node_rank=i --nproc_per_node 8 \
    --rdzv_id 101 --rdzv_backend c10d \
    --rdzv_endpoint "<master_address>:29500" \
    --role rank --tee 3 \
    -m torchtitan.train --job.config_file $CONFIG_FILE \
    > <log_dir>/torchtitan-logs/out-node{i}/training.log 2>&1 &
```

### Model → CONFIG_FILE Mapping

| model_name | tt_module | tt_config |
|------------|-----------|-----------|
| `llama3_1_8b` | `llama3` | `llama3_8b` |
| `llama3_1_70b` | `llama3` | `llama3_70b` |
| `llama3_1_405b` | `llama3` | `llama3_405b` |
| `deepseek_*` | `deepseek_v3` | `deepseek_v3_16b` |
| `qwen*_7b` | `qwen3` | `qwen3_7b` |
| `qwen*_14b` | `qwen3` | `qwen3_14b` |
| `qwen*_32b` | `qwen3` | `qwen3_32b` |

---

## Result Metrics

TorchTitan logs in the format:
```
step: 10, loss: 2.453, tok/s: 12450.3, mem: 58.2 GB
```

The lib extracts three metrics from the last 20 lines of the training log:

| Metric key | Log pattern | Description |
|------------|-------------|-------------|
| `tokens_per_sec` | `tok/s: <float>` | Aggregate tokens/second across all GPUs |
| `loss` | `loss: <float>` | Training loss at final step |
| `mem_usage_gb` | `mem: <float> GB` | Peak GPU memory usage in GB |

Only `tokens_per_sec` and `loss` are compared against `result_dict` thresholds.
`mem_usage_gb` is extracted but not validated against a threshold (no key in `result_dict`).

---

## How to Extend

### Add a New Model (e.g., Llama 3.1 405B)

1. Add a new test file `torchtitan_llama3_1_405b_single.py` and/or `torchtitan_llama3_1_405b_distributed.py` — copy an existing file and change the model name string to `'llama3_1_405b'` in the `TorchTitanTrainingJob` constructor call.

2. Add `llama3_1_405b` entries to the config JSON under `single_node` and/or `multi_node`.

3. The lib already maps `llama3_1_405b` → `tt_config = 'llama3_405b'` automatically.

### Add a New GPU Type (e.g., mi350)

In the config JSON, add a new block alongside `mi300x` and `mi325`:
```json
"mi350": {
    "tokenizer_path": "meta-llama/Llama-3.1-8B",
    "model_size": "8b",
    ...
    "result_dict": {
        "tokens_per_sec": "<changeme>",
        "loss": "<changeme>"
    }
}
```
No changes to the lib or test files are needed — `gpu_type` is detected automatically from `rocm-smi` output.

### Update Result Thresholds

Edit the `result_dict` values in the config JSON. The threshold is a **minimum** — the test fails if `actual < threshold`. Use `_example_*` prefixed keys as reference values without affecting validation:
```json
"result_dict": {
    "_example_tokens_per_sec": "12000.0",
    "tokens_per_sec": "10000.0"
}
```

### Add a New Model Architecture (e.g., DeepSeek, Qwen)

The `TorchTitanTrainingJob.__init__` already handles model name detection via regex:
- Names matching `deepseek` → `tt_module = 'deepseek_v3'`
- Names matching `qwen` → `tt_module = 'qwen3'`

Create a test file using the new model name and add corresponding entries in the config JSON.

---

## Troubleshooting

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `AttributeError: TorchTitanTrainingJob` | Wrong import | Check `from cvs.lib import torchtitan_training_lib` |
| Training log empty after poll | Container not running | Check `docker ps` on nodes |
| `step: N` never found in log | Training crashed early | Check `scan_for_training_errors` output and dmesg |
| `tok/s` below threshold | Performance regression or wrong config | Verify `global_batch_size`, `nproc_per_node`, and GPU count |
| Firewall test fails | `ufw` still active | Manually run `sudo ufw stop` on all nodes |
| Broadcom NIC setup fails | Library not at expected path | Verify `/usr/local/lib/libbnxt_re-rdmav34.so` exists on host |
| ROCm path not detected | Unusual ROCm install layout | Set `rocm_dir` explicitly in config JSON |
