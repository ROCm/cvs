# Primus DeepSeek Training Tests

This directory contains CVS integration tests for DeepSeek models using the Primus framework.

## Overview

Primus is AMD's optimized training framework that provides:
- **MoE (Mixture of Experts)** support with Expert Parallelism
- **Turbo optimizations** (turbo_deepep, turbo_grouped_mlp)
- **AINIC networking** support for MI355X clusters
- **DeepSeek model configurations** (V2 Lite, V3)
- **Advanced parallelism** strategies (PP, EP, VPP)

Container: `docker.io/tasimage/primus:pr-563-ainic`

## Supported Models

- **DeepSeek-V3**: 61 layers, 256 experts, full production scale
- **DeepSeek-V2-Lite**: 27 layers, lighter model for testing

## Directory Structure

```
cvs/tests/training/primus/
├── primus_deepseek_v3_distributed.py    # Main test file
└── README.md                             # This file

cvs/input/config_file/training/primus/
└── mi355x_primus_deepseek_distributed.json  # Unified config with <changeme> placeholders
```

## Prerequisites

### 1. Cluster Configuration

Create a cluster JSON file (or use existing):

```json
{
  "username": "your_username",
  "priv_key_file": "/home/your_username/.ssh/id_rsa",
  "node_dict": {
    "node1.example.com": {},
    "node2.example.com": {},
    ...
  }
}
```

### 2. C4 Dataset (Optional)

For training with real data, download C4 shards:

```bash
# On shared storage visible to all nodes
cd /shared/c4
mkdir -p en

# Download C4 English shards (200 shards ≈ 70GB compressed)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/c4-train.{00000..00199}-of-01024.json.gz"
```

Or set `mock_data: true` in config to skip data preparation.

### 3. HuggingFace Token

Save your HF token to a file:

```bash
echo "your_hf_token" > ~/.hf_token
chmod 600 ~/.hf_token
```

## Running Tests

### Configuration Setup

The config file uses `<changeme>` placeholders following CVS conventions. Before running tests, edit [cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json](cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json) and replace placeholders with your environment values:

- **`nnodes`**: Number of nodes (8, 128, etc.) - see `_example_nnodes_*` for guidance
- **`master_address`**: Head node IP address
- **`training_iterations`**: Number of iterations to run
- **`nccl_ib_hca`**: Your AINIC/IB HCA list
- **`nccl_socket_ifname`**: Network interface name
- **`mock_data`**: `"True"` for testing without real data, `"False"` for production
- **Model parameters**: `pipeline_parallel`, `expert_parallel`, `micro_batch_size`, etc.

The config includes `_example_*` fields showing recommended values for 8-node vs 128-node scales.

### Basic Usage

```bash
cd /home/ichristo/github/ROCm/cvs

# Run DeepSeek V3 training test
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  --cluster_file=/path/to/cluster.json \
  --config_file=cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json \
  -v -s
```

### Test Progression

Start with smaller scale and work up:

#### 1. **2-Node Test with Mock Data**

```bash
# Edit config: Set nnodes=2, mock_data=True, training_iterations=10
# Use 8-node parallelism settings (PP=4, EP=4, VPP=1)
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  --cluster_file=cluster_2node.json \
  --config_file=cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json \
  -k test_deepseek_v3_distributed_training \
  -v -s
```

#### 2. **8-Node Test with Real Data**

```bash
# Edit config: Set nnodes=8, mock_data=False, training_iterations=20
# Use 8-node parallelism settings (PP=4, EP=4, VPP=1)
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  --cluster_file=cluster_8node.json \
  --config_file=cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json \
  -v -s
```

#### 3. **128-Node Production Run**

```bash
# Edit config: Set nnodes=128, mock_data=False, training_iterations=50
# Use 128-node parallelism settings (PP=8, EP=8, VPP=2)
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  --cluster_file=cluster_128node.json \
  --config_file=cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json \
  -v -s
```

### Running Specific Tests

```bash
# Only data preparation
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  -k test_cleanup -v

# Only container launch
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  -k test_launch -v

# Full training test
pytest cvs/tests/training/primus/primus_deepseek_v3_distributed.py \
  -k test_deepseek_v3_distributed_training -v -s
```

## Configuration Guide

### Key Parameters

Edit `cvs/input/config_file/training/primus/mi355x_primus_deepseek_distributed.json` and replace all `<changeme>` placeholders:

#### Training Scale
```json
{
  "config": {
    "training_iterations": 50,      // Number of training iterations
    "nnodes": 128,                   // Number of nodes
    "mock_data": false               // Use real C4 data
  }
}
```

#### MoE Parallelism
```json
{
  "model_params": {
    "multi_node": {
      "deepseek_v3": {
        "mi300x": {
          "pipeline_parallel": 8,          // Pipeline stages
          "expert_parallel": 8,            // Expert parallelism
          "virtual_pipeline_parallel": 2,  // Virtual pipeline
          "tensor_parallel": 1             // Tensor parallelism
        }
      }
    }
  }
}
```

#### Batch Sizes
```json
{
  "micro_batch_size": 2,           // Per-device batch size
  "global_batch_size": 16384       // Total across all devices
}
```

#### AINIC Networking
```json
{
  "using_ainic": true,
  "nccl_ib_hca": "ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1",
  "nccl_socket_ifname": "ens9np0",
  "nccl_ib_timeout": 23
}
```

#### Primus Optimizations
```json
{
  "turbo_deepep": true,             // Enable Turbo DeepEP
  "turbo_grouped_mlp": false,       // Turbo Grouped MLP
  "legacy_grouped_gemm": true,      // Stable GEMM
  "manual_gc": true,                // Manual garbage collection
  "pp_warmup": true                 // Pipeline warmup
}
```

## Data Preparation

The test automatically prepares C4 data if enabled:

```json
{
  "data_prep": {
    "enabled": true,
    "num_shards": 200,
    "data_dir": "/shared/c4",
    "tokenized_data_path": "/shared/c4/tokenized/c4_en_train_text_document"
  }
}
```

Data preparation happens once and is cached. To force re-preparation:

```bash
# Delete tokenized files
rm -rf /shared/c4/tokenized/*
```

## Troubleshooting

### Container Issues

```bash
# Check container status
docker ps | grep primus

# View container logs
docker logs primus_deepseek_v3

# Exec into container
docker exec -it primus_deepseek_v3 bash
```

### Network Issues

```bash
# Verify AINIC devices
ibv_devices | grep ionic

# Test RDMA connectivity
ibv_devinfo

# Check NCCL
NCCL_DEBUG=INFO <your_training_command>
```

### Training Logs

Logs are saved to `/shared/logs/cvs/deepseek_v3/`:

```bash
# Tail training log
tail -f /shared/logs/cvs/deepseek_v3/deepseek_training_node0.log

# Check for errors
grep -i error /shared/logs/cvs/deepseek_v3/*.log

# Monitor progress
watch -n 5 'tail -20 /shared/logs/cvs/deepseek_v3/deepseek_training_node0.log'
```

## Performance Expectations

### DeepSeek V3 (128 nodes, 1024 MI355X GPUs)

- **Throughput**: >50,000 tokens/sec
- **Memory**: ~80GB per GPU
- **Time per iteration**: ~60-120 seconds

### DeepSeek V2 Lite (8 nodes, 64 MI355X GPUs)

- **Throughput**: >100,000 tokens/sec
- **Memory**: ~40GB per GPU
- **Time per iteration**: ~20-40 seconds

## Comparison with Manual Scripts

| Feature | Manual Scripts | CVS Primus Tests |
|---------|----------------|------------------|
| **Orchestration** | SLURM | Docker + pytest |
| **Configuration** | Bash env vars | JSON files |
| **Container** | ✅ Same Primus | ✅ Same Primus |
| **Validation** | Manual | Automated |
| **Monitoring** | Manual log review | Built-in checks |
| **Network stats** | Not collected | Pre/post RDMA |
| **Repeatability** | Manual rerun | Pytest fixtures |

## Development

### Adding New Models

1. Add model params to config JSON:
```json
{
  "model_params": {
    "multi_node": {
      "your_model": {
        "mi300x": {
          // model configuration
        }
      }
    }
  }
}
```

2. Update test file if needed (usually not required).

### Adding New Tests

Create a new test function in `primus_deepseek_v3_distributed.py`:

```python
def test_your_new_test(phdl, training_dict, model_params_dict, hf_token):
    """Your test description"""
    globals.error_list = []
    
    # Your test logic
    
    update_test_result()
```

## Support

For issues or questions:
- Check logs in `/shared/logs/cvs/deepseek_v3/`
- Review test output with `-v -s` flags
- Verify cluster configuration
- Ensure Primus container is accessible

## References

- [Manual Primus Scripts](../../../manual_primus/)
- [CVS Megatron Tests](../megatron/)
- [Primus Documentation](https://example.com/primus-docs)
