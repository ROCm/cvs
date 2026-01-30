# PyTorch XDit Inference Tests

This directory contains inference microbenchmark tests for PyTorch XDit models (WAN, Flux, etc.).

**IMPORTANT**: Always run these tests on SLURM compute nodes, never on login nodes (shared resources).

## WAN 2.2 Image-to-Video A14B Test

Test file: `pytorch_xdit_wan22_i2v_a14b.py`

### Overview

Runs WAN 2.2 I2V-A14B inference inside the `amdsiloai/pytorch-xdit:v25.11.2` container and validates:
- Successful container execution
- Presence of benchmark JSONs (`rank0_step*.json`)
- Presence of generated artifact (`video.mp4`)
- Average inference time meets GPU-specific threshold

### Prerequisites

1. **Cluster configuration** (`cluster.json`) with:
   - Node definitions
   - SSH credentials (username + private key)

2. **HF Token** (for model download):
   - Place your Hugging Face token in `~/.hf_token`
   - Or specify a different path in the config

3. **Compute node allocation** (SLURM):
   ```bash
   # Check current allocations
   squeue -u $USER
   
   # If no jobs running, allocate a compute node
   srun -N1 -p rccl --reservation=rccl_92 -Arccl -n8 --gres=gpu:8 --gpus-per-node=8 /bin/bash
   ```

4. **Storage requirements**:
   - ~40GB for model cache (`hf_home`)
   - ~10GB for output artifacts

### Running the Test

#### Basic invocation

```bash
cvs run pytorch_xdit_wan22_i2v_a14b \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_wan22_i2v_a14b.json
```

#### Example with absolute paths

```bash
cvs run pytorch_xdit_wan22_i2v_a14b \
  --cluster_file=/home/user/cluster.json \
  --config_file=/home/user/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_wan22_i2v_a14b.json
```

### Configuration

Edit `cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_wan22_i2v_a14b.json`:

```json
{
    "config": {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "wan22-benchmark",
        "hf_token_file": "/home/{user-id}/.hf_token",
        "hf_home": "/home/{user-id}",
        "output_base_dir": "/home/{user-id}/cvs_outputs",
        "model_repo": "Wan-AI/Wan2.2-I2V-A14B",
        "model_rev": "206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {},
            "env_dict": {}
        }
    },
    "benchmark_params": {
        "wan22_i2v_a14b": {
            "prompt": "Summer beach vacation...",
            "size": "720*1280",
            "frame_num": 81,
            "num_benchmark_steps": 5,
            "compile": true,
            "torchrun_nproc": 8,
            "expected_results": {
                "auto": {"max_avg_total_time_s": 15.0},
                "mi300x": {"max_avg_total_time_s": 12.0},
                "mi355": {"max_avg_total_time_s": 10.0}
            }
        }
    }
}
```

**Placeholders** (automatically resolved):
- `{user-id}`: Current system user
- `{home}`: User's home directory

**Key parameters**:
- `hf_home`: Host directory for HF model cache (mounted to `/hf_home` in container)
- `output_base_dir`: Host directory for benchmark outputs
- `num_benchmark_steps`: Number of inference iterations (5 recommended)
- `torchrun_nproc`: Number of GPU processes (typically 8 for MI300X)
- `expected_results`: Performance thresholds by GPU type

### Expected Output Artifacts

After a successful run, you'll find:

```
${output_base_dir}/wan_22_${hostname}_outputs/
├── outputs/
│   ├── outputs/
│   │   ├── video.mp4          # Generated video artifact
│   │   ├── rank0_step0.json   # Benchmark JSON (step 0)
│   │   ├── rank0_step1.json   # Benchmark JSON (step 1)
│   │   ├── rank0_step2.json   # ...
│   │   ├── rank0_step3.json
│   │   └── rank0_step4.json
```

**Note**: The `outputs/outputs/` nesting is expected (the benchmark writes to `/outputs/outputs/` inside the container).

Each `rank0_step*.json` contains:
```json
{
  "total_time": 10.234,
  ...
}
```

### Test Execution Flow

1. **Cleanup**: Remove any stale containers
2. **Cache verification**: Check if model is cached, download if needed
3. **Benchmark run**: Execute WAN inference with torchrun
4. **Result parsing**: 
   - Locate `rank0_step*.json` files
   - Parse `total_time` values
   - Compute average across steps
   - Verify `video.mp4` artifact exists
5. **Threshold validation**: Compare average time to GPU-specific threshold

### Pass/Fail Criteria

The test **PASSES** if:
- Model cache exists or downloads successfully
- Container executes without errors
- At least one `rank0_step*.json` file is generated
- `video.mp4` artifact is generated
- Average `total_time` ≤ `expected_results[gpu_type].max_avg_total_time_s`

The test **FAILS** if:
-  Model download fails (HF token missing/invalid)
-  Container execution fails (Docker/GPU errors)
-  No benchmark JSONs found
-  Artifact missing
-  Average time exceeds threshold

### Troubleshooting

#### HF Token Issues
```
Error: HF token required to download...
```
**Solution**: Create `~/.hf_token` with your Hugging Face access token

#### Model Download Timeout
```
Model download failed with exception: timeout
```
**Solution**: Increase timeout in test or manually pre-download:
```bash
docker run --rm \
  --mount type=bind,source=$HOME,target=/hf_home \
  -e HF_HOME=/hf_home \
  -e HF_TOKEN=$(cat ~/.hf_token) \
  amdsiloai/pytorch-xdit:v25.11.2 \
  hf download Wan-AI/Wan2.2-I2V-A14B --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3
```

#### Performance Threshold Exceeded
```
FAIL: Average total_time 13.45s > threshold 12.0s (GPU: mi300x)
```
**Solution**: Either optimize the run or adjust threshold in config if the baseline changed

#### Missing Output Artifacts
```
Artifact 'video.mp4' not found...
```
**Solution**: Check container logs for inference errors, verify GPU devices are accessible

### Configuration Validation

The test uses **Pydantic schemas** for fail-fast validation. If your config has issues, you'll see clear errors:

```
Invalid WAN configuration: 
  config.model_repo: field required
```

Common validation errors:
- Missing required fields (`hf_token_file`, `hf_home`, etc.)
- Invalid types (e.g., `compile` must be boolean)
- Invalid ranges (e.g., `num_benchmark_steps` must be ≥ 1)
- Missing `expected_results` or no `auto`/GPU-specific threshold

### GPU Type Detection

The test auto-detects GPU type from `rocm-smi` output:
- `mi300x` → uses `expected_results.mi300x` threshold
- `mi355` → uses `expected_results.mi355` threshold
- Other/unknown → uses `expected_results.auto` threshold

### Listing Available Tests

```bash
# List all CVS tests
cvs list

# List test functions within this test module
cvs list pytorch_xdit_wan22_i2v_a14b
```

### Example Output

```
============================= test session starts ==============================
collecting ... collected 4 items

cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_i2v_a14b.py::test_cleanup_stale_containers PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_i2v_a14b.py::test_verify_hf_cache_or_download PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_i2v_a14b.py::test_run_wan22_benchmark PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_i2v_a14b.py::test_parse_and_validate_results PASSED

============================== 4 passed in 652.34s ==============================
```

---

## FLUX.1-dev Text-to-Image Test

Test file: `pytorch_xdit_flux1_dev_t2i.py`

### Overview

Runs FLUX.1-dev text-to-image inference inside the `amdsiloai/pytorch-xdit:v25.11.2` container and validates:
- Successful container execution
- Presence of timing.json with pipe_time measurements
- Presence of generated images (`flux_*.png`)
- Average inference time meets GPU-specific threshold

### Prerequisites

1. **Cluster configuration** (`cluster.json`) with:
   - Node definitions
   - SSH credentials (username + private key)

2. **HF Token** (for model download):
   - Place your Hugging Face token in `~/.hf_token`
   - Or specify a different path in the config
   - **Note**: FLUX.1-dev requires accepting the model license on Hugging Face

3. **Compute node allocation** (SLURM):
   ```bash
   # Check current allocations
   squeue -u $USER
   
   # If no jobs running, allocate a compute node
   srun -N1 -p rccl --reservation=rccl_92 -Arccl -n8 --gres=gpu:8 --gpus-per-node=8 /bin/bash
   ```

4. **Storage requirements**:
   - ~35GB for model cache (`hf_home`)
   - ~5GB for output artifacts

### Running the Test

#### Basic invocation

```bash
cvs run pytorch_xdit_flux1_dev_t2i \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_flux1_dev_t2i.json
```

#### Example with absolute paths

```bash
cvs run pytorch_xdit_flux1_dev_t2i \
  --cluster_file=/home/user/cluster.json \
  --config_file=/home/user/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_flux1_dev_t2i.json
```

### Configuration

Edit `cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_flux1_dev_t2i.json`:

```json
{
    "config": {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "flux-benchmark",
        "hf_token_file": "/home/{user-id}/.hf_token",
        "hf_home": "/home/{user-id}",
        "output_base_dir": "/home/{user-id}/cvs_flux_output",
        "model_repo": "black-forest-labs/FLUX.1-dev",
        "model_rev": "",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {},
            "env_dict": {}
        }
    },
    "benchmark_params": {
        "flux1_dev_t2i": {
            "prompt": "A small cat",
            "seed": 42,
            "num_inference_steps": 25,
            "max_sequence_length": 256,
            "no_use_resolution_binning": true,
            "warmup_steps": 1,
            "warmup_calls": 5,
            "num_repetitions": 25,
            "height": 1024,
            "width": 1024,
            "ulysses_degree": 8,
            "ring_degree": 1,
            "use_torch_compile": true,
            "torchrun_nproc": 8,
            "expected_results": {
                "auto": {"max_avg_pipe_time_s": 10.0},
                "mi300x": {"max_avg_pipe_time_s": 8.0},
                "mi355": {"max_avg_pipe_time_s": 7.0}
            }
        }
    }
}
```

**Placeholders** (automatically resolved):
- `{user-id}`: Current system user
- `{home}`: User's home directory

**Key parameters**:
- `hf_home`: Host directory for HF model cache (mounted to `/hf_home` in container)
- `output_base_dir`: Host directory for benchmark outputs
- `model_rev`: Model revision (empty string means use any available snapshot)
- `num_repetitions`: Number of inference iterations (25 recommended for stable averages)
- `num_inference_steps`: Number of denoising steps (25 is default for FLUX.1-dev)
- `torchrun_nproc`: Number of GPU processes (typically 8 for MI300X)
- `use_torch_compile`: Enable torch.compile optimization (recommended)
- `expected_results`: Performance thresholds by GPU type

### Expected Output Artifacts

After a successful run, you'll find:

```
${output_base_dir}/flux_${hostname}_outputs/
├── results/
│   ├── timing.json        # Benchmark timing data (JSON list with pipe_time)
│   ├── flux_0.png         # Generated image (repetition 0)
│   ├── flux_1.png         # Generated image (repetition 1)
│   ├── flux_2.png         # ...
│   └── ...
```

The `timing.json` file contains a JSON list where each entry has:
```json
[
  {
    "pipe_time": 5.234,
    ...
  },
  ...
]
```

### Test Execution Flow

1. **Cleanup**: Remove any stale containers
2. **Cache verification**: Check if model is cached, download if needed
   - If `model_rev` is set: requires specific snapshot
   - If `model_rev` is empty: any snapshot is acceptable
3. **Benchmark run**: Execute Flux inference with torchrun
4. **Result parsing**: 
   - Locate `results/timing.json`
   - Parse `pipe_time` values from JSON list
   - Compute average across repetitions
   - Verify at least one `flux_*.png` image exists
5. **Threshold validation**: Compare average pipe_time to GPU-specific threshold

### Pass/Fail Criteria

The test **PASSES** if:
- Model cache exists or downloads successfully
- Container executes without errors
- `timing.json` file is generated with valid pipe_time entries
- At least one `flux_*.png` image is generated
- Average `pipe_time` ≤ `expected_results[gpu_type].max_avg_pipe_time_s`

The test **FAILS** if:
- Model download fails (HF token missing/invalid or license not accepted)
- Container execution fails (Docker/GPU errors)
- No timing.json found or no valid pipe_time values
- No images generated
- Average time exceeds threshold

### Troubleshooting

#### HF Token Issues / License Not Accepted
```
Error: HF token required to download...
Error: Repository not found or access denied
```
**Solution**: 
1. Create `~/.hf_token` with your Hugging Face access token
2. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the model license
3. Ensure your token has read access to gated repositories

#### Model Download Timeout
```
Model download failed with exception: timeout
```
**Solution**: Increase timeout in test or manually pre-download:
```bash
docker run --rm \
  --mount type=bind,source=$HOME,target=/hf_home \
  -e HF_HOME=/hf_home \
  -e HF_TOKEN=$(cat ~/.hf_token) \
  amdsiloai/pytorch-xdit:v25.11.2 \
  hf download black-forest-labs/FLUX.1-dev
```

#### Performance Threshold Exceeded
```
FAIL: Average pipe_time 9.23s > threshold 8.0s (GPU: mi300x)
```
**Solution**: Either optimize the run or adjust threshold in config if the baseline changed

#### Missing Output Artifacts
```
No images matching 'flux_*.png' found...
```
**Solution**: Check container logs for inference errors, verify GPU devices are accessible

#### timing.json Not Found
```
timing.json not found under output directory
```
**Solution**: Check that `--benchmark_output_directory /outputs` was passed to the Flux script and verify container completed successfully

### Configuration Validation

The test uses **Pydantic schemas** for fail-fast validation. If your config has issues, you'll see clear errors:

```
Invalid Flux configuration: 
  config.model_repo: field required
```

Common validation errors:
- Missing required fields (`hf_token_file`, `hf_home`, etc.)
- Invalid types (e.g., `use_torch_compile` must be boolean)
- Invalid ranges (e.g., `num_repetitions` must be ≥ 1)
- Missing `expected_results` or no `auto`/GPU-specific threshold

### GPU Type Detection

The test auto-detects GPU type from `rocm-smi` output:
- `mi300x` → uses `expected_results.mi300x` threshold
- `mi355` → uses `expected_results.mi355` threshold
- Other/unknown → uses `expected_results.auto` threshold

### Listing Available Tests

```bash
# List all CVS tests
cvs list

# List test functions within this test module
cvs list pytorch_xdit_flux1_dev_t2i
```

### Example Output

```
============================= test session starts ==============================
collecting ... collected 4 items

cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_t2i.py::test_cleanup_stale_containers PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_t2i.py::test_verify_hf_cache_or_download PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_t2i.py::test_run_flux1_benchmark PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_t2i.py::test_parse_and_validate_results PASSED

============================== 4 passed in 485.21s ==============================
```