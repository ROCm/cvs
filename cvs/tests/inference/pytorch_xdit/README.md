# PyTorch XDit Inference Tests

This directory contains inference microbenchmark tests for PyTorch XDit models (WAN, Flux, etc.).

**IMPORTANT**: Use GPU-capable targets with the drivers and device nodes this workload expects (for example ROCm and `/dev/kfd` for these tests). Do not run GPU benchmarks on machines that are not set up for them.

For how Hugging Face stores models and what `HF_HOME` means, see the official docs: [Hugging Face Hub cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) and [`HF_HOME` / cache environment variables](https://huggingface.co/docs/huggingface/package_reference/environment_variables). Here, `config.hf_home` is bind-mounted to `/hf_home` in the container and should be the directory that contains the `hub/` cache layout the test expects (not necessarily your UNIX home).

**Config placeholders** (see `cvs.lib.utils_lib`): test JSON files resolve `{user-id}`, `{user}`, and `{home}`. Cluster files resolve `{user-id}` only.

**Cleanup test**: the first test stops the named benchmark container and runs `docker system prune --force` on targets. To skip only the prune step after the kill, set `CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1` (values `1`, `true`, `yes`, `on`).

**Operational notes**: long SSH sessions may hit a stale client; `Pssh` retries once on `SessionError`. The run steps fail fast if the container image is missing on a node, and they verify expected output artifacts exist before parse/threshold steps. Output directories are named from each target’s `cluster.json` / `node_dict` key (the SSH target string), not the remote `hostname`, so runs stay stable when short vs FQDN hostnames differ.

**Thresholds** in sample configs are placeholders; tune them for your hardware and workload.

## First-Time Setup

The shortest path to a successful first run is:

1. SSH to the reserved compute node.
2. Stage models into a real Hugging Face cache root on that node.
3. Pull the XDit container image on that node.
4. Create a single-node `cluster.json`.
5. Run the full pytest module (or `cvs run`) with `CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1` on shared systems.

### 1. Connect to the compute node

```bash
ssh <reserved-compute-node-host-or-ip>
```

### 2. Install `hf` CLI on the node if needed

```bash
python3 -m venv /tmp/hf-cli-venv
/tmp/hf-cli-venv/bin/pip install 'huggingface_hub[cli]'
```

### 3. Set the Hugging Face cache root

Use the directory you want to put in `config.hf_home`. A common choice is:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

If you use a gated model and have a token file:

```bash
export HF_TOKEN="$(tr -d '\n' < "$HOME/.hf_token")"
```

If your workflow is fully offline and the model is already present in `HF_HOME`, you can skip the download step.

### 4. Stage the models with `hf download`

#### WAN

```bash
HF_HOME="$HF_HOME" /tmp/hf-cli-venv/bin/hf download \
  Wan-AI/Wan2.2-I2V-A14B \
  --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3
```

#### FLUX

```bash
HF_HOME="$HF_HOME" /tmp/hf-cli-venv/bin/hf download \
  black-forest-labs/FLUX.1-dev
```

### 5. Pull the container image on the target node

```bash
docker pull amdsiloai/pytorch-xdit:v25.11.2
```

### 6. Quick sanity checks

```bash
test -e /dev/kfd && echo KFD_OK
test -d "$HF_HOME/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3" && echo WAN_OK
test -d "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots" && echo FLUX_OK
docker image inspect amdsiloai/pytorch-xdit:v25.11.2 >/dev/null && echo IMG_OK
```

### 7. Minimal single-node `cluster.json`

A minimal single-node example with placeholders looks like:

```json
{
  "username": "{user-id}",
  "priv_key_file": "/path/to/private/key",
  "head_node_dict": {
    "mgmt_ip": "REPLACE_WITH_COMPUTE_NODE_HOST_OR_IP"
  },
  "env_vars": {},
  "node_dict": {
    "REPLACE_WITH_COMPUTE_NODE_HOST_OR_IP": {
      "bmc_ip": "NA",
      "vpc_ip": "REPLACE_WITH_COMPUTE_NODE_HOST_OR_IP"
    }
  }
}
```

The **same SSH target string** should be used for:

- `head_node_dict.mgmt_ip`
- the `node_dict` key
- `node_dict.<node>.vpc_ip`

Unlike the test JSON files, cluster files resolve `{user-id}` only. Do not use `{home}` in `priv_key_file`; use a real absolute path if `/home/{user-id}/.ssh/id_rsa` is not correct on your system.

This keeps the output directory name stable (`wan_22_<cluster_target>_outputs`, `flux_<cluster_target>_outputs`) even if the remote host reports a different short hostname or FQDN.

### 8. Run the full module

If your `cvs` CLI environment is installed cleanly:

```bash
CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1 \
cvs run pytorch_xdit_wan22_14b_single \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_wan22_14b_single.json
```

```bash
CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1 \
cvs run pytorch_xdit_flux1_dev_single \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json
```

If you are running from a source checkout and do not have a working `cvs` entrypoint, the equivalent form is:

```bash
CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1 \
PYTHONPATH=/path/to/cvs \
python -m pytest cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_14b_single.py \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_wan22_14b_single.json \
  -q -s --tb=short
```

```bash
CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1 \
PYTHONPATH=/path/to/cvs \
python -m pytest cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_single.py \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json \
  -q -s --tb=short
```

---

## WAN 2.2 Image-to-Video A14B Test

Test file: `pytorch_xdit_wan22_14b_single.py`

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

2. **Model must be pre-staged locally** (no runtime downloads):
   - Set `config.model_repo` to an explicit on-disk path on every node (recommended), e.g. `/models/Wan-AI/Wan2.2-I2V-A14B`
   - Alternatively, use a HF repo id in `config.model_repo` only if the snapshot is already cached under `hf_home` (offline mode). For `Wan-AI/Wan2.2-I2V-A14B`, the verification step applies additional snapshot layout checks.

3. **Targets** with Docker, the configured `container_image` already pulled locally, and ROCm device nodes as required by the container (sample configs use eight processes).

4. **Storage requirements**:
   - ~40GB for model cache (`hf_home`)
   - ~10GB for output artifacts

### Running the Test

#### Basic invocation

```bash
cvs run pytorch_xdit_wan22_14b_single \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_wan22_14b_single.json
```

#### Example with absolute paths

```bash
cvs run pytorch_xdit_wan22_14b_single \
  --cluster_file=/home/user/cluster.json \
  --config_file=/home/user/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_wan22_14b_single.json
```

### Configuration

Edit `cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_wan22_14b_single.json`:

```json
{
    "config": {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "wan22-benchmark",
        "hf_token_file": "{home}/.hf_token",
        "hf_home": "{home}/.cache/huggingface",
        "output_base_dir": "{home}/cvs_outputs",
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
                "auto": {"max_avg_total_time_s": 9999.0}
            }
        }
    }
}
```

**Placeholders** (automatically resolved in test config):

- `{user-id}`: Username
- `{home}`: Home directory of that user

**Key parameters**:

- `hf_home`: Host directory for HF model cache (mounted to `/hf_home` in container). This should be the root that contains `hub/`, typically `{home}/.cache/huggingface`.
- `output_base_dir`: Host directory for benchmark outputs
- `num_benchmark_steps`: Number of inference iterations (5 recommended)
- `torchrun_nproc`: Number of GPU processes (typically 8 for MI300X)
- `expected_results`: Performance thresholds by GPU type

### Expected Output Artifacts

After a successful run, outputs are under a directory named with the cluster **SSH target** for that node (see shared notes above), not necessarily `hostname`:

```
${output_base_dir}/wan_22_<cluster_target>_outputs/
├── outputs/
│   ├── outputs/
│   │   ├── video.mp4          # Generated video artifact
│   │   ├── rank0_step0.json   # Benchmark JSON (example)
│   │   └── ...
```

Step JSON names follow `rank0_step*.json`; with **`num_benchmark_steps: 1`**, a successful run may produce **`rank0_step1.json`** rather than `rank0_step0.json`. **`video.mp4`** may be located via recursive search under the run directory.

Each `rank0_step*.json` contains:

```json
{
  "total_time": 10.234,
  ...
}
```

### Test Execution Flow

1. **Cleanup**: Stop stale benchmark container; optional prune (see above).
2. **Model verification (offline)**: Paths and, for standard WAN repo id flows, snapshot layout checks.
3. **Preflight**: `/dev/kfd` and local `config.container_image` on each target.
4. **Benchmark run**: WAN inference with torchrun; verifies non-empty `rank0_step*.json` and `video.mp4` on success.
5. **Result parsing**: Locate step JSONs and `video.mp4`, average timings, threshold check.

### Pass/Fail Criteria

The test **PASSES** if:

- Model is present and passes verification (including WAN snapshot checks where applicable).
- Container executes without errors; expected artifacts are present.
- Average `total_time` meets the configured threshold for the detected GPU type.

The test **FAILS** if:

- Model is missing or incomplete on a target node.
- Preflight checks fail (`/dev/kfd`, container image).
- Container execution fails or artifacts are missing/empty.
- Average time exceeds threshold.

### Troubleshooting

#### Missing Model Path

```
Local model path not found on <node>: /models/...
```

**Solution**: Pre-stage the model directory on every node and set `config.model_repo` to that path.

#### Container Image Missing

```
Container image not found locally on <node>...
```

**Solution**: `docker pull` the configured image on each target node before running the test.

#### Performance Threshold Exceeded

```
FAIL: Average total_time XX.XXs > threshold YY.YYs (GPU: <gpu_type>)
```

**Solution**: Tune the workload or thresholds for your environment.

#### Missing Output Artifacts

```
Artifact 'video.mp4' not found...
```

**Solution**: Inspect logs on the target; confirm you are looking under `wan_22_<cluster_target>_outputs` for that node’s SSH key in `node_dict`.

#### SSH errors after a long run

If a lightweight remote command fails after a long run, `Pssh` retries once on `SessionError`; check logs for the retry line.

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
cvs list pytorch_xdit_wan22_14b_single
```

### Example Output

```
============================= test session starts ==============================
collecting ... collected 4 items

cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_14b_single.py::test_cleanup_stale_containers PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_14b_single.py::test_verify_hf_cache_or_download PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_14b_single.py::test_run_wan22_benchmark PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_14b_single.py::test_parse_and_validate_results PASSED

============================== 4 passed in X.XXs ==============================
```

---

## FLUX.1-dev Text-to-Image Test

Test file: `pytorch_xdit_flux1_dev_single.py`

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

2. **Model must be pre-staged locally** (no runtime downloads):
   - Set `config.model_repo` to an explicit on-disk path on every node (recommended), e.g. `/models/black-forest-labs/FLUX.1-dev`
   - Alternatively, use a HF repo id in `config.model_repo` only if the snapshot is already cached under `hf_home` (offline mode)
   - If you pre-download from Hugging Face, ensure any required model license is accepted beforehand

3. **Targets** with Docker, the configured `container_image` already pulled locally, and ROCm as required by the container.

4. **Storage requirements**:
   - ~35GB for model cache (`hf_home`)
   - ~5GB for output artifacts

### Running the Test

#### Basic invocation

```bash
cvs run pytorch_xdit_flux1_dev_single \
  --cluster_file=/path/to/cluster.json \
  --config_file=/path/to/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json
```

#### Example with absolute paths

```bash
cvs run pytorch_xdit_flux1_dev_single \
  --cluster_file=/home/user/cluster.json \
  --config_file=/home/user/cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json
```

### Configuration

Edit `cvs/cvs/input/config_file/inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json`:

```json
{
    "config": {
        "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
        "container_name": "flux-benchmark",
        "hf_token_file": "{home}/.hf_token",
        "hf_home": "{home}/.cache/huggingface",
        "output_base_dir": "{home}/cvs_flux_output",
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
                "auto": {"max_avg_pipe_time_s": 9999.0}
            }
        }
    }
}
```

**Placeholders** (automatically resolved in test config):

- `{user-id}`: Username
- `{home}`: Home directory of that user

**Key parameters**:

- `hf_home`: Host directory for HF model cache (mounted to `/hf_home` in container). This should be the root that contains `hub/`, typically `{home}/.cache/huggingface`.
- `output_base_dir`: Host directory for benchmark outputs
- `model_rev`: Model revision (empty string means use any available snapshot)
- `num_repetitions`: Number of inference iterations (25 recommended for stable averages)
- `num_inference_steps`: Number of denoising steps (25 is default for FLUX.1-dev)
- `torchrun_nproc`: Number of GPU processes (typically 8 for MI300X)
- `use_torch_compile`: Enable torch.compile optimization (recommended)
- `expected_results`: Performance thresholds by GPU type

### Expected Output Artifacts

After a successful run:

```
${output_base_dir}/flux_<cluster_target>_outputs/
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

1. **Cleanup**: Same as WAN (see above).
2. **Model verification (offline)**: Diffusers-style checks for absolute paths; HF cache rules for repo ids.
3. **Preflight**: `/dev/kfd` and local `config.container_image` on each target.
4. **Benchmark run**: Flux inference; verifies non-empty `results/timing.json` and at least one non-empty `flux_*.png`.
5. **Result parsing**: Average `pipe_time`, threshold check.

### Pass/Fail Criteria

The test **PASSES** if:

- Model is present and passes verification.
- Container executes without errors; timing and images are present.
- Average `pipe_time` meets the configured threshold for the detected GPU type.

The test **FAILS** if:

- Model is missing on a target node.
- Preflight or container execution fails.
- Timing data or images are missing.
- Average time exceeds threshold.

### Troubleshooting

#### Missing Model Path

If you see a failure like:

```
Local model path not found on <node>: /models/...
```

**Solution**: Pre-stage the model directory on every node and set `config.model_repo` to that path.

#### Incomplete FLUX directory

Errors about missing transformer/VAE weights or `model_index.json`.

**Solution**: Stage a complete diffusers-style tree.

#### Container Image Missing

```
Container image not found locally on <node>...
```

**Solution**: `docker pull` the configured image on each target node before running the test.

#### Performance Threshold Exceeded

```
FAIL: Average pipe_time XX.XXs > threshold YY.YYs (GPU: <gpu_type>)
```

**Solution**: Tune the workload or thresholds for your environment.

#### Missing Output Artifacts

```
No images matching 'flux_*.png' found...
```

**Solution**: Check container logs and GPU access.

#### timing.json Not Found

```
timing.json not found under output directory
```

**Solution**: Confirm the container finished successfully; look under `flux_<cluster_target>_outputs/results/`.

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
cvs list pytorch_xdit_flux1_dev_single
```

### Example Output

```
============================= test session starts ==============================
collecting ... collected 4 items

cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_single.py::test_cleanup_stale_containers PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_single.py::test_verify_hf_cache_or_download PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_single.py::test_run_flux1_benchmark PASSED
cvs/cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_single.py::test_parse_and_validate_results PASSED

============================== 4 passed in X.XXs ==============================
```
