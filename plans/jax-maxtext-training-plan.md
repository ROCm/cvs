# JAX-MaxText Distributed Training — CVS Test Suite Implementation Plan

## Phase 1 scope

End-to-end implementation for one model workload covering:

| # | Item | Source |
|---|---|---|
| FW-2 | Full training run — end-to-end loop completes successfully | Training Frameworks |
| WL-12 | Llama-3.3-70B \| BF16 \| SeqLen=8192 | Workload Coverage |
| PM-21 | Training Throughput — Tokens/sec & Tokens/sec/GPU | Performance Metrics |
| PM-22 | TFLOPS/sec/GPU | Performance Metrics |
| AM-32 | Loss Curve — training loss logged at step intervals, should decrease | Accuracy Metrics |
| AM-33 | Convergence — Time to Target Accuracy (steps + wall-clock) | Accuracy Metrics |

Platform-specific batch sizes (from the requirements table):
- MI300X: per_device_batch_size=2, GBS=16
- MI325X: per_device_batch_size=3, GBS=96
- MI355X: per_device_batch_size=6, GBS=48

---

## 1. Files to create

### 1.1 Library: `cvs/lib/training/jax_maxtext_training_lib.py`

The driver class. Common for both single-node and multi-node runs.

### 1.2 Config loader: `cvs/lib/training/utils/training_config_loader.py`

Pydantic schema subclassing `BaseVariantConfig` from `cvs/lib/utils/config_loader.py`.

### 1.3 Metric parsing: `cvs/lib/training/utils/maxtext_parsing.py`

Pure functions to parse MaxText training log output into namespaced metrics dict.

### 1.4 Suite test file: `cvs/tests/training/jax_maxtext/jax_maxtext_training.py`

Single test file for both single-node and distributed. The mode is determined by the config file passed at runtime.

### 1.5 Suite conftest: `cvs/tests/training/jax_maxtext/conftest.py`

Fixtures, lifecycle state, HTML hooks.

### 1.6 Suite shared: `cvs/tests/training/jax_maxtext/_shared.py`

`test_print_results_table` — the summary table test.

### 1.7 Config files (per platform, in `cvs/input/config_file/training/jax_maxtext/`)

| File | Purpose |
|---|---|
| `mi300x_jax-maxtext_llama-3.3-70b_distributed_config.json` | MI300X distributed config |
| `mi300x_jax-maxtext_llama-3.3-70b_distributed_threshold.json` | MI300X distributed thresholds |
| `mi300x_jax-maxtext_llama-3.3-70b_single_config.json` | MI300X single-node config |
| `mi300x_jax-maxtext_llama-3.3-70b_single_threshold.json` | MI300X single-node thresholds |
| `mi325x_jax-maxtext_llama-3.3-70b_distributed_config.json` | MI325X distributed config |
| `mi325x_jax-maxtext_llama-3.3-70b_distributed_threshold.json` | MI325X distributed thresholds |
| `mi355x_jax-maxtext_llama-3.3-70b_distributed_config.json` | MI355X distributed config |
| `mi355x_jax-maxtext_llama-3.3-70b_distributed_threshold.json` | MI355X distributed thresholds |

### 1.8 Package init files

- `cvs/lib/training/__init__.py`
- `cvs/lib/training/utils/__init__.py`
- `cvs/tests/training/jax_maxtext/__init__.py`

---

## 2. Config schema design

### 2.1 Config JSON structure (`*_config.json`)

Subclasses `BaseVariantConfig` (inherits `schema_version`, `enforce_thresholds`,
`threshold_json`, `paths`, `model`, `container`, `thresholds`). Adds
training-specific sections.

```jsonc
{
  "schema_version": 1,
  "framework": "jax_maxtext",
  "gpu_arch": "mi300x",
  "enforce_thresholds": false,
  "threshold_json": "mi300x_jax-maxtext_llama-3.3-70b_distributed_threshold.json",

  "paths": {
    "shared_fs": "/home/{user-id}",
    "models_dir": "{shared_fs}/cache/maxtext",
    "log_dir": "{shared_fs}/LOGS/jax_maxtext",
    "hf_token_file": "{shared_fs}/.hf_token"
  },

  "model": {
    "id": "llama3.3-70b",
    "remote": 0,
    "precision": "bfloat16"
  },

  "container": {
    "lifetime": "per_run",
    "name": "rocm-jax-maxtext-llama3.3-70b",
    "image": "rocm/jax-training:maxtext-v26.3.1",
    "runtime": {
      "name": "docker",
      "args": {
        "network": "host",
        "ipc": "host",
        "privileged": true,
        "shm-size": "256G",
        "volume": [
          "/home/{user-id}:/home/{user-id}",
          "/dev/infiniband:/dev/infiniband"
        ]
      }
    }
  },

  // --- training-specific sections below ---

  "training": {
    "distributed": true,
    "steps": 30,
    "enable_checkpointing": false,

    // Path to train.py inside the container.
    // The driver sources the env script, then runs:
    //   python -m maxtext.trainers.pre_train.train <yml_config> [overrides...]
    "train_module": "maxtext.trainers.pre_train.train",

    // MaxText YAML config keys (written into a temp YAML inside the container).
    // These are the complete model/training params; the driver generates a YAML
    // from this dict and passes it as the first arg to train.py.
    "maxtext_config": {
      "base_config": "base.yml",
      "hardware": "gpu",
      "attention": "cudnn_flash_te",
      "dtype": "bfloat16",
      "dataset_type": "synthetic",
      "remat_policy": "full",
      "use_iota_embed": true,
      "scan_layers": true,
      "per_device_batch_size": 2,
      "max_target_length": 8192,
      "enable_checkpointing": false,
      "async_checkpointing": false,
      "quantization": "",
      "weight_dtype": "bfloat16",
      "shardy": false,
      "logits_dot_in_fp32": false,
      "ici_fsdp_parallelism": 8,
      "ici_data_parallelism": 1,
      "ici_sequence_parallelism": 1,
      "ici_tensor_parallelism": 1,
      "ici_pipeline_parallelism": 1,
      "dcn_data_parallelism": -1,
      "dcn_fsdp_parallelism": 1,
      "dcn_pipeline_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "dcn_sequence_parallelism": 1,
      "packing": true
    },

    // Tokenizer setup
    "tokenizer": {
      "hf_model_id": "NousResearch/Meta-Llama-3-70B",
      "tokenizer_path": "{paths.models_dir}/Meta-Llama-70-B"
    },

    // NIC type for RDMA library setup ("thor2", "mlnx", or "none")
    "nic_type": "thor2",

    // RDMA library paths (host -> container mount -> container dest)
    "rdma_lib": {
      "host_source_file": "/usr/local/lib/libbnxt_re-rdmav34.so",
      "container_mount_file": "/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host",
      "container_dest_file": "/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so"
    },

    // Environment variables set inside the container before training.
    "env_vars": {
      "GPU_MAX_HW_QUEUES": "2",
      "HSA_FORCE_FINE_GRAIN_PCIE": "1",
      "HIP_FORCE_DEV_KERNARG": "1",
      "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93",
      "NCCL_DEBUG": "ERROR",
      "NCCL_PROTO": "Simple",
      "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
      "NVTE_USE_HIPBLASLT": "1",
      "NVTE_FUSED_ATTN": "1",
      "NVTE_CK_USES_BWD_V3": "1",
      "NVTE_CK_USES_FWD_V3": "1",
      "NVTE_CK_IS_V3_ATOMIC_FP32": "0",
      "NVTE_CK_HOW_V3_BF16_CVT": "2",
      "NVTE_FUSED_ATTN_CK": "1",
      "NVTE_FUSED_ATTN_AOTRITON": "0"
    },

    // XLA flags (assembled into XLA_FLAGS string by the driver)
    "xla_flags": {
      "xla_gpu_enable_latency_hiding_scheduler": "True",
      "xla_gpu_enable_triton_gemm": "False",
      "xla_gpu_memory_limit_slop_factor": "95",
      "xla_gpu_enable_command_buffer": "",
      "xla_gpu_enable_cublaslt": "True",
      "xla_gpu_autotune_level": "0",
      "xla_gpu_reduce_scatter_combine_threshold_bytes": "8589934592",
      "xla_gpu_all_gather_combine_threshold_bytes": "8589934592",
      "xla_gpu_enable_all_gather_combine_by_dim": "FALSE"
    },

    // NCCL IB/network config (distributed only; single-node ignores these)
    "nccl": {
      "ib_hca_list": "<changeme>",
      "ib_hca": "<changeme>",
      "socket_ifname": "enp159s0np0",
      "gloo_socket_ifname": "enp159s0np0",
      "ib_tc": "41",
      "ib_sl": "0",
      "ib_gid_index": "3"
    },

    // JAX distributed settings (distributed only)
    "jax_distributed": {
      "coordinator_ip": "<changeme>",
      "coordinator_port": "12346",
      "initialization_timeout_seconds": "1800",
      "heartbeat_timeout_seconds": "900"
    }
  }
}
```

**Single-node config differences**: `training.distributed: false`, omit/ignore `nccl` and `jax_distributed`
sections, different `per_device_batch_size`.

### 2.2 Threshold JSON structure (`*_threshold.json`)

Cell keys use a training-specific format:
`STEPS=<N>,BATCH=<B>,SEQLEN=<S>,NODES=<N>`.

```jsonc
{
  "_comment": "Thresholds for Llama-3.3-70B BF16 on MI300X distributed",

  "STEPS=30,BATCH=2,SEQLEN=8192,NODES=2": {
    "training.tflops_per_sec_per_gpu": { "kind": "min", "value": 350.0 },
    "training.tokens_per_sec_per_gpu": { "kind": "min", "value": 700.0 },
    "training.tokens_per_sec_total":   { "kind": "min", "value": 0 },
    "training.final_loss":             { "kind": "max", "value": 15.0 },
    "training.loss_decreased":         { "kind": "min", "value": 1 }
  }
}
```

Start with `enforce_thresholds: false` (record-only) until real baselines are established.

---

## 3. Config loader: `training_config_loader.py`

Subclasses `BaseVariantConfig` from `cvs/lib/utils/config_loader.py`. Follows the
same pattern as `inferencing_config_loader.py`.

### 3.1 Pydantic models

```python
class Tokenizer(_Forbid):
    hf_model_id: str
    tokenizer_path: str

class NcclConfig(_Allow):
    ib_hca_list: str = ""
    ib_hca: str = ""
    socket_ifname: str = ""
    gloo_socket_ifname: str = ""
    ib_tc: str = "41"
    ib_sl: str = "0"
    ib_gid_index: str = "3"

class JaxDistributed(_Forbid):
    coordinator_ip: str = ""
    coordinator_port: str = "12346"
    initialization_timeout_seconds: str = "1800"
    heartbeat_timeout_seconds: str = "900"

class RdmaLib(_Allow):
    host_source_file: str = ""
    container_mount_file: str = ""
    container_dest_file: str = ""

class TrainingConfig(_Allow):
    distributed: bool = True
    steps: int = 30
    enable_checkpointing: bool = False
    train_module: str = "maxtext.trainers.pre_train.train"
    maxtext_config: Dict[str, Any] = {}
    tokenizer: Tokenizer
    nic_type: str = "thor2"
    rdma_lib: RdmaLib = RdmaLib()
    env_vars: Dict[str, str] = {}
    xla_flags: Dict[str, str] = {}
    nccl: NcclConfig = NcclConfig()
    jax_distributed: JaxDistributed = JaxDistributed()

class TrainingVariantConfig(BaseVariantConfig):
    framework: Literal["jax_maxtext"]
    gpu_arch: str
    training: TrainingConfig
    # Inherits: schema_version, enforce_thresholds, threshold_json,
    #           paths, model, container, thresholds

    def cell_key(self) -> str:
        """Canonical threshold key for this training run."""
        t = self.training
        mc = t.maxtext_config
        batch = mc.get("per_device_batch_size", "-")
        seqlen = mc.get("max_target_length", "-")
        nodes = "1"  # overridden by orch.hosts count at runtime
        return f"STEPS={t.steps},BATCH={batch},SEQLEN={seqlen},NODES={nodes}"
```

### 3.2 Public API

```python
def load_training_variant(config_path: str, cluster_dict: dict) -> TrainingVariantConfig:
    """Read config + threshold, resolve placeholders, return typed config."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return TrainingVariantConfig(**raw)
```

---

## 4. Driver: `jax_maxtext_training_lib.py`

### 4.1 Class: `MaxTextTrainingJob`

Takes an `orch` (ContainerOrchestrator) and a typed `TrainingVariantConfig`.
All container interaction goes through `orch.exec()`. No direct Pssh or docker_lib.

```python
class MaxTextTrainingJob:
    def __init__(self, orch, variant: TrainingVariantConfig, hf_token: str):
        self.orch = orch
        self.variant = variant
        self.hf_token = hf_token
        self.training = variant.training
        self.log_dir = variant.paths.log_dir
        self.out_dir = f"{self.log_dir}/jax_maxtext"
        self.training_log = f"{self.out_dir}/training.log"
        # Populated after training completes
        self.step_metrics = []      # list of per-step dicts
        self.summary_metrics = {}   # aggregated metrics
```

### 4.2 Methods

| Method | Purpose | orch calls |
|---|---|---|
| `setup_training_env()` | Write env script (XLA_FLAGS, NCCL vars, HF token, custom env_vars) and MaxText YAML config into the container. Create output dirs. | `orch.exec()` to write files via `printf`, `mkdir -p` |
| `setup_rdma_lib()` | Copy host RDMA library into container (thor2 NIC workaround) | `orch.exec()` to `cp` files |
| `exec_nic_setup_scripts()` | Run NIC setup scripts inside container (Broadcom workarounds) | `orch.exec()` |
| `setup_tokenizer()` | Download HuggingFace tokenizer into the models dir | `orch.exec()` to run `hf download` |
| `build_training_cmd()` | Assemble the `python -m maxtext.trainers.pre_train.train <yml> [overrides]` command string. For distributed: set JAX_COORDINATOR_IP, NNODES, node-specific JAX_PROCESS_INDEX. Writes a launcher script. | `orch.exec()` to write launcher script |
| `start_training()` | Launch training in background via nohup | `orch.exec("bash -c 'nohup ... &'")` |
| `is_complete()` | Check training log for completion marker or errors | `orch.exec("grep ...")` with `detailed=True` |
| `poll_for_completion()` | Poll `is_complete()` with timeout; detect NaN/errors early | `orch.exec("tail ...")` per poll |
| `parse_results()` | Parse per-step metrics from training log, compute aggregates | `orch.exec("cat <training_log>")` |
| `stop_training()` | Kill any running training processes | `orch.exec("pkill ...")` |

### 4.3 Training command construction

The driver builds the training command in Python (no external `.sh` scripts),
following the vllm_single pattern. The env script is sourced first, then the
MaxText train module is invoked.

```python
def _build_train_argv(self):
    """Build the python -m maxtext.trainers.pre_train.train arg list."""
    argv = [
        "python", "-m", self.training.train_module,
        "/tmp/maxtext_config.yml",  # generated YAML
    ]
    # CLI overrides for dynamic values
    argv.extend([
        f"run_name=jax_maxtext_{self.variant.model.id}",
        f"steps={self.training.steps}",
        f"base_output_directory={self.out_dir}",
        f"tokenizer_path={self.training.tokenizer.tokenizer_path}",
    ])
    if self.training.enable_checkpointing:
        argv.append("enable_checkpointing=True")
    else:
        argv.append("enable_checkpointing=False")
    return argv
```

For distributed training, the launcher script per node sets:
```bash
export JAX_COORDINATOR_IP=<coordinator_ip>
export JAX_COORDINATOR_PORT=<port>
export NNODES=<num_nodes>
export JAX_PROCESS_INDEX=<node_index>  # derived from host position
```

The coordinator IP is determined from `orch.hosts[0]` (head node) at runtime
rather than from config, so the same config works on different clusters.

### 4.4 Log parsing

MaxText logs per-step metrics in a parseable format. The parser extracts:
- `perf_step_time_seconds` — per-step wall-clock time
- `learning/loss` — training loss per step
- `perf/step_time_seconds` — step time
- `Tokens/s/device` — tokens per second per GPU
- `TFLOP/s/device` — TFLOP/s per GPU

The parsing logic lives in `maxtext_parsing.py` as pure functions:

```python
TRAINING_METRICS = [
    ("tflops_per_sec_per_gpu", "TFLOP/s/GPU"),
    ("tokens_per_sec_per_gpu", "tok/s/GPU"),
    ("tokens_per_sec_total", "tok/s total"),
    ("step_time_seconds", "s/step"),
    ("loss", "loss"),
    ("final_loss", "loss"),
    ("loss_decreased", "bool"),
]

GATED_METRICS = [
    "training.tflops_per_sec_per_gpu",
    "training.tokens_per_sec_per_gpu",
]

def parse_training_log(log_text: str, num_gpus: int) -> dict:
    """Parse MaxText training log into namespaced metrics dict."""
    # Returns: {"training.<metric>": value, ...}

def extract_step_metrics(log_text: str) -> list[dict]:
    """Extract per-step metrics for loss curve analysis."""
    # Returns: [{"step": N, "loss": X, "tflops": Y, ...}, ...]
```

### 4.5 Single-node vs distributed logic

The driver uses `self.training.distributed` (from config) to branch:

| Aspect | Single-node | Distributed |
|---|---|---|
| JAX env vars | None | JAX_COORDINATOR_IP, NNODES, JAX_PROCESS_INDEX |
| NCCL IB vars | Skipped | Set from `training.nccl` |
| RDMA lib setup | Skipped | Runs `setup_rdma_lib()` |
| NIC scripts | Skipped | Runs `exec_nic_setup_scripts()` |
| Launch | `orch.exec()` on all hosts (1 host) | `orch.exec()` on all hosts with per-node JAX_PROCESS_INDEX |
| Coordinator IP | Not needed | Derived from `orch.hosts[0]` at runtime |

Both paths use the same `MaxTextTrainingJob` class; the config drives the
behavior.

---

## 5. Suite: `cvs/tests/training/jax_maxtext/`

### 5.1 Lifecycle-as-tests (test order)

```
test_launch_container          # orch.setup_containers()
test_setup_sshd                # orch.setup_sshd() (distributed only)
test_setup_rdma                # RDMA lib copy (distributed + thor2 only)
test_setup_nic                 # NIC setup scripts (distributed only)
test_setup_tokenizer           # HF tokenizer download
test_training_run              # build cmd + start + poll + parse
test_metric[tflops_per_sec_per_gpu]  # one HTML row per metric
test_metric[tokens_per_sec_per_gpu]
test_metric[tokens_per_sec_total]
test_metric[final_loss]
test_metric[loss_decreased]
test_print_results_table       # summary table
test_teardown                  # orch.teardown_containers()
```

### 5.2 `conftest.py` — fixtures

```python
# cluster_dict — loads cluster JSON, resolves placeholders
# variant_config — calls load_training_variant(config_file, cluster_dict)
# lifecycle — _Lifecycle() cross-test state
# orch — ContainerOrchestrator built from deep-merged cluster+variant container blocks
#         (same pattern as vllm/conftest.py)
# hf_token — reads from variant_config.paths.hf_token_file
# training_res_dict — module-scoped {} for storing results
```

The `orch` fixture follows the vllm pattern: deep-merges the variant's container
block onto the cluster's container block, creates `OrchestratorConfig` with
`orchestrator="container"`, and has a leak-guard finalizer.

### 5.3 `conftest.py` — parametrization

`pytest_generate_tests` parametrizes `test_metric` over `TRAINING_METRICS`:

```python
def pytest_generate_tests(metafunc):
    if "metric" in metafunc.fixturenames:
        metric_cases = [(short, unit) for short, unit in TRAINING_METRICS]
        ids = [short for short, _ in TRAINING_METRICS]
        metafunc.parametrize("metric", [m[0] for m in metric_cases], ids=ids)
```

### 5.4 `conftest.py` — test ordering

```python
def pytest_collection_modifyitems(items):
    rank = {
        "test_launch_container": 0,
        "test_setup_sshd": 1,
        "test_setup_rdma": 2,
        "test_setup_nic": 3,
        "test_setup_tokenizer": 4,
        "test_training_run": 5,
        "test_metric": 6,
        "test_print_results_table": 7,
        "test_teardown": 8,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))
```

### 5.5 `jax_maxtext_training.py` — test functions

```python
def test_launch_container(orch, variant_config, lifecycle, request):
    """Launch the container. Verify it is running."""
    # Same pattern as vllm_single.py test_launch_container

def test_setup_sshd(orch, lifecycle, request):
    """Start sshd inside the container (distributed only; single-node skips)."""
    if not variant_config.training.distributed:
        pytest.skip("single-node: sshd not needed")
    # Same pattern as vllm

def test_setup_rdma(orch, variant_config, lifecycle, request):
    """Copy RDMA library into container (thor2 NIC only)."""
    if not variant_config.training.distributed:
        pytest.skip("single-node: RDMA not needed")
    if variant_config.training.nic_type != "thor2":
        pytest.skip("non-thor2 NIC: RDMA lib copy not needed")
    job.setup_rdma_lib()

def test_setup_nic(orch, variant_config, lifecycle, request):
    """Run NIC setup scripts (distributed only)."""
    if not variant_config.training.distributed:
        pytest.skip("single-node: NIC setup not needed")
    job.exec_nic_setup_scripts()

def test_setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Download HF tokenizer into models dir."""
    job.setup_tokenizer()

def test_training_run(orch, variant_config, hf_token, training_res_dict, lifecycle, request):
    """Build training command, start training, poll for completion, parse results."""
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_training_env()
    job.build_training_cmd()
    t = time.monotonic()
    job.start_training()
    job.poll_for_completion()
    wall_time = time.monotonic() - t
    results = job.parse_results()
    results["training.wall_time_seconds"] = wall_time
    # Store per-step metrics for loss curve (AM-32)
    results["training.step_metrics"] = job.step_metrics
    # Convergence metric (AM-33): steps to reach stable loss
    results["training.convergence_steps"] = job.training.steps
    results["training.convergence_wall_time"] = wall_time
    training_res_dict["results"] = results
    training_res_dict["step_metrics"] = job.step_metrics

def test_metric(metric, training_res_dict, variant_config, lifecycle, request):
    """One HTML row per training metric. Record-only unless enforce_thresholds."""
    results = training_res_dict.get("results", {})
    full = "training." + metric
    value = results.get(full)
    unit = dict(TRAINING_METRICS).get(metric, "-")
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))
    if not variant_config.enforce_thresholds:
        return
    cell = variant_config.cell_key()
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if spec is None:
        return
    evaluate_all(results, {full: spec})

def test_teardown(orch, lifecycle, request):
    """Tear down the container."""
    # Same pattern as vllm
```

### 5.6 `_shared.py` — results table

```python
def test_print_results_table(training_res_dict):
    """Print a summary table of training results."""
    results = training_res_dict.get("results", {})
    if not results:
        log.info("training_res_dict empty, nothing to print")
        return
    headers = ["Metric", "Value", "Unit"]
    rows = []
    for short, unit in TRAINING_METRICS:
        full = "training." + short
        val = results.get(full)
        rows.append([short, f"{val:.4f}" if isinstance(val, float) else str(val), unit])
    log.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    # Loss curve summary (AM-32)
    step_metrics = training_res_dict.get("step_metrics", [])
    if step_metrics:
        loss_rows = [[s["step"], f"{s['loss']:.6f}"] for s in step_metrics if "loss" in s]
        if loss_rows:
            log.info("\n\nLoss Curve:\n" + tabulate(loss_rows, headers=["Step", "Loss"], tablefmt="github"))
```

---

## 6. How `cvs run` invocation works

```bash
# Distributed (multi-node) run
cvs run jax_maxtext_training \
  --cluster_file ./cluster.json \
  --config_file ./input/config_file/training/jax_maxtext/mi300x_jax-maxtext_llama-3.3-70b_distributed_config.json \
  --html=report.html --self-contained-html --capture=tee-sys

# Single-node run (same test file, different config)
cvs run jax_maxtext_training \
  --cluster_file ./cluster.json \
  --config_file ./input/config_file/training/jax_maxtext/mi300x_jax-maxtext_llama-3.3-70b_single_config.json \
  --html=report.html --self-contained-html --capture=tee-sys
```

The test file is the same (`jax_maxtext_training.py`); only the `--config_file`
differs. The config's `training.distributed` field drives skipping of
distributed-only stages (sshd, RDMA, NIC setup, JAX coordinator env vars).

---

## 7. Key design decisions

### 7.1 Orchestrator instead of raw Pssh

The existing `jax_training_lib.py` + `jax_llama3_1_70b_distributed.py` use raw
`Pssh` handles and `docker_lib` directly. The new implementation uses the
`ContainerOrchestrator` via the `orch` fixture:

| Old (existing jax tests) | New (jax_maxtext) |
|---|---|
| `docker_lib.launch_docker_container(phdl, ...)` | `orch.setup_containers()` |
| `docker_lib.kill_docker_container(phdl, ...)` | `orch.teardown_containers()` |
| `phdl.exec("docker exec ...")` | `orch.exec(cmd)` (auto-routes into container) |
| Container config in `training_dict` (untyped) | Container config in typed `ContainerSpec` |
| `phdl.exec("sudo ufw stop")` on bare host | Not needed (orch handles network) |

### 7.2 No external shell scripts

The training command is built in Python. The env script and MaxText YAML config
are written into the container by the driver. This makes runs self-contained and
eliminates the dependency on scripts from the MAD repo.

MaxText config YAML values are sourced from the config JSON's
`training.maxtext_config` dict. The driver serializes this dict into a `.yml`
file inside the container at runtime.

### 7.3 Config-driven model parameters

All model-specific parameters (batch size, parallelism, attention, remat) live
in the config JSON's `training.maxtext_config` section. No model parameters are
hardcoded in the library. To add a new model, create a new config+threshold pair.

### 7.4 Thresholds separated from config

Performance thresholds (`tflops_per_sec_per_gpu`, `tokens_per_sec_per_gpu`,
`final_loss`, `loss_decreased`) are in the `*_threshold.json` file, not in the
config. This matches the vllm_single pattern and enables calibrating thresholds
independently of the run config.

### 7.5 Loss curve and convergence metrics

For AM-32 (Loss Curve): the driver captures per-step loss values from the
training log and stores them in `training_res_dict["step_metrics"]`. The results
table prints the loss curve. A `training.loss_decreased` metric (1 = loss
decreased from first to last step) is computed and gated.

For AM-33 (Convergence): total steps and wall-clock time are recorded as
`training.convergence_steps` and `training.convergence_wall_time`. In Phase 2,
a "time to target loss" metric can be added by scanning the step metrics for
when loss drops below a target.

---

## 8. Implementation order

| Step | Task | Files |
|---|---|---|
| 1 | Create package directories + `__init__.py` | `cvs/lib/training/`, `cvs/lib/training/utils/`, `cvs/tests/training/jax_maxtext/` |
| 2 | Write config loader with pydantic models | `cvs/lib/training/utils/training_config_loader.py` |
| 3 | Write metric parsing module | `cvs/lib/training/utils/maxtext_parsing.py` |
| 4 | Write the driver class | `cvs/lib/training/jax_maxtext_training_lib.py` |
| 5 | Write suite conftest (fixtures, ordering, HTML hooks) | `cvs/tests/training/jax_maxtext/conftest.py` |
| 6 | Write the test module | `cvs/tests/training/jax_maxtext/jax_maxtext_training.py` |
| 7 | Write the shared results table | `cvs/tests/training/jax_maxtext/_shared.py` |
| 8 | Write MI300X distributed config + threshold pair | `cvs/input/config_file/training/jax_maxtext/mi300x_jax-maxtext_llama-3.3-70b_distributed_{config,threshold}.json` |
| 9 | Write MI300X single-node config + threshold pair | `cvs/input/config_file/training/jax_maxtext/mi300x_jax-maxtext_llama-3.3-70b_single_{config,threshold}.json` |
| 10 | Write MI325X and MI355X config + threshold pairs | Remaining platform configs |
| 11 | Run `make fmt && make lint` and fix issues | All new files |
| 12 | Add unit tests for config loader and parsing | `cvs/lib/training/utils/unittests/` |
| 13 | Run `make test` to verify unit tests + CLI smoke tests pass | — |

---

## 9. Existing files NOT modified

Per the requirements, the following existing files are left untouched:

- `cvs/lib/jax_training_lib.py` — the old JAX training library (Pssh-based)
- `cvs/tests/training/jax/jax_llama3_1_70b_*.py` — the old JAX test files
- `cvs/input/config_file/training/jax/mi300x_jax_llama3_1_*.json` — the old configs

The new implementation is entirely parallel, under `jax_maxtext/` directories.

---

## 10. Phase 2–4 outlook

| Phase | Scope | What to add |
|---|---|---|
| Phase 2 | Remaining performance + accuracy metrics | PM-23 (time per step p50/p95), PM-24 (scaling efficiency), PM-25 (MFU), PM-28 (checkpoint I/O), AM-30 (perplexity), AM-34 (validation loss) — add to `TRAINING_METRICS`, extend parsing, add threshold specs |
| Phase 3 | Remaining P1 training framework tests | FW-1 (smoke test — 10 steps), FW-3 (multi-GPU scaling), FW-5 (checkpoint save+resume) — new test functions or parametrized cases |
| Phase 4 | Remaining P1 workload coverage | WL-7 through WL-20 — new config+threshold JSON pairs per model/platform; no code changes needed if config schema covers the model params |
