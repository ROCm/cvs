# PyTorch XDit Inference Suite (single-node and distributed)

Cluster validation suite that runs **PyTorch XDit** diffusion inference benchmarks on AMD
Instinct GPUs (single-node scale-out or unified multi-node distributed) and gates each
run on artifact presence, docker exit codes, and GPU-specific latency thresholds.

Library code: `cvs/lib/inference/xdit/` (`pytorch_xdit_flux_job.py`,
`pytorch_xdit_wan_job.py`, `pytorch_xdit_benchmark_job.py`, and output parsers).

> **Use GPU compute nodes.** Targets must have ROCm device nodes (for example `/dev/kfd`),
> Docker, and the configured `container_image` pulled locally. Do not run these benchmarks
> on login or management nodes without GPUs.

## Overview

The suite drives a docker+torchrun benchmark inside `amdsiloai/pytorch-xdit` (or a
compatible image) on one or more cluster nodes, then parses output artifacts and compares
latency against per-GPU thresholds. It provides:

1. **Six suites** - FLUX and WAN 2.2 I2V, each in single-node and distributed variants;
   WAN also has Diffusers/xFuser single and distributed suites.
2. **Two execution modes** - **single-node** runs one independent job per node in
   `cluster.json` / `s_phdl.host_list`; **distributed** runs one coordinated torchrun
   job across `config.nnodes` with distinct `--node_rank`.
3. **Offline model staging** - models must be present on every participating node before
   the run (explicit host path or pre-populated Hugging Face cache under `hf_home`).
4. **Preflight checks** - `/dev/kfd`, local container image, parallelism config
   (distributed), WAN xFuser bind-mount validation, and FLUX.2 `flux2_example.py`
   fallback mount when the image does not ship `/app/external/xdit/examples/flux2_example.py`.
5. **Exit-code gating** - benchmark pass/fail uses docker `exit_code` from
   `exec(..., detailed=True)`, not log-regex scanning.
6. **Threshold gating** - average FLUX `pipe_time` or WAN `total_time` compared to
   `expected_results` for the auto-detected GPU type (`mi300x`, `mi350`, `mi355`, or
   `auto`).

Single-node output dirs use the cluster **SSH target** string from `node_dict`, not the
remote `hostname` (`flux_<target>_outputs`, `wan_22_<target>_outputs`).

## Quick Start

Single-node FLUX.1-dev:

```bash
cvs run pytorch_xdit_flux_dev_single \
  --cluster_file ./p3_1n_cluster.json \
  --config_file cvs/input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
  --html ./logs/pytorch_xdit_flux_single.html --self-contained-html -vvv
```

Distributed FLUX (multi-node unified torchrun):

```bash
cvs run pytorch_xdit_flux_dev_distributed \
  --cluster_file ./p3_2n_cluster.json \
  --config_file cvs/input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_distributed.json \
  --html ./logs/pytorch_xdit_flux_distributed.html --self-contained-html -vvv
```

Single-node WAN 2.2 I2V (native checkpoint layout):

```bash
cvs run pytorch_xdit_wan22_14b_single \
  --cluster_file ./p3_1n_cluster.json \
  --config_file cvs/input/config_file/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json \
  --html ./logs/pytorch_xdit_wan_single.html --self-contained-html -vvv
```

- `--cluster_file` - JSON describing node(s), SSH credentials, and optional `env_vars`.
- `--config_file` - one of the templates under
  `cvs/input/config_file/inference/xdit/`. Replace every `<changeme>` (especially
  in distributed NCCL/network fields) before running.
- `--html` / `--self-contained-html` - pytest HTML report for the suite run.

> Use a **single-node** config with `pytorch_xdit_*_single` suites and a **distributed**
> config (with `nnodes >= 2` and matching parallel degrees) with `pytorch_xdit_*_distributed`
> suites. FLUX.1-dev, FLUX.2-dev, and WAN model family are selected via `model_repo`,
> local `model_index.json`, or benchmark params — the same suite file covers each.

On shared clusters, skip aggressive docker prune during cleanup:

```bash
export CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1
```

## The six suites

| Suite (`cvs run <name>`) | File | Model / launcher | Mode |
|---|---|---|---|
| `pytorch_xdit_flux_dev_single` | `pytorch_xdit_flux_dev_single.py` | FLUX.1 (`run_usp.py`) or FLUX.2 (`flux2_example.py`) | one job per node in cluster |
| `pytorch_xdit_flux_dev_distributed` | `pytorch_xdit_flux_dev_distributed.py` | FLUX.1 / FLUX.2 unified torchrun | `nnodes >= 2`, shared rank-0 output |
| `pytorch_xdit_wan22_14b_single` | `pytorch_xdit_wan22_14b_single.py` | WAN 2.2 I2V native (`/app/Wan2.2/run.py`) | one job per node |
| `pytorch_xdit_wan22_14b_distributed` | `pytorch_xdit_wan22_14b_distributed.py` | WAN 2.2 native unified torchrun | `nnodes >= 2` |
| `pytorch_xdit_wan22_14b_diffusers_single` | `pytorch_xdit_wan22_14b_diffusers_single.py` | WAN Diffusers xFuser (`wan_i2v_example.py`) | one job per node |
| `pytorch_xdit_wan22_14b_diffusers_distributed` | `pytorch_xdit_wan22_14b_diffusers_distributed.py` | WAN Diffusers xFuser unified torchrun | `nnodes >= 2` |

List suites and test functions:

```bash
cvs list
cvs list pytorch_xdit_flux_dev_single
```

## Test lifecycle (report rows)

Tests run in this order within each suite file.

| Order | Test | Single | Distributed | Purpose |
|---|---|---|---|---|
| 1 | `test_cleanup_stale_containers` | yes | yes | Stop named benchmark container; optional `docker system prune` |
| 2 | `test_verify_hf_cache_or_download` | FLUX, WAN native | FLUX, WAN native | Offline model presence on every node |
| 2 | `test_verify_model_on_nodes` | WAN Diffusers | WAN Diffusers | Model / mount preflight for Diffusers layout |
| 3 | `test_verify_parallelism_config` | no | yes | `ulysses × ring × … == nnodes × torchrun_nproc` |
| 4 | `test_run_*_benchmark` | yes | yes | docker+torchrun benchmark; gates on exit code |
| 5 | `test_parse_and_validate_results` | yes | yes | Parse artifacts, average latency, threshold PASS/FAIL |

Distributed suites scope `s_phdl` to `server_node_list` / `nnodes` participating nodes,
not necessarily every entry in `cluster.json`.

## Metrics and PASS/FAIL

GPU type is detected once per suite from `rocm-smi` on rank-0 (or the sole node).
Threshold lookup order: exact GPU key → `auto`.

| Model family | Parsed metric | Threshold key | Required artifacts |
|---|---|---|---|
| FLUX | average `pipe_time` from `results/timing.json` | `max_avg_pipe_time_s` | non-empty `timing.json`, at least one `flux_*.png` |
| WAN native | average `total_time` from `rank0_step*.json` | `max_avg_total_time_s` | step JSONs, `video.mp4` (recursive search) |
| WAN Diffusers xFuser | average epoch time from `results/timing.json` | `max_avg_total_time_s` | `results/timing.json`, `results/video_i2v.mp4` |

A suite **passes** when model preflight succeeds, the benchmark exits 0 on every
participating node, expected artifacts exist, and the averaged metric is at or below the
configured threshold.

A suite **fails** when model paths are missing, preflight checks fail, docker exits
non-zero, artifacts are missing/empty, or latency exceeds the threshold.

Sample thresholds in-repo are starting points; tune `expected_results` for your hardware,
software stack, and benchmark settings before production gating.

## Output layout

**FLUX** (single-node example):

```
${output_base_dir}/flux_<cluster_target>_outputs/
└── results/
    ├── timing.json
    └── flux_*.png
```

**WAN native**:

```
${output_base_dir}/wan_22_<cluster_target>_outputs/
└── outputs/.../rank0_step*.json, video.mp4
```

**WAN Diffusers xFuser**:

```
${output_base_dir}/wan_22_<cluster_target>_outputs/
└── results/timing.json, results/video_i2v.mp4
```

Distributed runs write to `flux_<rank0_target>_outputs` or `wan_22_<rank0_target>_outputs`
shared across ranks.

## Config files

Templates live in `cvs/input/config_file/inference/xdit/`:

| Config file | Typical suite |
|---|---|
| `mi3xx_pytorch_xdit_flux1_dev_single.json` | `pytorch_xdit_flux_dev_single` |
| `mi3xx_pytorch_xdit_flux1_dev_distributed.json` | `pytorch_xdit_flux_dev_distributed` |
| `mi3xx_pytorch_xdit_flux2_dev_single.json` | `pytorch_xdit_flux_dev_single` |
| `mi3xx_pytorch_xdit_flux2_dev_distributed.json` | `pytorch_xdit_flux_dev_distributed` |
| `mi3xx_pytorch_xdit_wan22_14b_single.json` | `pytorch_xdit_wan22_14b_single` |
| `mi3xx_pytorch_xdit_wan22_14b_distributed.json` | `pytorch_xdit_wan22_14b_distributed` |
| `mi3xx_pytorch_xdit_wan22_14b_diffusers_single.json` | `pytorch_xdit_wan22_14b_diffusers_single` |
| `mi3xx_pytorch_xdit_wan22_14b_diffusers_distributed.json` | `pytorch_xdit_wan22_14b_diffusers_distributed` |

**Placeholders** (test JSON): `{user-id}`, `{user}`, `{home}` — resolved at startup.
Cluster JSON resolves `{user-id}` only; use real absolute paths for `priv_key_file` when
`/home/{user-id}/.ssh/id_rsa` is wrong on your system.

**Key config fields:**

- `hf_home` - host Hugging Face cache root (mounted at `/hf_home`); must contain `hub/`
  when using repo-id + offline cache mode. See
  [HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
- `model_repo` - Hugging Face repo id or absolute on-disk model path (preferred at scale).
- `output_base_dir` - host directory for benchmark outputs.
- `container_config.device_list` - typically `["/dev/dri", "/dev/kfd"]`.
- `container_config.volume_dict` - optional host:container bind mounts. WAN Diffusers
  xFuser requires mounting `cvs/lib/inference/xdit/scripts/wan_i2v_example.py`.
  FLUX.2 probes `/app/external/xdit/examples/flux2_example.py` in the image and, when
  it is missing, bind-mounts `cvs/lib/inference/xdit/scripts/flux2_example.py` to
  `/benchmark/flux2_example.py` (same pattern as WAN xFuser). Flux2 sample configs
  already set this mapping in `volume_dict`.
- `nnodes`, `master_addr`, `nccl_*`, `gloo_socket_ifname` - distributed rendezvous and
  NCCL tuning (replace `<changeme>` values).
- `benchmark_params.flux1_dev_t2i` or `benchmark_params.wan22_i2v_a14b` - torchrun
  parallelism, warmup/repetition counts, and `expected_results`.

Configs are validated through Pydantic schemas (`PytorchXditFluxConfigFile`,
`PytorchXditWanConfigFile`) at load time.

## First-time setup

1. SSH to each compute node (or a shared staging host with cluster-visible storage).
2. Stage models under a real `hf_home` or bind-mount path (`hf download` or rsync).
3. `docker pull` the configured `container_image` on every execution node.
4. Create `cluster.json` with the same SSH target string for `mgmt_ip`, `node_dict` key,
   and `vpc_ip`.
5. Run with `CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1` on shared systems.

**Model download examples** (on the node):

```bash
export HF_HOME="$HOME/.cache/huggingface"
hf download black-forest-labs/FLUX.1-dev
hf download Wan-AI/Wan2.2-I2V-A14B --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3
```

**Sanity checks:**

```bash
test -e /dev/kfd && echo KFD_OK
docker image inspect amdsiloai/pytorch-xdit:v25.11.2 >/dev/null && echo IMG_OK
```

**Multi-node storage:** `hf_home` is verified independently per node. If `$HOME` is not
shared across the cluster, each node needs its own full model copy unless you mount a
shared path via `container_config.volume_dict`.

## Prerequisites

- Passwordless SSH from the control host to each execution node (key in cluster file).
- Docker on every execution node with the benchmark image pre-pulled.
- ROCm GPUs with `/dev/kfd` and `/dev/dri` available to the container.
- Models staged offline on every participating node (no runtime Hugging Face downloads
  at scale).
- Hugging Face token file at `hf_token_file` when using gated models or FLUX.2 chat
  template fetch paths.
- Distributed runs: NCCL/RDMA network settings matching the cluster; parallel-degree
  product must equal `nnodes × torchrun_nproc`.

## Operational notes

- **Cleanup:** `test_cleanup_stale_containers` kills the named container and runs
  `docker system prune --force` unless `CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1`.
- **SSH retries:** long benchmark sessions may hit stale SSH clients; `Pssh` retries once
  on `SessionError`.
- **Local single-node:** when the sole cluster target resolves to localhost, single suites
  may use a `LocalPssh` path instead of SSH.
- **WAN xFuser:** Diffusers suites may require bind-mounting `wan_i2v_example.py` and an
  I2V input image in `volume_dict`, or enabling auto-generated in-container input.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `/dev/kfd not found` | non-GPU or wrong node class | run on GPU compute nodes |
| `Container image not found locally` | image not pulled | `docker pull` on each node |
| `Local model path not found` | model not staged | rsync/`hf download` to every node |
| `Parallel degree product != world_size` | config mismatch | align ulysses/ring/… with `nnodes × nproc` |
| Threshold exceeded | slow hardware or wrong baseline | tune workload or `expected_results` |
| Missing `video.mp4` / `timing.json` | benchmark failed mid-run | inspect benchmark log tail on failing node |
