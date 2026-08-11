# ATOM Inference — Config and Threshold Files

This folder holds the input files for the `atom` suite (see
`cvs/tests/inference/atom/README.md` for how to run it). Each **config** file
has a sibling **threshold** file (referenced by its `threshold_json` field).
One config = one GPU arch + topology + driver mode (single-node, multinode PP,
baseline matrix, …).

W1 workloads target **DeepSeek R1 FP8** on 8× GPU per node (ISL/OSL sweeps,
TP8 unless noted).

## File inventory

In the CVS repo, variants are flat sibling pairs in **this directory**:

```text
{gpu}_atom_{model}_{precision}[_{mode}].json
{gpu}_atom_{model}_{precision}[_{mode}]_threshold.json
```

| Config | Threshold | GPU | Driver | Notes |
|---|---|---|---|---|
| `mi300x_atom_deepseek-r1_fp8_single` | `…_single_threshold.json` | MI300X | `atom` | W1 single-node; portable min-SLO thresholds; server reuse |
| `mi300x_atom_deepseek-r1_fp8_baseline_sweep` | `…_baseline_sweep_threshold.json` | MI300X | `atom` | DTNI baseline: 1K/1K + 8K/1K × C=4–256 (14 cells) |
| `mi300x_atom_deepseek-r1_fp8_baseline_sweep_distributed` | `…_baseline_sweep_distributed_threshold.json` | MI300X | `vllm_atom` | 2-node DTNI baseline (14 cells); PP=2, scaling gates |
| `mi300x_atom_deepseek-r1_fp8_distributed` | `…_distributed_threshold.json` | MI300X | `vllm_atom` | W1 2-node PP=2; lab-calibrated thresholds |
| `mi300x_atom_deepseek-r1_fp8_sglang_distributed` | `…_sglang_distributed_threshold.json` | MI300X | `sglang` | W1 2-node PP=2 SGLang path; record-only until lab confirm |
| `mi300x_atom_deepseek-r1_fp8_mtp3` | `…_mtp3_threshold.json` | MI300X | `atom` | W1 FP8 + MTP3 |
| `mi355x_atom_deepseek-r1_fp8_single` | `…_single_threshold.json` | MI355X | `atom` | W1 single-node; CI seeds, `enforce_thresholds: false` |
| `mi355x_atom_deepseek-r1_fp8_baseline_sweep` | `…_baseline_sweep_threshold.json` | MI355X | `atom` | DTNI baseline matrix; record-only |
| `mi355x_atom_deepseek-r1_fp8_distributed` | `…_distributed_threshold.json` | MI355X | `vllm_atom` | W1 2-node PP=2; record-only until lab confirm |
| `mi355x_atom_deepseek-r1_fp8_mtp3` | `…_mtp3_threshold.json` | MI355X | `atom` | W1 FP8 + MTP3 |
| `mi300x_atom_gpt-oss-120b_bf16` | `…_threshold.json` | MI300X | `vllm` | GPT-OSS uplift placeholder |
| `mi355x_atom_gpt-oss-120b_bf16` | `…_threshold.json` | MI355X | `vllm` | GPT-OSS uplift placeholder |

Add analogous config + threshold pairs for other archs or models as needed.

Keys prefixed with `_` (e.g. `_comment`) are inline comments and are ignored by
the loader.

**Removed:** `*_smoke` variants — use `-k` on a full variant for a one-cell smoke.
**Removed:** bare `driver=atom` multinode PP — use `vllm_atom` or `sglang`
distributed stems above.

## What you MUST change for your cluster / setup

Start from the config closest to your target GPU / topology / driver and edit
these:

| Where | Variable | Change to |
|---|---|---|
| `container.image` | container image | Your ATOM / vLLM+ATOM / SGLang ROCm image on the nodes |
| `container.name` | container name | Any unique name (optional) |
| `paths.shared_fs` | base path | A path reachable from all nodes; `{user-id}` resolves to the cluster/OS user |
| `paths.models_dir` | model cache | Host path to the staged model (shipped configs use `/home/models`) |
| `paths.log_dir` | benchmark logs | Usually `{shared_fs}/LOGS` |
| `paths.hf_token_file` | HF token path | Location of your Hugging Face token file |
| `model.id` | model repo id | The model under test (W1: `deepseek-ai/DeepSeek-R1-0528`) |
| `model.remote` | fetch mode | `0` = already cached on nodes; `1` = not implemented |
| `params.driver` | execution stack | `atom`, `vllm_atom`, `sglang`, or interim `vllm` (see below) |
| `params.nnodes` | node count | `1` single-node; `2` for shipped multinode PP variants |
| `params.master_addr` | PP coordinator | Head node VPC IP (replace `{head-node-ip}` on multinode stems) |
| `params.master_port` | PP coordinator port | Usually `29501` |
| `params.pipeline_parallel_size` | PP size | `2` on shipped multinode stems |
| `params.scaling_baseline_output_throughput` | 1-node baseline | Measured single-node output tok/s for `scaling.efficiency_pct` (multinode) |
| `roles.server.atom_args` | ATOM server CLI | Tokens after `--model` / `--server-port` when `driver=atom` |
| `roles.server.serve_args` | vLLM serve flags | Dict of `vllm serve` flags when `driver=vllm_atom` or `vllm` |
| `roles.server.sglang_args` | SGLang server CLI | Args for `sglang.launch_server` when `driver=sglang` |
| `roles.server.ib_hca_devices` | RDMA HCAs | `"auto"` (default) or explicit list; probed in `test_discover_topology` |
| `roles.server.ib_netdev` | socket netdev | `"auto"` (default on distributed) or explicit name; **not** `mlx5_*` |
| `roles.server.env` | server env | Driver-specific env (ATOM mmap, vLLM AITER flags, …) |
| `.json` gated values | thresholds | Calibrated PASS/FAIL bounds for your hardware |
| cluster file `node_dict` | node IPs | Your node IPs; **host count must equal `params.nnodes`** |
| cluster template | `atom_cluster.json` | Copy from `cvs/input/cluster_file/atom_cluster.json` |

Also set `enforce_thresholds` to `true` for real PASS/FAIL or `false` for
record-only (MI355X stems ship record-only until lab calibration).

### Lab directory layout

On your lab machine (`~/input/config_file/inference/atom/`), copy each variant
into its **own subdirectory** so threshold discovery is unambiguous:

```text
~/input/.../atom/single/              # single-node config + threshold only
~/input/.../atom/distributed/         # vllm_atom PP=2
~/input/.../atom/sglang_distributed/  # sglang PP=2
~/input/.../atom/baseline_sweep/      # DTNI single-node matrix
```

`substitute_config` globs the config's parent directory; multiple `*threshold.json`
files in one folder raises `ValueError: multiple *threshold.json files … (ambiguous)`.

Each shipped config sets `"threshold_json"` to the sibling threshold filename
(relative to the config directory). You may also use an absolute path.

## Placeholder substitution

Configs use placeholders resolved at load time:

- `{user-id}` — the cluster username (or the local OS user as fallback).
- `{shared_fs}` — self-reference within the `paths` block.
- `{paths.models_dir}` (and other `{paths.*}`) — cross-referenced anywhere.
- `{head-node-ip}` — replace manually in copied multinode configs (not auto-resolved).

`threshold_json` is a literal filename resolved next to the config; no
placeholder substitution is applied to it.

## Config structure

Top-level (framework-agnostic) fields:

| Field | Meaning |
|---|---|
| `schema_version` | Always `1` |
| `framework` | `atom` |
| `gpu_arch` | `mi300x` / `mi355x` (labels the run) |
| `enforce_thresholds` | `true` = metrics gate PASS/FAIL; `false` = record-only |
| `threshold_json` | Sibling threshold filename |
| `run_card` | Optional metadata (`atom_image_pin`, `upstream_run_url`, `notes`) logged at session start |
| `paths` | `shared_fs`, `models_dir`, `log_dir`, `hf_token_file` |
| `model` | `id`, `remote` (0 = already cached), `precision` (label) |
| `container` | `lifetime`, `name`, `image`, `runtime` (docker `args`: network/ipc/privileged/shm-size/volumes/devices) |

### `params` block

| Field | Meaning |
|---|---|
| `driver` | `atom`, `vllm_atom`, `sglang`, or interim `vllm` |
| `tensor_parallelism` | TP size (W1: `8`) |
| `pipeline_parallel_size` | PP size (`1` single-node; `2` on multinode stems) |
| `nnodes` | Node count (`1` or `2` on shipped variants) |
| `master_addr` / `master_port` | Multinode coordinator (vLLM / SGLang) |
| `port_no` | Server HTTP port (default `8000`) |
| `num_prompts` | Benchmark prompt count per cell |
| `max_model_length` | Server MML; must cover `(ISL+OSL) × (1+random_range_ratio)` |
| `random_range_ratio` | Random workload ratio passed to bench client |
| `metric_percentiles` | Tail percentiles requested (e.g. `95,99`) |
| `reuse_server_across_sweep` | `true` = keep server warm across cells with matching session key |
| `scaling_baseline_output_throughput` | Single-node output tok/s baseline for `scaling.efficiency_pct` |
| `bench_extra_args` | Extra bench client tokens (MTP3 variants) |
| `server_*` / `client_*` poll waits | Timeouts for server ready and client completion |

### `roles.server` block

| Field | Meaning |
|---|---|
| `atom_args` | Extra CLI tokens for `python -m atom.entrypoints.openai_server` (`driver=atom`) |
| `serve_args` | Dict merged into `vllm serve` argv (`driver=vllm_atom` / `vllm`) |
| `sglang_args` | Args for `sglang.launch_server` (`driver=sglang`) |
| `env` | Exported in `/tmp/server_env_script.sh` (orchestrator-managed NCCL keys are stripped) |
| `ib_hca_devices` | `"auto"` or explicit HCA list for `NCCL_IB_HCA` |
| `ib_netdev` | `"auto"` or explicit socket interface for `NCCL/GLOO/TP_SOCKET_IFNAME` |

### Execution drivers (`params.driver`)

Standalone ATOM has **no native pipeline parallel**. Multinode PP requires a
framework coordinator:

| Driver | When to use | Server | Multinode PP |
|---|---|---|---|
| `atom` | Single-node W1, baseline sweep, MTP3 | `atom.entrypoints.openai_server` | No |
| `vllm_atom` | Shipped 2-node PP stems | `vllm serve` + ATOM ROCm env | Yes |
| `sglang` | Shipped SGLang multinode stem | `sglang.launch_server` | Yes |
| `vllm` | GPT-OSS placeholder | `vllm serve` | When PP flags configured |

Multinode fabric is probed once per run in `test_discover_topology` (on the
**cluster host OS**, not inside the container). Probes can be skipped on
single-node (`nnodes=1`). Lazy resolution also runs on first `build_server_cmd`
if topology discovery is omitted via a smoke `-k` filter.

### `sweep` block

| Field | Meaning |
|---|---|
| `sequence_combinations` | Named `(isl, osl)` shapes, e.g. `{name, isl, osl}` |
| `runs` | `{combo, concurrency}` pairs — one benchmark cell each |

Each run produces a **threshold cell key** via `cell_key()`:

- Single-node: `ISL=1024,OSL=1024,TP=8,CONC=128`
- Multinode PP: `ISL=1024,OSL=1024,TP=8,PP=2,NNODES=2,CONC=128`

The key must match a top-level entry in the threshold file. Parametrize IDs in
pytest look like `w1_1k_1k-conc128` or `w1_1k_1k-conc128-throughput` (metric
tier suffix on gate rows).

## Threshold files

A threshold file maps each **cell key** to `{metric: spec}`. Metrics use the
`client.*` namespace (plus `scaling.efficiency_pct` on multinode). A metric is
gated only when `enforce_thresholds: true`, the metric belongs to the tier under
test, and a spec exists; otherwise it is recorded. Metrics missing from the
benchmark artifact are skipped (ATOM may omit tail percentiles).

Threshold kinds:

| kind | Passes when | Notes |
|---|---|---|
| `min` | `actual >= value` | lower bound (e.g. success_rate) |
| `max` | `actual <= value` | upper bound (e.g. failed count) |
| `max_ms` | `actual <= value` | upper bound, `ms` in the message |
| `min_tok_s` | `actual >= value` | lower bound, `tok/s` in the message |
| `within` | `value +/- tolerance_pct%` | needs `tolerance_pct` |
| `min_ratio` | `actual / actuals[reference] >= value` | needs `reference` |
| `info` | always | record-only; retains a default `value` to calibrate later |

Example single-node cell:

```json
"ISL=1024,OSL=1024,TP=8,CONC=128": {
  "client.total_token_throughput": { "kind": "min_tok_s", "value": 3000 },
  "client.output_throughput":        { "kind": "min_tok_s", "value": 1500 },
  "client.per_gpu_throughput":       { "kind": "min_tok_s", "value": 375 },
  "client.p99_ttft_ms":              { "kind": "max_ms", "value": 1000000 },
  "client.success_rate":             { "kind": "min", "value": 1 },
  "client.failed":                   { "kind": "max", "value": 0 }
}
```

Example multinode cell (adds scaling):

```json
"ISL=1024,OSL=1024,TP=8,PP=2,NNODES=2,CONC=128": {
  "client.output_throughput":        { "kind": "min_tok_s", "value": 2500 },
  "client.p99_ttft_ms":              { "kind": "max_ms", "value": 5000 },
  "scaling.efficiency_pct":          { "kind": "min", "value": 80 }
}
```

To start gating a metric currently marked `info`: replace `"kind": "info"` with
`min`/`max`/etc. and set a calibrated `value`. The threshold cell key must match
the sweep cell exactly (including `PP` and `NNODES` on multinode), or the metric
falls back to record-only.

## Cluster file

Template: `cvs/input/cluster_file/atom_cluster.json`. Copy to
`~/input/cluster_file/atom_cluster.json` and edit IPs, `username`, and
`priv_key_file`.

| Variant type | `params.nnodes` | `node_dict` |
|---|---|---|
| Single-node (`*_single`, baseline sweep, MTP3) | `1` | Head node only |
| Multinode PP (`*_distributed`, `*_baseline_sweep_distributed`, `*_sglang_distributed`) | `2` | Head + worker |

`test_setup_sshd` runs when `len(node_dict) > 1`.

## Running on a lab machine

**Launcher vs GPU node:** CVS pytest runs on the launcher;
`ContainerOrchestrator` SSHes to cluster nodes and runs `sudo docker` there.

| Item | Launcher | GPU node |
|---|---|---|
| `cvs run`, venv, `~/input/`, `~/cvs_results/` | Yes | No |
| `priv_key_file`, HF token file | Yes | No |
| `/home/models` (when `model.remote: 0`) | No | Yes |
| Container image, `sudo docker` | No | Yes |
| `~/LOGS/` (via volume mount) | No | Yes |

After `git pull`, run **`make install` first**, then **`source .cvs_venv/bin/activate`**
(do not activate the venv before `make install`).

Typical workflow:

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

SINGLE_DIR=~/input/config_file/inference/atom/single
mkdir -p "$SINGLE_DIR"

cvs copy-config inference/atom/mi300x_atom_deepseek-r1_fp8_single.json \
  --output "$SINGLE_DIR/mi300x_atom_deepseek-r1_fp8_single.json"
cvs copy-config inference/atom/mi300x_atom_deepseek-r1_fp8_single_threshold.json \
  --output "$SINGLE_DIR/mi300x_atom_deepseek-r1_fp8_single_threshold.json"
cvs copy-config atom_cluster.json --output ~/input/cluster_file/atom_cluster.json

# Edit cluster IPs; trim node_dict to one host for single-node variants.

TS=$(date +%Y%m%d_%H%M%S)
cvs run atom \
  --cluster_file ~/input/cluster_file/atom_cluster.json \
  --config_file "$SINGLE_DIR/mi300x_atom_deepseek-r1_fp8_single.json" \
  --html=~/cvs_results/${TS}_atom-w1-single_mi300x.html \
  --self-contained-html \
  --log-file=~/cvs_results/${TS}_atom-w1-single_mi300x.log \
  -vvv -s
```

For multinode PP variants, use a two-host cluster file, copy the matching
`*_distributed*` (or `*_sglang_distributed*`) config pair into its own
subdirectory, set `container.image`, `params.master_addr`, and verify fabric
discovery (or set `roles.server.ib_netdev` explicitly).

Smoke a single cell with `-k`, for example `-k "w1_1k_1k-conc128"`.

When `--html` is set, the **ATOM Run Deck** is generated at session end and
bundled into the pytest zip (render-only; does not affect gates). See
`cvs/lib/report/README.md`.

## Related docs

| Doc | Purpose |
|---|---|
| `cvs/tests/inference/atom/README.md` | Suite lifecycle, sweeps, metric tiers, quick start |
| `cvs/lib/inference/utils/docs/atom-parsing.md` | `client.*` metric vocabulary and tiers |
| `docs/reference/configuration-files/atom.rst` | Published config reference (Sphinx) |
