# Framework topology rules

## Model identification (all frameworks)

**Always ask the user for the HuggingFace model repo ID** before generating or customizing a config. Do not reuse model values from shipped templates.

| Framework | Field | Single-node | Distributed |
|-----------|-------|-------------|-------------|
| pytorch_xdit | `config.model_repo` | HuggingFace repo ID | HuggingFace repo ID if cached, or local path on every node |
| sglang | `benchmark_params.<workload>.model` | HuggingFace repo ID or local path | same |
| vllm / atom | see framework README | HuggingFace repo ID or local path | same |

Set `hf_token_file` when the model is fetched from HuggingFace at runtime. Leave empty or omit when using pre-staged local weights.

---

## PyTorch XDit (WAN / Flux workload types)

Schema: `cvs/parsers/schemas.py` (`PytorchXditWanConfigFile`, `PytorchXditFluxConfigFile`).

Structure: top-level `config` + `benchmark_params`. No `threshold_json` — gates live in `expected_results`.

Workload type (`wan22_i2v_a14b` vs `flux1_dev_t2i`) is a **schema key**, not the model name. Pick the template whose `benchmark_params` key matches the user's workload; set `config.model_repo` to the user's HuggingFace repo ID.

### Single-node

Omit or leave unset:
- `config.nnodes`
- `config.master_addr`, `master_port`
- `config.nccl_*`, `gloo_socket_ifname`

WAN parallelism (1 node × 8 GPUs):
- `ulysses_size`: 8
- `ring_size`: 1
- `torchrun_nproc`: 8

Flux parallelism (1 node × 8 GPUs):
- `ulysses_degree`: 8
- `ring_degree`: 1
- `pipefusion_parallel_degree`: 1 (default)
- `torchrun_nproc`: 8

### Distributed (multi-node torchrun)

Add to `config`:
- `nnodes`: node count (shipped examples use 2)
- `master_addr`: `<changeme>` (head node IP)
- `master_port`: 29500
- `nccl_ib_hca`, `nccl_socket_ifname`, `gloo_socket_ifname`: `<changeme>`
- `nccl_ib_gid_index`: 3 (RoCE; adjust for cluster)
- `nccl_debug`: INFO
- `container_config.env_dict.NCCL_PROTO`: Simple (recommended)

Set `model_repo` to the user's HuggingFace repo ID, or a **local path** on every node when weights are pre-staged (e.g. `/data/models/<user-hf-org>/<user-model>`).

Parallelism must satisfy schema validators:

**WAN** (`nnodes >= 2`):
```text
ulysses_size × ring_size == nnodes × torchrun_nproc
```
Shipped 2-node example: `ulysses_size=8`, `ring_size=2`, `torchrun_nproc=8` → world_size 16.

**Flux** (`nnodes >= 2`):
```text
ulysses_degree × ring_degree × pipefusion_parallel_degree
  × tensor_parallel_degree × data_parallel_degree
  == nnodes × torchrun_nproc
```
Shipped 2-node example: `ulysses_degree=8`, `ring_degree=2`, others 1 → world_size 16.

Relax `expected_results` thresholds for distributed (higher latency is normal).

### Disaggregated

Not supported for PyTorch XDit in CVS today.

### User-must-set fields

| Field | Notes |
|-------|-------|
| `config.model_repo` | **User-provided** HuggingFace repo ID (single); HF repo ID or local path (distributed) |
| `config.master_addr` | Head node IP (distributed) |
| `config.nccl_ib_hca` | RDMA HCAs from `ibv_devices` |
| `config.nccl_socket_ifname` | Control NIC from `ip -br link` |
| `config.gloo_socket_ifname` | Usually same as NCCL socket ifname |
| Path fields | Use `{user-id}` not hardcoded usernames |

Optional: `hf_token_file` may be empty when using pre-staged local models.

---

## vLLM / ATOM (`schema_version: 1`)

Templates: `inference/vllm_mi300x_workloads/`, `inference/atom/`

Each config has a sibling `*_threshold.json` via `threshold_json`.

Replace model path/id fields with the user's HuggingFace repo ID (see framework README).

| Field | single | distributed |
|-------|--------|-------------|
| `params.nnodes` | `"1"` | `"2"` |
| `params.pipeline_parallel_size` | `"1"` | `"2"` |
| `params.master_addr` | absent | `"<changeme>"` |
| `roles.server.ib_*` | absent | present |
| Threshold cell keys | `TP=<tp>,CONC=16` | append `,PP=2` |

ATOM: `params.driver` is `atom` (single) or `vllm_atom` (distributed).

---

## SGLang (legacy nested `config`)

Templates: `inference/sglang/`

Pick a shipped template by **GPU arch and topology**, not model name. Set `benchmark_params.<workload>.model` to the user's HuggingFace repo ID or local path.

| Topology | Key additions |
|----------|---------------|
| single | `nnodes: "1"`, `benchmark_serv_node` only |
| distributed | NCCL fields, multi-node `nnodes`, `server_node_list` |
| disaggregated | `prefill_node_list`, `decode_node_list`, `proxy_router_node`, coordinator addrs/ports |

**Always create two files:** config JSON + threshold JSON sibling. Reference threshold via `benchmark_params.*.threshold_file`.

---

## Training (Megatron, TorchTitan, JaxMaxText)

Templates: `training/<framework>/`

| Field | single | distributed |
|-------|--------|-------------|
| `framework` | `*_single` | `*_distributed` |
| `config.nnodes` | `"1"` | `<changeme>` |
| NCCL / master | localhost defaults | `<changeme>` required |
| `scaling_baseline` | absent | required for scaling metrics |

Set model/checkpoint fields from user input per framework README.

---

## Validation checklist

### All configs
- [ ] JSON parses; keys prefixed with `_` are comments only
- [ ] Model field(s) set to **user-provided** HuggingFace repo ID (not copied from template)
- [ ] No unintended unresolved `<changeme>` (distributed NCCL/master must stay `<changeme>` until user fills)
- [ ] Path fields use `{user-id}` where appropriate
- [ ] Cluster node count matches `nnodes` / node lists

### PyTorch XDit
- [ ] Exactly one of `wan22_i2v_a14b` or `flux1_dev_t2i` under `benchmark_params`
- [ ] `expected_results` has `auto` or a GPU-specific key (`mi300x`, `mi355`, …)
- [ ] Distributed: parallel product equals `nnodes × torchrun_nproc`
- [ ] Validate: `python .claude/skills/cvs-config-generator/scripts/validate_xdit_config.py <config.json>` (same script under `.cursor/skills/...`)

### Threshold-based suites (vLLM, ATOM, SGLang, …)
- [ ] Config and threshold are siblings in one directory
- [ ] Threshold filename matches `threshold_json` or `benchmark_params.*.threshold_file`
- [ ] Sweep `runs` entries have matching threshold cell keys
