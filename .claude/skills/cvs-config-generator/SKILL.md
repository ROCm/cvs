---
name: cvs-config-generator
description: >-
  Generates CVS training and inference config JSON plus matching threshold JSON
  for any engine (sglang, vllm, atom, xdit, megatron, torchtitan, jaxmaxtext)
  from existing templates under cvs/input/config_file. Use when the user mentions
  cvs-config-generator, generating a config and threshold, adding a model
  workload, or writing files under cvs/input/config_file for training or inference.
---

# CVS config generator

Copy the closest existing CVS template pair, then retarget it to the requested
engine, GPU, model, precision, and topology. Do not invent a new schema.

Read [reference.md](reference.md) for engine directories, filename patterns, and
threshold key formats. Read [examples.md](examples.md) for prompt/output pairs.

## Required inputs

Parse from the user prompt (ask only for missing items):

| Field | Examples |
|---|---|
| Engine | sglang, vllm, atom, xdit, megatron, torchtitan, jaxmaxtext |
| Topology | single, distributed (unified TP/PP), disaggregated (PD split) |
| GPU | MI300X, MI325X, MI355X |
| Model | Hugging Face id or local name (`moonshotai/Kimi-K2.6`) |
| Precision | bf16, fp8, mxfp4, w4a8, int4 |
| Parallelism | nnodes, TP, PP (and EP/DP if the template has them) |
| Output paths | if omitted, derive from naming rules |
| Threshold mode | record-only (default) vs calibrated |

**Distributed vs disaggregated:** unified multi-node TP/PP is **distributed**.
Prefill/decode node lists, coordinators, and PD ports are **disaggregated**.
Never mix those skeletons.

## Workflow

1. Map engine + topology to a base **config** and **threshold** template (user
   overrides win). Prefer the same engine, same topology, closest model size,
   then closest GPU family (`mi3xx` configs are valid for MI300X/MI325X).
2. Read both templates fully. Keep unknown keys; delete only fields that belong
   to a different topology (for example PD lists on a distributed config).
3. Retarget model, precision, GPU hints, TP/PP/nnodes, sweep cells, comments,
   `threshold_json` basename, `model_num_params` / peak TFLOPS when present.
4. Leave every cluster-specific value that already contains `<changeme>` as
   `<changeme>` (hosts, NICs, `gpu_name`, container `image`, mounts). Do not
   invent IPs, hostnames, or RDMA device lists.
5. Build threshold cells for **every** combo in `sweeps` / `sweep.runs` /
   `sequence_combinations` using that engine's key pattern. Extra template cells
   that no longer match TP/PP may be dropped.
6. Write JSON only. Pretty-print with 2-space indent. Point `threshold_json` at
   the sibling threshold **basename** (not a path).
7. Summarize: files written, template sources, topology, leftover `<changeme>`
   fields, and that thresholds are uncalibrated if record-only.

## Naming

If the user gives paths, use them.

Otherwise, under `cvs/input/config_file/{inference|training}/{engine}/`:

- Config: `{gpu_family}_{engine}_{model_slug}_{precision}_{topology}.json`
- Threshold: `{gpu}_{engine}_{model_slug}_{precision}_threshold.json`

Conventions:

- Slug: lowercase, `_` not `/` or `.` (`Kimi-K2.6` → `kimi_k26`). Some atom/vllm
  files use hyphens; match the engine's existing files.
- SGLang configs often use `mi3xx_` for MI300/MI325; thresholds use the GPU
  under test (`mi325_`, `mi300x_`).
- Training megatron/jaxmaxtext/torchtitan often use `mi325x_` / `mi300x_` /
  `mi355x_` on **both** files.

## Cluster placeholders

Keep `<changeme>` (with any trailing hint text the template already has).
Typical fields: `container.image`, `gpu_name`, `server_node_list`,
`benchmark_serv_node`, `master_addr` / `master_address`, `NCCL_IB_HCA`,
`HCA_ID_PREFIX`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `ib_netdev`,
model mounts. `{user-id}` path tokens stay as-is.

Put the Hugging Face id in `server_params.model`, `model.id`, or
`model_params` **only if that template already stores a model id there**. SGLang
skeletons often use `/root/models/<Name>`; keep that shape and use a local
directory derived from the model name (`moonshotai/Kimi-K2.6` →
`/root/models/Kimi-K2.6`). Mention the HF id in `_comment`.

## Thresholds

**Record-only (default unless the user supplies calibrated numbers):**

- Config: `"enforce_thresholds": false`
- Conservative placeholders: `min` / `min_tok_s` near `0` or `1`; `max` /
  `max_ms` copied from the template (already loose) or larger. Do **not** copy
  another model's measured throughput/MFU/accuracy as if they were this model.
- Comment that values are uncalibrated; replace after a lab run before
  enabling enforcement.

**Calibrated:** copy metric names from the template; fill user-supplied values.

Always keep the template's metric **names** and `{ "kind", "value" }` shape
(SGLang `output_throughput_per_sec` vs vLLM `client.*` vs training
`training.*`). Copy accuracy / `BENCH=` / `ACC_ISL=` keys when the config still
lists those tasks; use `0` mins when record-only.

## Model metadata

- `model_num_params`: total parameters (MoE = full size, not activated-only).
- `peak_gpu_tflops`: copy from same-GPU sibling (MI325X SGLang samples use
  `"2615"`).
- Large MoE: keep the large-model `memory_fraction` (often `0.7`).
- Do not add `--trust-remote-code` to SGLang `add_flags` if the launcher
  already injects it (`sglang_distributed_lib` / `sglang_single_lib`).
- Keep `GPU_ARCHS=gfx942` on MI300X/MI325X SGLang `ADD_EXPORT_ENV` when the
  template has it; use the MI355 template's arch list for MI355X.

## Do not

- Write credentials, real hostnames, or cluster-specific NIC lists.
- Use a disaggregated skeleton for unified TP/PP distributed (or the reverse).
- Leave `sweep.runs` combo keys that are missing from the threshold file.
- Enable `enforce_thresholds` on placeholder numbers.
