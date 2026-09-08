# Examples

## SGLang distributed Kimi-K2.6 BF16 on MI325X

User prompt:

```
Using the cvs-config-generator skill, generate a sglang distributed config
and threshold for moonshotai/Kimi-K2.6 on MI325X.

Base templates:
- Distributed skeleton: cvs/input/config_file/inference/sglang/mi3xx_sglang_deepseek_r1_0528_distributed.json
- Threshold key pattern: cvs/input/config_file/inference/sglang/mi325_sglang_deepseek_r1_0528_threshold.json

Topology: distributed (unified multi-node TP/PP, NOT disaggregated PD)
- 2 nodes, TP=8 per node, PP=2

Write to:
cvs/input/config_file/inference/sglang/mi3xx_sglang_kimi_k26_bf16_distributed.json
cvs/input/config_file/inference/sglang/mi325_sglang_kimi_k26_bf16_threshold.json

Cluster placeholders (leave as <changeme>):
Thresholds: record-only (conservative placeholders), enforce_thresholds false.
```

Expected:

- Copy the distributed DeepSeek skeleton (not the disaggregated file).
- `nnodes=2`, `tensor_parallelism=8`, `pipeline_parallelism=2`.
- `server_params.model` = `/root/models/Kimi-K2.6`; HF id in `_comment`.
- Sweep and threshold keys use `TP=8,PP=2` (not `PP=1` from the DeepSeek
  threshold sample).
- Cluster fields stay `<changeme>`.
- `enforce_thresholds: false`; conservative min/max placeholders.

## Training Megatron, new model, same GPU

User: generate megatron single config and threshold for Llama-3.1-70B FP8 on
MI325X.

- Template: `cvs/input/config_file/training/megatron/mi325x_megatron_llama-3.1-8b_single.json` plus its `*_threshold.json`.
- Keep `framework`, `gpu_arch`, cell-key shape `MBS=...,GBS=...,PRECISION=...`.
- Retarget `model_name`, tokenizer, `model_size`, container name.
- Leave `training_iterations`, NIC names, `image` as `<changeme>`.
- Record-only unless the user provides measured training metrics.
