---
name: cvs-config-generator
description: >-
  Generates CVS config.json files for single-node, distributed, and
  disaggregated topologies from shipped framework templates in
  cvs/input/config_file/. Use when creating or customizing pytorch_xdit
  (WAN, Flux), atom, vllm, sglang, megatron, torchtitan, or jaxmaxtext
  configs.
---

# CVS Config Generator

## Invocation

Follow this skill when the user asks to generate, derive, or customize CVS config files. Also listed in project [CLAUDE.md](../../../CLAUDE.md).

## Quick start

1. Collect: suite, framework, topology, GPU arch, **HuggingFace model repo ID** (user-provided), cluster size.
2. Read the framework README under `cvs/input/config_file/<suite>/<framework>/` when present.
3. Pick the closest shipped config for the target **topology and workload type** — not a specific model name (see [frameworks.md](frameworks.md)).
4. Apply topology transforms; preserve schema shape — do not invent fields.
5. Set all model fields to the **user's HuggingFace repo ID** (or local path if the user pre-staged weights on cluster nodes).
6. **Write output files** — for SGLang/vLLM/ATOM/training, create **both** config and threshold JSON as siblings.
7. Validate with the checklist in [frameworks.md](frameworks.md).
8. Run `python .claude/skills/cvs-config-generator/scripts/validate_config.py <path>`.

## Model name (required user input)

**Do not pick or assume a model.** Ask the user for the HuggingFace model repo ID (e.g. `org/model-name`) before generating configs.

| Framework | Model field(s) to set from user input |
|-----------|---------------------------------------|
| pytorch_xdit | `config.model_repo` |
| sglang | `benchmark_params.<workload>.model` |
| vllm / atom | model path/id fields in `params` (see framework README) |
| training | model/checkpoint fields per framework README |

For distributed PyTorch XDit, the user may provide a **local path** on every node instead of a HuggingFace repo ID when weights are pre-staged.

## Topology support

| Framework | single | distributed | disaggregated |
|-----------|--------|-------------|---------------|
| pytorch_xdit (WAN, Flux) | yes | yes | no |
| vllm | yes | yes | no |
| atom | yes | yes | no |
| sglang | yes | yes | yes (PD) |
| megatron | yes | yes | no |
| torchtitan | yes | yes | no |
| jaxmaxtext | no | yes | no |

## PyTorch XDit

Canonical templates: `cvs/input/config_file/inference/pytorch_xdit/`

Pick the shipped template that matches **GPU arch, workload type, and topology**:

| Workload type | single template pattern | distributed template pattern |
|---------------|-------------------------|------------------------------|
| WAN I2V | `*_<gpu>_pytorch_xdit_wan*_single.json` | `*_<gpu>_pytorch_xdit_wan*_distributed.json` |
| Flux T2I | `*_<gpu>_pytorch_xdit_flux*_single.json` | `*_<gpu>_pytorch_xdit_flux*_distributed.json` |

Replace `config.model_repo` with the user's HuggingFace repo ID. Do not copy the model value from the shipped template.

PyTorch XDit embeds thresholds in `benchmark_params.*.expected_results` — **one file only**, no sibling threshold JSON.

## Other frameworks

See [frameworks.md](frameworks.md). SGLang, vLLM, and ATOM require **two files**: config + threshold sibling.

## Output naming

```text
{gpu}_{framework}_{user-model-slug}_{topology}.json
```

Derive `{user-model-slug}` from the HuggingFace repo ID the user provided (e.g. `org/my-model` → `my-model`).

Place one topology variant per directory when sibling threshold files are used.

## Additional resources

- Transform rules: [frameworks.md](frameworks.md)
- Worked diffs: [examples.md](examples.md)
