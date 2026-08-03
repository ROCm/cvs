# InferenceMax configs (legacy, deprecated)

**Status:** Legacy — retained for OSS users migrating gradually. Scheduled for removal after a deprecation window (target: post Aug 2026 main merge stabilization).

**Replacement:** Use `inferencex_atom_single` configs under `../inferencex_atom_single/` and `cvs run inferencex_atom_single`. See `docs/reference/configuration-files/inferencex_atom.rst`.

## Layout

Monolithic JSON files with top-level `config` + `benchmark_params` (no `schema_version`, no sibling `*_threshold.json`):

| File | GPU | Model |
|------|-----|-------|
| `mi300x_inferencemax_gpt_oss_120b_single.json` | MI300X | GPT-OSS 120B single-node |
| `mi355x_inferencemax_gpt_oss_120b_single.json` | MI355X | GPT-OSS 120B single-node |

## Run (legacy)

```bash
cvs copy-config inference/inferencemax/mi300x_inferencemax_gpt_oss_120b_single.json \
  --output ~/input/config_file/inference/inferencemax/mi300x_inferencemax_gpt_oss_120b_single.json

cvs run inferencemax_gpt_oss_120b_single \
  --cluster_file ~/input/cluster_file/cluster_container.json \
  --config_file ~/input/config_file/inference/inferencemax/mi300x_inferencemax_gpt_oss_120b_single.json
```

The pytest module is `cvs/tests/inference/inferencemax/inferencemax_gpt_oss_120b_single.py`; the job class is `cvs.lib.inference.inference_max.InferenceMaxJob`.

## Migration path

1. Run existing InferenceMax configs to confirm baseline on your cluster.
2. Copy equivalent `inferencex_atom_single` variant configs and cluster JSON.
3. Switch to `cvs run inferencex_atom_single` when ready; thresholds and HTML reporting follow the new suite layout.
