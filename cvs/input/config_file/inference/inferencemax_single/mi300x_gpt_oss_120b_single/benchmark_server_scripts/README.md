# Deprecated location

Canonical **GPT-OSS MI300x** server wrapper (`gptoss_fp4_mi300x.sh`) now lives in the CVS Python package:

`cvs/lib/dtni/vllm_benchmark_scripts/`

Set `host_benchmark_scripts_relpath` to `lib/dtni/vllm_benchmark_scripts` (default in `InferenceMaxJob`) or rely on discovery, which prefers that directory when the script basename matches.

This directory is kept so existing relative paths in old configs still resolve to a folder; it no longer ships a copy of the script.
