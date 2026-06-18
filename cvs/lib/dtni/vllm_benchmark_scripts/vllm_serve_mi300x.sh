#!/usr/bin/env bash
# Model-agnostic vLLM server entrypoint for CVS benchmark flows on MI300-class GPUs.
# Pass the checkpoint via MODEL (from server env); extra vLLM CLI flags via "$@".
# Shared canonical copy: cvs.lib.dtni.vllm_benchmark_scripts (see README.md).
# CVS runs: source /tmp/server_env_script.sh; nohup bash this_script ...
# The load client is started separately by CVS (vLLM benchmarks/ driver).
#
# Default: --enforce-eager to reduce HIP "too many resources requested for launch"
# failures during CUDA graph capture on some ROCm + vLLM + AITER stacks.
# Disable with container env: VLLM_ENFORCE_EAGER=0

set -euo pipefail

: "${MODEL:?}" "${PORT:?}" "${TP:?}" "${MAX_MODEL_LEN:?}"

# Bash: ${VAR:-1} does not apply default when VAR is set-but-empty (e.g. from env_dict).
# Normalize whitespace-only to default, then drop the var so vLLM does not log
# "Unknown vLLM environment variable: VLLM_ENFORCE_EAGER" (only the CLI flag is official).
_raw_eager="${VLLM_ENFORCE_EAGER-}"
[[ -z "${_raw_eager//[[:space:]]/}" ]] && _raw_eager=1
ENFORCE_EAGER="${_raw_eager}"
unset VLLM_ENFORCE_EAGER || true

EAGER_FLAG=()
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES) EAGER_FLAG=(--enforce-eager) ;;
  *) ;;
esac

GPU_MEM="${VLLM_GPU_MEMORY_UTIL:-0.92}"

echo "$(date -Is) [cvs vllm_serve_mi300x] vllm serve model=${MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN} port=${PORT} enforce_eager=${ENFORCE_EAGER} eager_flag_count=${#EAGER_FLAG[@]}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --block-size 64 \
  --no-enable-prefix-caching \
  "${EAGER_FLAG[@]}" \
  "$@"
