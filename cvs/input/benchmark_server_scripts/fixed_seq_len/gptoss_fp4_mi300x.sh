#!/usr/bin/env bash
# Server-only entrypoint for CVS InferenceMax + vLLM on MI300-class GPUs.
# CVS runs: source /tmp/server_env_script.sh; nohup bash this_script ...
# The benchmark client is started separately by CVS (bench_serving).
#
# Default: --enforce-eager to reduce HIP "too many resources requested for launch"
# failures during CUDA graph capture on some ROCm + vLLM + AITER stacks.
# Disable with container env: VLLM_ENFORCE_EAGER=0

set -euo pipefail

: "${MODEL:?}" "${PORT:?}" "${TP:?}" "${MAX_MODEL_LEN:?}"

ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
EAGER_FLAG=()
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES) EAGER_FLAG=(--enforce-eager) ;;
  *) ;;
esac

GPU_MEM="${VLLM_GPU_MEMORY_UTIL:-0.92}"

echo "$(date -Is) [cvs gptoss_fp4_mi300x] vllm serve model=${MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN} port=${PORT} enforce_eager=${ENFORCE_EAGER:-unset}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --block-size 64 \
  "${EAGER_FLAG[@]}" \
  "$@"
