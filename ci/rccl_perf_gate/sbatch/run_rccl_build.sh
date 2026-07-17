#!/usr/bin/env bash
# RCCL build-only runner: build reference + candidate librccl.so and template
# their lib paths into CONFIG_JSON. No MPI/detect happens here — see
# run_rccl_ab.sh for the 4-node detection step that follows.
#
# Invoked by sbatch/rccl_build.sbatch. Env contract:
#   CANDIDATE_SRC   rocm-systems checkout at PR head
#   BASE_REF        ref to merge-base against for the reference build
#   CONFIG_JSON     ci_detect.json to template (defaults to configs/ci_detect.json)

set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"

: "${CANDIDATE_SRC:?run_rccl_build.sh requires CANDIDATE_SRC (RCCL PR checkout)}"
: "${BASE_REF:?run_rccl_build.sh requires BASE_REF (e.g. origin/develop)}"
export CONFIG_JSON="${CONFIG_JSON:-${RCCL_CI_ROOT}/configs/ci_detect.json}"

[[ -f "${CONFIG_JSON}" ]] || { echo "[ERROR] CONFIG_JSON not found: ${CONFIG_JSON}" >&2; exit 1; }

mkdir -p "${RCCL_CI_ROOT}/logs"

echo "========================================================================"
echo "RCCL build-only CI"
echo "  Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "  Node         : ${SLURMD_NODENAME:-N/A}"
echo "  Config       : ${CONFIG_JSON}"
echo "========================================================================"

build_args=(--candidate-src "${CANDIDATE_SRC}" --base-ref "${BASE_REF}" --config "${CONFIG_JSON}")
[[ -n "${BUILD_OUT:-}" ]] && build_args+=(--out "${BUILD_OUT}")

bash "${RCCL_CI_ROOT}/cvs-sbatch/lib/build_rccl.sh" "${build_args[@]}"
