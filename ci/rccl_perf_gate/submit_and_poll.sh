#!/usr/bin/env bash
##############################################################
# Submit the RCCL A/B regression sbatch job, wait for it to finish, and map the
# job's exit code to this script's exit code so a CI step can gate on it.
#
# Usage:
#   cvs/ci/rccl_perf_gate/submit_and_poll.sh [CONFIG_JSON]
#
# Env overrides:
#   CONFIG_JSON     benchmark config (default: configs/ci_detect.json)
#   SBATCH_SCRIPT   sbatch entry point (default: cvs/ci/rccl_perf_gate/sbatch/rccl_ab.sbatch)
#   POLL_INTERVAL   seconds between status polls (default: 30)
#   MAX_WAIT_SEC    hard timeout in seconds (default: 28800 = 8h)
#   NODELIST        override pinned nodes (passed to sbatch --nodelist)
#
# Exit codes:
#   0   job completed, gate PASS (pytest exit 0)
#   1   job completed, regression detected / pytest failure
#   2   submission or polling error (job never produced a terminal state)
##############################################################
set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
readonly CONFIG_JSON="${1:-${CONFIG_JSON:-${RCCL_CI_ROOT}/configs/ci_detect.json}}"
readonly SBATCH_SCRIPT="${SBATCH_SCRIPT:-${RCCL_CI_ROOT}/cvs/ci/rccl_perf_gate/sbatch/rccl_ab.sbatch}"
readonly POLL_INTERVAL="${POLL_INTERVAL:-30}"
readonly MAX_WAIT_SEC="${MAX_WAIT_SEC:-28800}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

[[ -f "${CONFIG_JSON}" ]]   || { echo "[ERROR] config not found: ${CONFIG_JSON}" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "[ERROR] sbatch script not found: ${SBATCH_SCRIPT}" >&2; exit 2; }

# Forward the optional Phase-4 build vars so the detector job builds ref+cand
# librccl in-allocation before detecting (CI detect mode sets BUILD_RCCL=1).
export_list="ALL,CONFIG_JSON=${CONFIG_JSON}"
if [[ "${BUILD_RCCL:-0}" == "1" ]]; then
  : "${CANDIDATE_SRC:?BUILD_RCCL=1 requires CANDIDATE_SRC}"
  : "${BASE_REF:?BUILD_RCCL=1 requires BASE_REF}"
  export_list+=",BUILD_RCCL=1,CANDIDATE_SRC=${CANDIDATE_SRC},BASE_REF=${BASE_REF}"
fi

submit_args=(--parsable --export="${export_list}")
[[ -n "${NODELIST:-}" ]] && submit_args+=(--nodelist="${NODELIST}")
# Pass reservation only if set — the runner user may not have access to it
[[ -n "${SLURM_RESERVATION:-}" ]] && submit_args+=(--reservation="${SLURM_RESERVATION}")

log "Submitting ${SBATCH_SCRIPT} (config=${CONFIG_JSON})"
JOB_ID="$(sbatch "${submit_args[@]}" "${SBATCH_SCRIPT}")"
JOB_ID="${JOB_ID%%;*}"   # strip cluster suffix from --parsable output
[[ -n "${JOB_ID}" ]] || { echo "[ERROR] sbatch returned no job id" >&2; exit 2; }
log "Submitted job ${JOB_ID}"

# Poll until the job leaves the queue / reaches a terminal state.
elapsed=0
state=""
while (( elapsed < MAX_WAIT_SEC )); do
  # squeue lists only pending/running jobs; empty => terminal.
  if ! squeue -h -j "${JOB_ID}" -o "%T" 2>/dev/null | grep -q .; then
    break
  fi
  state="$(squeue -h -j "${JOB_ID}" -o "%T" 2>/dev/null | head -1)"
  log "job ${JOB_ID} state=${state} (${elapsed}s)"
  sleep "${POLL_INTERVAL}"
  elapsed=$(( elapsed + POLL_INTERVAL ))
done

if (( elapsed >= MAX_WAIT_SEC )); then
  log "[ERROR] timeout after ${MAX_WAIT_SEC}s; cancelling job ${JOB_ID}"
  scancel "${JOB_ID}" 2>/dev/null || true
  exit 2
fi

# Resolve the final exit code from accounting. Retry: sacct can lag briefly.
final_state=""
exit_code=""
for _ in $(seq 1 10); do
  read -r final_state exit_code < <(
    sacct -n -X -j "${JOB_ID}" -o State,ExitCode 2>/dev/null \
      | head -1 | awk '{print $1, $2}'
  ) || true
  [[ -n "${final_state}" ]] && break
  sleep 5
done

log "job ${JOB_ID} final_state=${final_state:-unknown} exit=${exit_code:-unknown}"

# ExitCode is "code:signal"; take the code.
rc="${exit_code%%:*}"
case "${final_state}" in
  COMPLETED) exit "${rc:-0}" ;;
  FAILED)    exit "${rc:-1}" ;;
  *)         log "[ERROR] job ended in non-success state ${final_state}"; exit 2 ;;
esac
