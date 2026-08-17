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
#   MAX_WAIT_SEC    hard timeout on RUN time, measured from the job's StartTime
#                   (default: 15000 = 4h10m). Queue time is NOT charged here.
#   MAX_QUEUE_SEC   hard timeout on QUEUE time, from submission until the job
#                   starts (default: 14400 = 4h). 0 disables.
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
# These have to nest inside each other, innermost first, or the outer layer
# fires and the inner one never gets to say why:
#
#   rccl_ab.sbatch --time    5h00m   Slurm kills; job state becomes TIMEOUT
#   MAX_WAIT_SEC             5h10m   backstop if slurmctld never enforces it
#   MAX_QUEUE_SEC            4h00m   never started -> report a queue starvation
#   detect timeout-minutes   9h20m   > 4h00 queue + 5h10 run, so it fires last
#
# The run budget went 4h -> 5h when alltoall_perf was re-enabled (2026-08-12).
# The breaker abandons a group after 2 consecutive failures at a 360s
# per-collective timeout, so the all-timeouts bound is
# groups x 2 sides x 2 failures x 360s. At 8 groups that is 3.2h; at 10 it is
# exactly 4.0h, which is not a bound at all when --time is 4h -- Slurm would
# kill the job at the moment the breaker was about to explain itself. A healthy
# 10-group run is ~35min (job 16368), so this is still ~8x headroom.
#
# The previous numbers were 8h / 8h / 24h / 9h, which is not a nesting at all:
# the 24h queue budget sat outside the 9h GitHub timeout, so a job stuck in the
# queue was killed by Actions with a bare "exceeded the maximum execution time"
# and this script's own diagnostic -- the one that says the job never started
# and cancels it -- was unreachable code. Keep this table true if you retune.
readonly MAX_WAIT_SEC="${MAX_WAIT_SEC:-18600}"
readonly MAX_QUEUE_SEC="${MAX_QUEUE_SEC:-14400}"
# squeue talks to slurmctld over the network. A single empty reply is not proof
# the job is gone -- it is equally consistent with a controller restart, an RPC
# timeout, or a momentarily unreachable host. Treating one empty reply as
# terminal is how this script abandons a still-running job and then reads a stale
# report off NFS. Require several consecutive empty replies instead.
readonly EMPTY_POLLS_TO_CONFIRM="${EMPTY_POLLS_TO_CONFIRM:-3}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

[[ -f "${CONFIG_JSON}" ]]   || { echo "[ERROR] config not found: ${CONFIG_JSON}" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "[ERROR] sbatch script not found: ${SBATCH_SCRIPT}" >&2; exit 2; }

# Forward exactly the variables the job needs -- NOT `--export=ALL`.
#
# This script runs inside a GitHub Actions step, so its environment contains
# ACTIONS_RUNTIME_TOKEN, ACTIONS_ID_TOKEN_REQUEST_TOKEN, GITHUB_TOKEN and friends.
# `--export=ALL` copied every one of them into the Slurm job's environment, where
# they are visible to anyone who can run `scontrol show job` or read /proc on the
# compute nodes, and where they get written into the job's environment dump. An
# OIDC request token is enough to mint credentials. Enumerate instead.
declare -a _forward=(
  CONFIG_JSON
  RCCL_CI_ROOT RCCL_CI_WORKSPACE RCCL_CI_RUN_KEY RCCL_CI_CAPABILITY_GATE
  CVS_OUTPUT_BASE_DIR
  GITHUB_RUN_ID GITHUB_RUN_ATTEMPT GITHUB_SHA GITHUB_REF   # provenance only, not secrets
)
if [[ "${BUILD_RCCL:-0}" == "1" ]]; then
  : "${CANDIDATE_SRC:?BUILD_RCCL=1 requires CANDIDATE_SRC}"
  : "${BASE_REF:?BUILD_RCCL=1 requires BASE_REF}"
  _forward+=(BUILD_RCCL CANDIDATE_SRC BASE_REF)
fi
# NONE means "start from a clean environment", then add back only what we list.
# Verified on this cluster (jobs 16126/16127): under --export=NONE Slurm still
# gives the batch script a working login environment -- PATH, HOME, USER and
# /usr/bin/python3 all resolve -- while a variable that was set in the submitting
# shell but not named here does not reach the job.
export_list="NONE"
for _v in "${_forward[@]}"; do
  [[ -n "${!_v:-}" ]] && export_list+=",${_v}=${!_v}"
done

submit_args=(--parsable --export="${export_list}")
[[ -n "${NODELIST:-}" ]] && submit_args+=(--nodelist="${NODELIST}")
# Pass reservation only if set — the runner user may not have access to it
[[ -n "${SLURM_RESERVATION:-}" ]] && submit_args+=(--reservation="${SLURM_RESERVATION}")

log "Submitting ${SBATCH_SCRIPT} (config=${CONFIG_JSON})"
JOB_ID="$(sbatch "${submit_args[@]}" "${SBATCH_SCRIPT}")"
JOB_ID="${JOB_ID%%;*}"   # strip cluster suffix from --parsable output
[[ -n "${JOB_ID}" ]] || { echo "[ERROR] sbatch returned no job id" >&2; exit 2; }
log "Submitted job ${JOB_ID}"

# If this script dies -- runner killed, GitHub job cancelled, Ctrl-C -- the Slurm
# job it submitted keeps running and keeps holding the whole pinned 4-node
# reservation until its 8h wall clock expires. Nothing else on the cluster knows
# to clean it up, because nothing else knows this script owned it. Cancel on the
# way out. EXIT alone is not enough: bash does not run EXIT traps for untrapped
# fatal signals, which is exactly how a runner shutdown arrives.
cleanup_job() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${JOB_ID:-}" && "${JOB_FINISHED:-0}" != "1" ]]; then
    log "[WARN] exiting (rc=${rc}) with job ${JOB_ID} still active; cancelling it so the nodes are released"
    scancel "${JOB_ID}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap cleanup_job EXIT INT TERM HUP

# Poll until the job leaves the queue / reaches a terminal state.
#
# The budget is charged against RUN time, not wall time since submission. Those
# are different numbers whenever the cluster is busy: the sbatch script itself
# asks for --time=08:00:00 measured from StartTime, so a job that queued for two
# hours and then ran for seven was killed here at the 8h mark while Slurm still
# considered it perfectly healthy -- an infra error reported on a job that would
# have finished. Queue time gets its own, much longer, budget.
queued=0
running=0
empty_polls=0
state=""
started=0
while :; do
  raw="$(squeue -h -j "${JOB_ID}" -o "%T" 2>/dev/null | head -1)"
  if [[ -z "${raw}" ]]; then
    empty_polls=$(( empty_polls + 1 ))
    if (( empty_polls >= EMPTY_POLLS_TO_CONFIRM )); then
      log "job ${JOB_ID} absent from squeue for ${empty_polls} consecutive polls; treating as terminal"
      break
    fi
    log "job ${JOB_ID} not in squeue (${empty_polls}/${EMPTY_POLLS_TO_CONFIRM}) — could be a controller blip; re-checking"
  else
    empty_polls=0
    state="${raw}"
    if [[ "${state}" == "RUNNING" || "${state}" == "COMPLETING" ]]; then
      started=1
      log "job ${JOB_ID} state=${state} (running ${running}s)"
    else
      log "job ${JOB_ID} state=${state} (queued ${queued}s)"
    fi
  fi

  if (( started )); then
    if (( running >= MAX_WAIT_SEC )); then
      log "[ERROR] job ${JOB_ID} exceeded the ${MAX_WAIT_SEC}s run budget; cancelling"
      scancel "${JOB_ID}" 2>/dev/null || true
      exit 2
    fi
  elif (( MAX_QUEUE_SEC > 0 && queued >= MAX_QUEUE_SEC )); then
    log "[ERROR] job ${JOB_ID} sat in the queue for ${queued}s (limit ${MAX_QUEUE_SEC}s); cancelling"
    scancel "${JOB_ID}" 2>/dev/null || true
    exit 2
  fi

  sleep "${POLL_INTERVAL}"
  if (( started )); then
    running=$(( running + POLL_INTERVAL ))
  else
    queued=$(( queued + POLL_INTERVAL ))
  fi
done

# Past this point the job is no longer ours to cancel.
JOB_FINISHED=1

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
